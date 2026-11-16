"""VDSPlugin: virtualize NASA granules into icechunk / kerchunk-parquet stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from earthaccess_plugin_spec import Asset, Capabilities, Capability, PluginBase
from earthaccess_plugin_spec.granules import peek

__all__ = ["VDSPlugin"]

_OPERATIONS = frozenset({Capability.VIRTUALIZE, Capability.APPEND})
_FORMATS = frozenset({"icechunk", "kerchunk"})

_EXAMPLES = {
    Capability.VIRTUALIZE: (
        'ea.plugins.vds.request(results, concat_dim="time").results() '
        "# manifest-form Dataset"
    ),
    Capability.APPEND: (
        'ea.plugins.vds.request(new_results, store="mur_sst", '
        'concat_dim="time").results()  # store exists -> append'
    ),
}


class VDSPlugin(PluginBase):
    name = "vds"
    needs = frozenset({"umm"})
    auth = "edl"
    output = "assets"
    requires_results = False
    """``source=`` can build a store without new search results."""

    @dataclass
    class Options:
        # --- persistence (store= triggers write/append) ---
        store: Any = None
        """Target store: local path / ``s3://`` URI / obstore store / Storage."""

        prefix: str | None = None
        """Repository prefix when ``store`` is an obstore ``S3Store``."""

        href: str | None = None
        """Canonical location for the Asset when ``store`` is an icechunk
        ``Storage`` or ``Repository`` (neither carries a readable path)."""

        format: Literal["icechunk", "kerchunk"] = "icechunk"
        source: Any = None
        """An existing store (href/Asset) to merge into ``store``."""

        assets: list[Any] | None = None
        """File assets (Asset or URL) to virtualize instead of search results.

        Use this to virtualize the output of another plugin (for example a
        Harmony subset). OPeNDAP assets are rejected: they are responses, not
        byte-range-readable files.
        """

        append_dim: str | None = None
        """Defaults to ``concat_dim`` when the target store already exists."""

        replace: bool = False
        """Kerchunk only: rewrite the existing references in place."""

        commit: bool = True
        """Icechunk only: commit the write to ``main``."""

        repository_config: Any = None
        """Icechunk only: ``icechunk.RepositoryConfig`` used when the repository
        is created (manifest splitting, inline threshold, compression, ...).

        Ignored with a warning when the store already exists: icechunk keeps the
        stored configuration on append."""

        region: str = "us-west-2"
        """Region of the target S3 bucket (and of direct NASA S3 access).

        Defaults to ``us-west-2`` because NASA Earthdata S3 is only reachable
        from there. Pass your bucket's region for a custom S3 target; the
        ``icechunk.Storage`` / obstore ``S3Store`` forms take precedence."""

        dedupe: Literal["upsert", "skip", "off"] = "upsert"
        """Icechunk only: how a batch interacts with keys already in the store.

        ``"upsert"`` (default) skips unchanged granules, overwrites reprocessed
        ones in place, and appends new ones; ``"skip"`` never overwrites;
        ``"off"`` appends everything (the pre-idempotency behaviour)."""

        key: str | None = None
        """Coordinate identifying a slot (defaults to ``concat_dim``).

        Only the time coordinate can be derived from CMR metadata; any other
        value disables dedupe for the batch."""

        verify: Literal["metadata", "etag"] = "metadata"
        """``"etag"`` also probes each granule's origin ETag (one HTTP HEAD per
        granule, opt-in). Catches a file replaced in place without a CMR
        revision bump; probe failures are recorded as ``unverified``."""

        on_error: Literal["raise", "skip"] = "raise"
        """``"skip"`` records a failed granule and ingests the rest (the job
        ends ``PARTIAL``); the default fails the whole batch."""

        virtual_chunk_containers: dict[str, Any] | None = None
        """Icechunk only: extra ``VirtualChunkContainer`` store configs, keyed by
        ``url_prefix`` (``icechunk.s3_store(...)`` / ``icechunk.http_store(...)``).

        Merged with the containers derived from the dataset's chunk references;
        entries here win, so a private bucket or custom endpoint can be used."""

        # --- virtualize ---
        access: Literal["direct", "indirect"] = "indirect"
        parser: Any = "DMRPPParser"
        group: str = "/"
        concat_dim: str | None = None
        combine: Literal["nested", "by_coords"] = "nested"
        preprocess: Any = None
        loadable_variables: list[str] | None = None
        drop_variables: list[str] | None = None
        load: bool = False
        tree: bool = False
        reference_dir: str | None = None
        combine_kwargs: dict[str, Any] = field(default_factory=dict)

        # --- execution ---
        client: Any = None
        """An existing ``distributed.Client``; per-granule opens run there."""

        show_progress: bool | None = None

    def _supports(self, results: Any) -> Capabilities:
        if peek(results) is None:
            return Capabilities(
                None, reason="supply search results to virtualize these granules"
            )
        seen = {_has_data_links(granule) for granule in results}
        if True in seen:
            return Capabilities(
                True,
                operations=_OPERATIONS,
                formats=_FORMATS,
                examples=_EXAMPLES,
            )
        if seen == {False}:
            return Capabilities(False, reason="none of these granules has a data link")
        return Capabilities(None, reason="could not confirm the granules' data links")

    def _request(self, results: Any, options: VDSPlugin.Options) -> Any:
        granules = list(results) if results is not None else []
        output = _validate(granules, options)

        from .progress import ProgressExecutor, ProgressState, VirtualJob

        state = ProgressState()
        count = len(granules) if granules else len(options.assets or [])
        if count:
            state.start("virtualize", total=count)
        if options.client is not None:
            state.dashboard = getattr(options.client, "dashboard_link", None)
        executor = ProgressExecutor(
            state,
            show_progress=options.show_progress,
            client=options.client,
        )

        def work() -> Any:
            try:
                return _run(self._ctx.auth, granules, options, state, executor)
            finally:
                executor.shutdown(wait=False)

        return VirtualJob(
            work,
            plugin=self.name,
            output=output,
            state=state,
            executor=executor,
        )


_NON_VIRTUAL_FORMATS = frozenset({"zarr", "geotiff", "geoparquet", "csv"})


@dataclass(frozen=True)
class _AssetGranule:
    """A granule-shaped view over a plain file URL."""

    href: str

    def data_links(self, access: str = "indirect") -> list[str]:
        return [self.href]

    def __getitem__(self, key: str) -> Any:
        if key == "meta":
            return {"concept-id": self.href, "collection-concept-id": "assets"}
        raise KeyError(key)


def _asset_granules(assets: list[Any]) -> list[_AssetGranule]:
    return [_AssetGranule(str(getattr(asset, "href", asset))) for asset in assets]


def _validate_assets(assets: list[Any]) -> None:
    if not assets:
        msg = "assets= was given but is empty"
        raise ValueError(msg)
    for asset in assets:
        href = str(getattr(asset, "href", asset))
        if not href:
            msg = "every asset needs a non-empty href"
            raise ValueError(msg)
        if isinstance(asset, str):
            continue
        if getattr(asset, "access", None) == "opendap":
            msg = (
                "OPeNDAP assets are responses, not byte-range-readable files; "
                "virtualize the original search results instead of the "
                "constraint-expressed OPeNDAP URLs"
            )
            raise ValueError(msg)
        fmt = getattr(asset, "format", None)
        if fmt in _NON_VIRTUAL_FORMATS:
            msg = (
                f"{fmt!r} assets cannot be virtualized: the parsers read "
                "netCDF/HDF5 files (Harmony netCDF output works; Zarr/GeoTIFF "
                "do not)"
            )
            raise ValueError(msg)


def _has_data_links(granule: Any) -> bool | None:
    """Whether a granule exposes data links (``None`` when it cannot tell)."""

    links = getattr(granule, "data_links", None)
    if links is None:
        return None
    try:
        return bool(links())
    except Exception:  # noqa: BLE001 - a capability check must never crash
        return None


# ---------------------------------------------------------------------------
# request orchestration
# ---------------------------------------------------------------------------


def _icechunk_target_kind(target: Any) -> str | None:
    """Return ``"repository"``/``"storage"`` when *target* is an icechunk object."""

    if target is None:
        return None
    try:
        import icechunk
    except ImportError:  # pragma: no cover - hard dependency
        return None
    if isinstance(target, icechunk.Repository):
        return "repository"
    if isinstance(target, icechunk.Storage):
        return "storage"
    return None


def _validate(granules: list[Any], options: VDSPlugin.Options) -> str:
    """Validate the request eagerly; return the job's output kind."""

    if options.client is not None and not hasattr(options.client, "submit"):
        msg = (
            "client= must be a distributed.Client (or any object with submit()); "
            f"got {type(options.client).__name__}"
        )
        raise TypeError(msg)
    if options.dedupe not in ("upsert", "skip", "off"):
        msg = f"Unsupported dedupe={options.dedupe!r}: use 'upsert', 'skip' or 'off'"
        raise ValueError(msg)
    if options.verify not in ("metadata", "etag"):
        msg = f"Unsupported verify={options.verify!r}: use 'metadata' or 'etag'"
        raise ValueError(msg)
    if options.on_error not in ("raise", "skip"):
        msg = f"Unsupported on_error={options.on_error!r}: use 'raise' or 'skip'"
        raise ValueError(msg)
    if options.on_error == "skip" and options.tree:
        msg = "on_error='skip' is not supported with tree=True"
        raise ValueError(msg)
    if options.format not in _FORMATS:
        msg = f"Unsupported format {options.format!r}: use 'icechunk' or 'kerchunk'"
        raise ValueError(msg)
    if options.format != "icechunk" and (
        options.repository_config is not None
        or options.virtual_chunk_containers is not None
    ):
        msg = (
            "repository_config= and virtual_chunk_containers= only apply to "
            "format='icechunk'"
        )
        raise ValueError(msg)
    kind = _icechunk_target_kind(options.store)
    if kind is not None and options.format != "icechunk":
        msg = f"store= an icechunk {kind} requires format='icechunk'"
        raise ValueError(msg)
    if kind is not None and options.href is None:
        msg = (
            f"store= an icechunk {kind} carries no readable location; pass "
            "href=... so the returned Asset can be reopened"
        )
        raise ValueError(msg)
    if kind == "repository" and options.repository_config is not None:
        msg = (
            "repository_config= cannot be combined with store= an "
            "icechunk.Repository; configure the repository when you create it"
        )
        raise ValueError(msg)
    if options.load and options.store is not None:
        msg = (
            "load=True is only valid without store=; the job returns the Asset, "
            "and materializing is earthaccess.open's job: "
            "ds = ea.open(asset.href, load=True)"
        )
        raise ValueError(msg)
    if options.replace and options.format != "kerchunk":
        msg = "replace=True only applies to format='kerchunk'"
        raise ValueError(msg)
    if options.assets is not None and granules:
        msg = "pass search results or assets=, not both"
        raise ValueError(msg)
    if options.assets is not None and options.source is not None:
        msg = "assets= (raw files) and source= (virtual store) cannot be combined"
        raise ValueError(msg)
    if options.assets is not None:
        _validate_assets(options.assets)
    if options.source is not None and options.store is None:
        msg = "source= merges into a target store; pass store= as well"
        raise ValueError(msg)
    if options.tree and options.store is not None:
        msg = "tree=True cannot be persisted; open a single granule with tree=True only"
        raise ValueError(msg)
    if not granules and not options.assets and options.source is None:
        msg = (
            "no input provided: pass search results, assets= (file URLs), or "
            "source= (a virtual store to merge)"
        )
        raise ValueError(msg)
    if options.store is not None:
        return "assets"
    return "xarray" if options.load else "virtual"


def _run(
    auth: Any,
    granules: list[Any],
    options: VDSPlugin.Options,
    state: Any,
    executor: Any,
) -> Any:
    """Virtualize, optionally merge, and either return a dataset or write a store."""

    from .engine import (
        combine_virtual,
        materialize_coords,
        open_refs_store,
        virtualize,
    )
    from .writer import store_exists, write_kerchunk

    registry = None
    parser = options.parser
    if options.assets is not None:
        source_granules = _asset_granules(options.assets)
        from .credentials import build_registry_for_urls

        registry = build_registry_for_urls([g.href for g in source_granules], auth=auth)
        # Derived files have no DMR++ sidecars; default to the HDF reader.
        if parser == "DMRPPParser":
            parser = "HDFParser"
    else:
        source_granules = None

    def _virtualize(selected: list[Any]) -> Any:
        return virtualize(
            selected,
            access=options.access,
            load=options.load,
            group=options.group,
            concat_dim=options.concat_dim,
            combine=options.combine,
            preprocess=options.preprocess,
            loadable_variables=options.loadable_variables,
            drop_variables=options.drop_variables,
            parser=parser,
            registry=registry,
            reference_dir=options.reference_dir,
            tree=options.tree,
            on_error=options.on_error,
            auth=auth,
            parallel=executor,
            progress=state,
            **options.combine_kwargs,
        )

    plan = None
    upserts: list[tuple[Any, Any]] = []
    append_granules: list[Any] = []
    if _dedupe_applies(options, granules):
        from .idempotency import contiguous_runs, ledger_entries, plan_batch
        from .writer import read_icechunk_state

        index, ledger, resolved = read_icechunk_state(
            options.store,
            options.concat_dim,
            href=options.href,
            prefix=options.prefix,
            region=options.region,
        )
        plan = plan_batch(
            list(granules),
            key=options.key or options.concat_dim,
            index=index,
            ledger=ledger,
            dedupe=options.dedupe,
        )
        unverified = 0
        if options.verify == "etag":
            from .idempotency import apply_etags
            from .probes import probe_etags

            targets: dict[str, Any] = {}
            for granule, key, _fingerprint in [*plan.new, *plan.unchanged]:
                targets[key] = granule
            for granule, key, _fingerprint, _position in plan.reprocessed:
                targets[key] = granule
            etags = probe_etags(targets, auth=auth)
            unverified = apply_etags(plan, etags, index=index, ledger=ledger)

        summary = plan.summary()
        if unverified:
            summary["unverified"] = unverified
        state.set_summary(summary)

        if plan.writes == 0 and not plan.unkeyed:
            if options.verify == "etag":
                from .writer import write_ingested_only

                write_ingested_only(
                    options.store,
                    ledger_entries(plan, include_unchanged=True),
                    href=options.href,
                    prefix=options.prefix,
                    region=options.region,
                )
            return [
                Asset(
                    href=resolved.href,
                    access=resolved.access,
                    auth="none",
                    format="icechunk",
                )
            ]
        append_granules = [g for g, _k, _f in plan.new] + list(plan.unkeyed)
        runs = contiguous_runs(plan.reprocessed)
        upserts = [
            (
                slice(run[0][3], run[-1][3] + 1),
                _virtualize([granule for granule, _k, _f, _i in run]),
            )
            for run in runs
        ]
        plan_entries = ledger_entries(
            plan, include_unchanged=options.verify == "etag"
        )
    else:
        plan_entries = {}

    delta = None
    if plan is not None:
        if append_granules:
            delta = _virtualize(append_granules)
    elif granules or source_granules:
        delta = _virtualize(list(granules) + (source_granules or []))

    if options.store is None:
        return delta

    import xarray as xr

    if delta is not None and not isinstance(delta, xr.Dataset):
        msg = "tree=True cannot be persisted; store= needs a single Dataset"
        raise ValueError(msg)
    if delta is None and not upserts:
        msg = "nothing to write"
        raise ValueError(msg)

    state.set_phase("combine")

    if options.format == "icechunk":
        from .writer import credential_hints

        hint_source = delta if delta is not None else (upserts[0][1] if upserts else None)
        hints = credential_hints(list(granules), hint_source) if granules else {}
        state.set_phase("write")
        return [
            _write_icechunk_sequence(
                delta=delta,
                options=options,
                state=state,
                hints=hints,
                upserts=upserts,
                ingested=plan_entries,
            )
        ]

    target_exists = store_exists(options.store, format="kerchunk")
    dim = options.append_dim or options.concat_dim
    if target_exists and dim is None:
        msg = (
            f"the kerchunk store at {options.store!r} already exists; pass "
            "append_dim=... (or concat_dim=...) to append along a dimension"
        )
        raise ValueError(msg)

    pieces: list[Any] = []
    if options.source is not None:
        pieces.append(open_refs_store(options.source, access=options.access))
    if target_exists:
        pieces.append(open_refs_store(options.store, access=options.access))
    if delta is not None:
        pieces.append(materialize_coords(delta) if pieces else delta)

    vds = combine_virtual(pieces, concat_dim=dim)
    state.set_phase("write")
    return [write_kerchunk(vds, options.store, replace=options.replace)]


def _dedupe_applies(options: VDSPlugin.Options, granules: list[Any]) -> bool:
    """Dedupe only makes sense for an icechunk store fed by CMR granules."""

    return (
        options.format == "icechunk"
        and options.store is not None
        and options.dedupe != "off"
        and options.source is None
        and options.assets is None
        and bool(granules)
    )


def _write_icechunk_sequence(
    *,
    delta: Any,
    options: VDSPlugin.Options,
    state: Any,
    hints: dict[str, dict[str, Any]] | None = None,
    upserts: list[tuple[Any, Any]] | None = None,
    ingested: dict[str, dict[str, Any]] | None = None,
) -> Asset:
    """Write source (if any) then the delta; icechunk appends natively."""

    from .engine import open_refs_store
    from .writer import write_icechunk

    write_kwargs: dict[str, Any] = {
        "append_dim": options.append_dim,
        "concat_dim": options.concat_dim,
        "commit": options.commit,
        "href": options.href,
        "prefix": options.prefix,
        "repository_config": options.repository_config,
        "virtual_chunk_containers": options.virtual_chunk_containers,
        "credential_hints": hints,
        "upserts": upserts,
        "ingested": ingested,
        "region": options.region,
    }

    asset: Asset | None = None
    if options.source is not None:
        source_vds = open_refs_store(options.source, access=options.access)
        asset = write_icechunk(source_vds, options.store, **write_kwargs)
    if delta is not None:
        asset = write_icechunk(delta, options.store, **write_kwargs)
    if asset is None and upserts:
        # only reprocessed keys: region-write them, no append
        asset = write_icechunk(None, options.store, **write_kwargs)
    if asset is None:  # pragma: no cover - validation requires source or granules
        msg = "nothing to write"
        raise ValueError(msg)
    return asset
