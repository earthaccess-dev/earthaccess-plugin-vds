"""Writers for virtual stores: icechunk (default) and kerchunk-parquet.

Both writers take a manifest-form dataset (``load=False``) and return the
:class:`~earthaccess_plugin_spec.Asset` that identifies the written store, so
``earthaccess.open(asset.href)`` can route it. No credentials are serialized:
icechunk resolves them from the environment at the write site, and kerchunk
writes through fsspec's own env-based credentials.

Icechunk appends are O(delta): the delta is written straight into the existing
repository. Kerchunk has no append operation, so its append is a
read-combine-rewrite handled by the caller (see ``engine.open_refs_store``);
these writers only ever write one complete reference set.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from earthaccess_plugin_spec import Asset

from .idempotency import INGESTED_METADATA_KEY
from .regions import NASA_REGION
from .storage import (
    resolve_icechunk_target,
    resolve_kerchunk_target,
)

__all__ = [
    "derive_vccs_from_vds",
    "store_exists",
    "write_icechunk",
    "write_kerchunk",
    "write_virtual_store",
]

StoreFormat = Literal["icechunk", "kerchunk"]


CREDENTIALS_METADATA_KEY = "earthaccess:virtual_chunk_credentials"
"""Repository metadata key recording where to fetch credentials for S3 chunks.

The value is non-secret and namespaced so other icechunk clients can ignore it::

    {"version": 1, "containers": {"<url_prefix>": {
        "provider": ..., "collection_concept_id": ...,
        "credentials_endpoint": ..., "region": ...}}}
"""


def _chunk_prefixes(vds: Any) -> dict[str, str]:
    """Map every ``scheme://host/`` in *vds*' chunk references to its scheme."""

    try:
        from virtualizarr.manifests import ManifestArray
    except ImportError as exc:  # pragma: no cover - hard dependency
        msg = "earthaccess_vds requires `virtualizarr>=2.8`"
        raise ImportError(msg) from exc

    prefixes: dict[str, str] = {}
    for variable in vds.variables.values():
        data = variable.data
        if not isinstance(data, ManifestArray):
            continue
        for path in data.manifest.iter_nonempty_paths():
            from urllib.parse import urlparse

            parsed = urlparse(path)
            if parsed.scheme in ("s3", "https", "http"):
                prefixes[f"{parsed.scheme}://{parsed.netloc}/"] = parsed.scheme
    return prefixes


def s3_prefixes_from_vds(vds: Any) -> set[str]:
    """Return the ``s3://bucket/`` prefixes referenced by *vds*' chunks."""

    return {p for p, scheme in _chunk_prefixes(vds).items() if scheme == "s3"}


def credential_hints(
    granules: Sequence[Any], vds: Any
) -> dict[str, dict[str, Any]]:
    """Where to fetch credentials for each S3 virtual chunk container in *vds*.

    Derived from the source granules' metadata (granule, then collection) and
    keyed by the container's exact ``url_prefix``. Nothing secret is returned:
    only the endpoint, provider and collection identity, which are public.
    """

    from urllib.parse import urlparse

    from .credentials import get_granule_credentials_endpoint_and_region

    hints: dict[str, dict[str, Any]] = {}
    for prefix in sorted(s3_prefixes_from_vds(vds)):
        bucket = urlparse(prefix).netloc
        granule = _granule_for_bucket(granules, bucket)
        if granule is None:
            continue
        try:
            endpoint, region = get_granule_credentials_endpoint_and_region(granule)
        except (KeyError, ValueError):
            continue
        meta = granule.get("meta", {})
        hints[prefix] = {
            "provider": meta.get("provider-id"),
            "collection_concept_id": meta.get("collection-concept-id"),
            "credentials_endpoint": endpoint,
            "region": region,
        }
    return hints


def _granule_for_bucket(granules: Sequence[Any], bucket: str) -> Any:
    """Return the first granule with an S3 data link in *bucket*."""

    from urllib.parse import urlparse

    for granule in granules:
        try:
            links = granule.data_links(access="s3")
        except Exception:  # noqa: BLE001 - a hint must never fail a write
            continue
        if any(urlparse(link).netloc == bucket for link in links):
            return granule
    return None


def _write_credential_hints(repo: Any, hints: dict[str, dict[str, Any]]) -> None:
    """Merge *hints* into the repository metadata (never replaces other keys)."""

    if not hints:
        return
    metadata = repo.metadata or {}
    spec = metadata.get(CREDENTIALS_METADATA_KEY) or {}
    containers = dict(spec.get("containers") or {})
    containers.update(hints)
    repo.update_metadata(
        {CREDENTIALS_METADATA_KEY: {"version": 1, "containers": containers}}
    )


def derive_vccs_from_many(datasets: Any) -> dict[str, Any]:
    """Derive the containers needed for several virtual datasets at once."""

    import icechunk

    prefixes: dict[str, str] = {}
    for dataset in datasets:
        if dataset is not None:
            prefixes.update(_chunk_prefixes(dataset))
    if not prefixes:
        msg = (
            "writing a virtual store requires a virtual dataset (load=False) "
            "backed by ManifestArray variables; no virtual chunk references "
            "were found"
        )
        raise ValueError(msg)

    vccs: dict[str, Any] = {}
    for prefix, scheme in prefixes.items():
        if scheme == "s3":
            vccs[prefix] = icechunk.s3_store(region="us-west-2")
        else:
            vccs[prefix] = icechunk.http_store()
    return vccs


def derive_vccs_from_vds(vds: Any) -> dict[str, Any]:
    """Derive the virtual chunk containers needed to write *vds*.

    Walks every ``ManifestArray`` in the dataset and collects one container per
    unique ``scheme://host/`` prefix in its chunk references. ``s3://``
    containers use the default NASA cloud region; ``http(s)://`` containers
    need no credentials at write time.
    """

    import icechunk

    prefixes = _chunk_prefixes(vds)
    if not prefixes:
        msg = (
            "writing a virtual store requires a virtual dataset (load=False) "
            "backed by ManifestArray variables; no virtual chunk references "
            "were found"
        )
        raise ValueError(msg)

    vccs: dict[str, Any] = {}
    for prefix, scheme in prefixes.items():
        if scheme == "s3":
            vccs[prefix] = icechunk.s3_store(region="us-west-2")
        else:
            vccs[prefix] = icechunk.http_store()
    return vccs


def read_icechunk_state(
    target: Any,
    concat_dim: str | None,
    *,
    href: str | None = None,
    prefix: str | None = None,
    region: str = NASA_REGION,
) -> tuple[dict[str, int], dict[str, dict[str, Any]], Any]:
    """Read ``(key -> index, ledger, resolved target)`` from an existing store.

    The index maps the concat-coordinate values to their position in the store,
    so a reprocessed key can be overwritten in place. The ledger records which
    keys were ingested (see :data:`~earthaccess_vds.idempotency.INGESTED_METADATA_KEY`).
    A missing store returns empty structures.
    """

    import icechunk
    import pandas as pd
    import xarray as xr

    resolved = resolve_icechunk_target(
        target, href=href, prefix=prefix, region=region
    )
    repo = resolved.repository
    if repo is None:
        if not icechunk.Repository.exists(resolved.storage):
            return {}, {}, resolved
        repo = icechunk.Repository.open(resolved.storage)
    elif repo.lookup_branch("main") == _EMPTY_SNAPSHOT_ID:
        return {}, {}, resolved

    metadata = repo.metadata or {}
    ledger = dict((metadata.get(INGESTED_METADATA_KEY) or {}).get("keys") or {})

    index: dict[str, int] = {}
    if concat_dim:
        try:
            dataset = xr.open_zarr(
                repo.readonly_session("main").store,
                consolidated=False,
            )
        except Exception:  # noqa: BLE001 - unreadable store falls back to append
            return {}, ledger, resolved
        if concat_dim in dataset.coords:
            values = dataset[concat_dim].values
            index = {
                pd.Timestamp(value).isoformat(): position
                for position, value in enumerate(values)
            }
    return index, ledger, resolved


def store_exists(target: Any, *, format: StoreFormat = "icechunk") -> bool:
    """Return whether a store already exists at *target*."""

    if format == "icechunk":
        import icechunk

        resolved = resolve_icechunk_target(target)
        return bool(icechunk.Repository.exists(resolved.storage))
    if format == "kerchunk":
        resolved = resolve_kerchunk_target(target)
        fs, _ = _fsspec_for(resolved.href)
        return bool(fs.exists(resolved.href))
    msg = f"Unsupported format {format!r}: expected 'icechunk' or 'kerchunk'"
    raise ValueError(msg)


def write_virtual_store(
    vds: Any,
    target: Any,
    *,
    format: StoreFormat = "icechunk",
    append_dim: str | None = None,
    concat_dim: str | None = None,
    commit: bool = True,
    replace: bool = False,
) -> Asset:
    """Write *vds* to *target* and return the result Asset."""

    if format == "icechunk":
        return write_icechunk(
            vds,
            target,
            append_dim=append_dim,
            concat_dim=concat_dim,
            commit=commit,
        )
    if format == "kerchunk":
        return write_kerchunk(vds, target, replace=replace)
    msg = f"Unsupported format {format!r}: expected 'icechunk' or 'kerchunk'"
    raise ValueError(msg)


_EMPTY_SNAPSHOT_ID = "1CECHNKREP0F1RSTCMT0"
"""Snapshot icechunk reports for a repository with no commits yet."""


def write_icechunk(
    vds: Any,
    target: Any,
    *,
    append_dim: str | None = None,
    concat_dim: str | None = None,
    commit: bool = True,
    href: str | None = None,
    prefix: str | None = None,
    repository_config: Any = None,
    virtual_chunk_containers: dict[str, Any] | None = None,
    credential_hints: dict[str, dict[str, Any]] | None = None,
    upserts: list[tuple[slice, Any]] | None = None,
    ingested: dict[str, dict[str, Any]] | None = None,
    region: str = NASA_REGION,
) -> Asset:
    """Create or append to an icechunk repository at *target*.

    Create-vs-append is decided by the repository's existence (or, for a
    caller-supplied :class:`icechunk.Repository`, by whether it already has
    commits). On append the dataset is written along ``append_dim`` (defaulting
    to ``concat_dim``), and any virtual chunk container missing from the stored
    configuration is merged in explicitly — a config passed to
    ``open_or_create`` is ignored for an existing repository.

    ``repository_config`` is only applied when the repository is created; on
    append the stored configuration wins and a warning is emitted.
    ``virtual_chunk_containers`` overrides the containers derived from the
    dataset on both paths. ``credential_hints`` and ``ingested`` are merged
    into the repository metadata (see :data:`CREDENTIALS_METADATA_KEY` and
    :data:`~earthaccess_vds.idempotency.INGESTED_METADATA_KEY`).

    ``upserts`` rewrites existing time steps in place: each ``(slice, vds)``
    pair is written to that slice of ``concat_dim`` instead of being appended,
    so a reprocessed granule replaces its predecessor without duplicating the
    coordinate. ``vds`` may be ``None`` when the batch only upserts.
    """

    import icechunk

    resolved = resolve_icechunk_target(
        target, href=href, prefix=prefix, region=region
    )
    pieces = [piece for _slice, piece in (upserts or [])]
    derived = derive_vccs_from_many([vds, *pieces])
    explicit = virtual_chunk_containers or {}

    if resolved.repository is not None:
        return _write_into_repository(
            vds,
            resolved,
            append_dim=append_dim,
            concat_dim=concat_dim,
            commit=commit,
            derived=derived,
            explicit=explicit,
            credential_hints=credential_hints,
            upserts=upserts,
            ingested=ingested,
            region=region,
        )

    existed = icechunk.Repository.exists(resolved.storage)
    if existed:
        if repository_config is not None:
            warnings.warn(
                "repository_config= is ignored: the store already exists and "
                "icechunk keeps the stored configuration on append.",
                UserWarning,
                stacklevel=2,
            )
        append_dim = append_dim or concat_dim
        if append_dim is None and vds is not None:
            msg = (
                f"the icechunk store at {resolved.href!r} already exists; pass "
                "append_dim=... (or concat_dim=... to append along the "
                "combine dimension)"
            )
            raise ValueError(msg)
        config = icechunk.Repository.fetch_config(resolved.storage)
        if config is None:  # pragma: no cover - exists() said otherwise
            config = icechunk.RepositoryConfig.default()
        changed = _merge_vccs(config, derived)
        changed = _set_vccs(config, explicit) or changed
    else:
        base = (
            repository_config
            if repository_config is not None
            else icechunk.RepositoryConfig.default()
        )
        # merge() returns a new config, so the caller's object is never mutated
        config = icechunk.RepositoryConfig.default().merge(base)
        _merge_vccs(config, derived)
        _set_vccs(config, explicit)
        changed = False

    repo = icechunk.Repository.open_or_create(resolved.storage, config=config)
    if existed and changed:
        # The stored config wins on open; persist the merged containers so
        # future readers can authorize every host this store references.
        repo.save_config()

    session = repo.writable_session("main")
    _write_session(
        session,
        vds,
        upserts=upserts,
        append_dim=append_dim,
        region_dim=concat_dim or append_dim,
        region=region,
    )
    if commit:
        session.commit("vds append" if append_dim else "vds create")
    if credential_hints:
        _write_credential_hints(repo, credential_hints)
    if ingested:
        _write_ingested(repo, ingested)

    return Asset(
        href=resolved.href,
        access=resolved.access,
        auth="none",
        format="icechunk",
    )


def _write_into_repository(
    vds: Any,
    resolved: Any,
    *,
    append_dim: str | None,
    concat_dim: str | None,
    commit: bool,
    derived: dict[str, Any],
    explicit: dict[str, Any],
    credential_hints: dict[str, dict[str, Any]] | None,
    upserts: list[tuple[slice, Any]] | None,
    ingested: dict[str, dict[str, Any]] | None,
    region: str = NASA_REGION,
) -> Asset:
    """Write into a repository the caller already opened (never created here)."""


    repo = resolved.repository
    config = repo.config
    changed = _merge_vccs(config, derived)
    changed = _set_vccs(config, explicit) or changed
    if changed:
        # ``repo.config`` is a copy: reopen with it so save_config persists it.
        repo = repo.reopen(config=config)
        repo.save_config()

    empty = repo.lookup_branch("main") == _EMPTY_SNAPSHOT_ID
    if not empty:
        append_dim = append_dim or concat_dim
        if append_dim is None and vds is not None:
            msg = (
                f"the icechunk repository at {resolved.href!r} already has data; "
                "pass append_dim=... (or concat_dim=... to append along the "
                "combine dimension)"
            )
            raise ValueError(msg)

    session = repo.writable_session("main")
    _write_session(
        session,
        vds,
        upserts=upserts,
        append_dim=append_dim,
        region_dim=concat_dim or append_dim,
        region=region,
    )
    if commit:
        session.commit("vds create" if empty else "vds append")
    if credential_hints:
        _write_credential_hints(repo, credential_hints)
    if ingested:
        _write_ingested(repo, ingested)

    return Asset(
        href=resolved.href,
        access=resolved.access,
        auth="none",
        format="icechunk",
    )


def _write_session(
    session: Any,
    vds: Any,
    *,
    upserts: list[tuple[slice, Any]] | None,
    append_dim: str | None,
    region_dim: str | None,
    region: str = NASA_REGION,
) -> None:
    """Region-write the upserts, then append the new dataset (one session)."""

    try:
        for index_slice, piece in upserts or []:
            if region_dim is None:  # pragma: no cover - guarded by the callers
                msg = "upserting requires concat_dim to locate the existing keys"
                raise ValueError(msg)
            piece.virtualize.to_icechunk(
                session.store,
                region={region_dim: index_slice},
            )
        if vds is not None:
            vds.virtualize.to_icechunk(session.store, append_dim=append_dim)
    except Exception as exc:
        hint = _region_hint(exc, region)
        if hint is not None:
            raise hint from exc
        raise


def _region_hint(exc: BaseException, region: str) -> ValueError | None:
    """Turn a region-looking icechunk failure into an actionable error."""

    message = str(exc).lower()
    if not any(token in message for token in ("301", "permanentredirect", "region")):
        return None
    return ValueError(
        "the icechunk write failed with a region-looking error; the target "
        f"bucket may live in another region. Pass region=... (current: {region})."
    )


def write_ingested_only(
    target: Any,
    entries: dict[str, dict[str, Any]],
    *,
    href: str | None = None,
    prefix: str | None = None,
    region: str = NASA_REGION,
) -> None:
    """Merge ledger entries without touching the data (probe-only runs)."""

    if not entries:
        return
    import icechunk

    resolved = resolve_icechunk_target(
        target, href=href, prefix=prefix, region=region
    )
    repo = resolved.repository
    if repo is None:
        if not icechunk.Repository.exists(resolved.storage):
            return
        repo = icechunk.Repository.open(resolved.storage)
    _write_ingested(repo, entries)


def _write_ingested(repo: Any, entries: dict[str, dict[str, Any]]) -> None:
    """Merge ingested-key fingerprints into the repository metadata."""

    if not entries:
        return
    metadata = repo.metadata or {}
    spec = metadata.get(INGESTED_METADATA_KEY) or {}
    keys = dict(spec.get("keys") or {})
    keys.update(entries)
    repo.update_metadata({INGESTED_METADATA_KEY: {"version": 1, "keys": keys}})


def _set_vccs(config: Any, vccs: dict[str, Any]) -> bool:
    """Set explicit containers on *config* (they win); return whether it changed."""

    import icechunk

    if not vccs:
        return False
    existing = config.virtual_chunk_containers or {}
    changed = False
    for url_prefix, store_config in vccs.items():
        current = existing.get(url_prefix)
        if current is not None and str(current.store) == str(store_config):
            continue
        config.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(url_prefix, store_config),
        )
        changed = True
    return changed


def write_kerchunk(
    vds: Any,
    target: Any,
    *,
    replace: bool = False,
    storage_options: dict[str, Any] | None = None,
) -> Asset:
    """Write *vds* as a kerchunk-parquet references directory.

    The target must be named ``*.parquet`` — the references are a *directory*
    whose name carries the suffix, which is what URI sniffing matches. Writing
    over an existing store creates a versioned sibling (``refs.parquet`` →
    ``refs_v2.parquet``) unless ``replace=True``, in which case the existing
    references are removed and rewritten in place (stable href).
    """

    from kerchunk.df import refs_to_dataframe

    resolved = resolve_kerchunk_target(target)
    if not resolved.href.endswith(".parquet"):
        msg = (
            f"a kerchunk-parquet store target must end in '.parquet' "
            f"(got {resolved.href!r}); the references are written as a "
            "directory named *.parquet"
        )
        raise ValueError(msg)

    fs, _ = _fsspec_for(resolved.href, storage_options=storage_options)
    href = resolved.href
    exists = fs.exists(href)
    if exists and replace:
        fs.rm(href, recursive=True)
    elif exists:
        href = _next_version(href)

    refs = vds.virtualize.to_kerchunk(format="dict")
    refs_to_dataframe(refs, href, storage_options=storage_options)

    return Asset(href=href, access=resolved.access, auth="none", format="kerchunk")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _merge_vccs(config: Any, vccs: dict[str, Any]) -> bool:
    """Add missing containers to *config*; return whether anything changed."""

    import icechunk

    existing = config.virtual_chunk_containers or {}
    missing = {prefix: store for prefix, store in vccs.items() if prefix not in existing}
    for prefix, store_config in missing.items():
        config.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(prefix, store_config),
        )
    return bool(missing)


def _fsspec_for(
    href: str,
    *,
    storage_options: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    import fsspec

    fs, _, paths = fsspec.get_fs_token_paths(href, **(storage_options or {}))
    return fs, paths[0] if paths else href


_VERSIONED = re.compile(r"^(?P<base>.*?)(?:_v(?P<version>\d+))?(?P<suffix>\.parquet)$")


def _next_version(href: str) -> str:
    """``refs.parquet`` → ``refs_v2.parquet``; ``refs_v2.parquet`` → ``refs_v3.parquet``."""

    match = _VERSIONED.match(href)
    if match is None:  # pragma: no cover - validated by the caller
        return href
    version = int(match.group("version") or 1) + 1
    return f"{match.group('base')}_v{version}{match.group('suffix')}"


def _as_href(target: Any) -> str:
    if isinstance(target, Path):
        return str(target)
    return str(target)
