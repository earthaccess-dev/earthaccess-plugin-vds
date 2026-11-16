# earthaccess-plugin-vds

> **Plugin preview.** This package is part of the v2.0.0 plugin preview and only
> works with the `plugins` branch of earthaccess
> ([`earthaccess-dev/earthaccess@plugins`](https://github.com/earthaccess-dev/earthaccess/tree/plugins)).
> It is **not on PyPI yet** — see
> [Installing and testing the drafts](https://github.com/earthaccess-dev/earthaccess-plugin-spec/blob/main/docs/install.md).

Virtual datasets for earthaccess: virtualize NASA granules into
[icechunk](https://icechunk.io) (default) or kerchunk-parquet stores, create
new stores, and append to existing ones.

Built on [`earthaccess-plugin-spec`](https://github.com/earthaccess-dev/earthaccess-plugin-spec).

## Install

This package is **not on PyPI yet**. It is part of the v2.0.0 plugin preview and
must be installed together with the `plugins` branch of earthaccess. The full
steps (credentials, notebook deps, sanity check) are in
[Installing and testing the drafts](https://github.com/earthaccess-dev/earthaccess-plugin-spec/blob/main/docs/install.md).

```bash
git clone git@github.com:earthaccess-dev/earthaccess.git
cd earthaccess
git checkout plugins

uv venv
uv pip install -e ".[all]"
# add [kerchunk] and/or [distributed] if you need those extras
uv pip install "earthaccess-plugin-vds[kerchunk] @ git+https://github.com/earthaccess-dev/earthaccess-plugin-vds.git"
```

## Usage

```python
import earthaccess as ea

ea.login()
results = ea.search_data(
    short_name="MUR-JPL-L4-GLOB-v4.1",
    temporal=("2013-01-02", "2013-01-11"),
)

# Create a local icechunk store; the job returns one Asset.
job = ea.plugins.vds.request(results, store="mur_sst.icechunk", concat_dim="time")
asset = job.results()[0]
ds = ea.open(asset.href)

# Append: the same call. The store exists, so append_dim defaults to
# concat_dim and the write goes into the existing repository.
more = ea.search_data(
    short_name="MUR-JPL-L4-GLOB-v4.1",
    temporal=("2013-01-11", "2013-01-21"),
)
asset = ea.plugins.vds.request(
    more, store=asset.href, concat_dim="time"
).results()[0]

# No store → just a virtual dataset (manifest form).
vds = ea.plugins.vds.request(results, concat_dim="time").results()

# Kerchunk-parquet: a references *directory* named *.parquet. Appends write a
# versioned sibling (mur_refs.parquet → mur_refs_v2.parquet) unless
# replace=True, which rewrites in place.
asset = ea.plugins.vds.request(
    more, store="mur_refs.parquet", format="kerchunk", concat_dim="time"
).results()[0]

# Store-to-store merge (both formats): merge a source store into a target.
ea.plugins.vds.request(store="cube_v2", source="cube_v1", append_dim="time")

# Virtualize another plugin's *file* output (e.g. a Harmony subset):
# pass Assets or plain URLs. Derived files have no DMR++ sidecars, so the
# default parser switches to HDFParser. OPeNDAP assets (pydap) are rejected:
# a DAP response is not a byte-range-readable file.
assets = ea.plugins.harmony.request(results, variables=["analysed_sst"]).results()
asset = ea.plugins.vds.request(
    None, assets=assets, store="subset.icechunk", concat_dim="time"
).results()[0]
```

`store=` also accepts an obstore `S3Store` / `LocalStore` (or an
`icechunk.Storage` together with `href=`) as a preconfigured
client-side escape hatch. Credentials are never serialized: `s3://` targets and
granule reads resolve them from the environment where the code runs, so the
same request also works on a Dask cluster whose workers carry the secrets.

## Options

| Option | Meaning |
|---|---|
| `store` | Target store: local path, `s3://` URI, obstore store, `icechunk.Storage`, or `icechunk.Repository`. |
| `href` | Canonical location for the Asset when `store` is a `Storage`/`Repository` (they carry no readable path). |
| `prefix` | Repository prefix when `store` is an obstore `S3Store`. |
| `format` | `"icechunk"` (default) or `"kerchunk"`. |
| `source` | Existing store to merge into `store`. |
| `assets` | File assets/URLs to virtualize instead of search results. |
| `append_dim` | Append dimension; defaults to `concat_dim` when the store exists. |
| `replace` | Kerchunk only: rewrite references in place (stable href). |
| `commit` | Icechunk only: commit the write (default `True`). |
| `repository_config` | Icechunk only: `icechunk.RepositoryConfig` applied when the repository is created (manifest splitting, inline threshold, ...). Ignored on append. |
| `dedupe` | Icechunk only: `"upsert"` (default; skip unchanged, overwrite reprocessed, append new), `"skip"` (never overwrite) or `"off"`. |
| `key` | Coordinate identifying a slot; defaults to `concat_dim`. Only the time coordinate is derivable from CMR metadata. |
| `on_error` | `"raise"` (default) or `"skip"`: record a failed granule, ingest the rest, end the job `PARTIAL`. |
| `virtual_chunk_containers` | Icechunk only: extra `VirtualChunkContainer`s keyed by `url_prefix`, merged over the derived ones (entries win). |
| `access` | `"indirect"` (HTTPS, default) or `"direct"` (S3). |
| `parser` | VirtualiZarr parser name; `"DMRPPParser"` default with automatic `"HDFParser"` fallback. |
| `group`, `concat_dim`, `combine`, `preprocess`, `drop_variables`, `loadable_variables` | VirtualiZarr / xarray combine controls. |
| `load` | Without `store`: return a concrete lazy `Dataset` (parquet round-trip). With `store`: rejected — use `ea.open(asset.href, load=True)`. |
| `tree` | Open a single granule as a `DataTree` (cannot be persisted). |
| `show_progress` | Progress reporting (reserved for the distributed executor). |

## Outputs

Stores that reference `s3://` chunks also record *where to fetch credentials* in
the icechunk repository metadata, under `earthaccess:virtual_chunk_credentials`
(container `url_prefix`, provider, collection concept-id and credentials
endpoint — never secrets). `ea.open_virtual(asset.href)` reads that hint, so a
reader does not need to know the provider; other icechunk clients ignore it.

| Request | `job.output` | `job.results()` |
|---|---|---|
| `store=` given | `assets` | `[Asset(href, access, auth="none", format="icechunk"\|"kerchunk")]` |
| no store, `load=False` | `virtual` | manifest-form `xarray.Dataset` / `DataTree` |
| no store, `load=True` | `xarray` | concrete lazily-loaded `xarray.Dataset` |

## Idempotent ingestion

`request()` is idempotent by default: each granule is keyed by its time step and
the ingested keys are recorded in the repository metadata. Re-running a batch
skips unchanged granules, overwrites a reprocessed one in place (a CMR revision
change is the signal), and appends only genuinely new steps. `dedupe="skip"`
never overwrites; `dedupe="off"` restores plain appends. With `on_error="skip"`
a failed granule is recorded, the rest are ingested, and the job ends `PARTIAL`
— see `job.info()["summary"]` and `job.info()["errors"]`.

## Development

Requires Python 3.12+ (icechunk, virtualizarr and earthaccess all do).

```bash
pip install -e ".[test,kerchunk]"
pytest
ruff check .
mypy
```

The test suite is offline: the granule → virtual-dataset step is faked, while
create/append/reopen run against real local icechunk repositories and
kerchunk-parquet directories.

## License

MIT.
