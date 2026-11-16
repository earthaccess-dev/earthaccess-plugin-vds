"""Advanced icechunk configuration for the vds plugin.

Three additive escape hatches are exercised here:

- ``repository_config=``          apply an ``icechunk.RepositoryConfig`` on create
- ``store=<icechunk.Repository>`` use a repository the caller already created
- ``virtual_chunk_containers=``   override/merge the derived virtual chunk containers
"""

from __future__ import annotations

from typing import Any

import icechunk
import pytest
import xarray as xr
from earthaccess_plugin_spec import JobStatus
from earthaccess_plugin_spec.testing import fake_context, sample_result_set

from earthaccess_vds import VDSPlugin
from tests.synthetic import captured_virtualize, make_vds


def _request(monkeypatch: pytest.MonkeyPatch, vds: xr.Dataset, **options: Any) -> Any:
    captured_virtualize(monkeypatch, vds)
    job = VDSPlugin(fake_context()).request(sample_result_set(), **options)
    return job, job.wait(timeout=60)


def _storage(href: str) -> Any:
    return icechunk.Storage.new_local_filesystem(href)


def _repo(href: str) -> Any:
    return icechunk.Repository.open(_storage(href))


def _open(href: str) -> xr.Dataset:
    return xr.open_zarr(_repo(href).readonly_session("main").store, consolidated=False)


def _custom_config() -> Any:
    """A RepositoryConfig with non-default manifest splitting + inline threshold."""

    config = icechunk.RepositoryConfig.default()
    config.inline_chunk_threshold_bytes = 7
    config.manifest = icechunk.ManifestConfig(
        splitting=icechunk.ManifestSplittingConfig.from_dict(
            {
                icechunk.ManifestSplitCondition.name_matches("sst"): {
                    icechunk.ManifestSplitDimCondition.DimensionName("time"): 4,
                },
            },
        ),
    )
    return config


# ---------------------------------------------------------------------------
# repository_config=
# ---------------------------------------------------------------------------


def test_repository_config_is_applied_on_create(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    job, status = _request(
        monkeypatch,
        make_vds("2013-01-01"),
        store=store,
        concat_dim="time",
        repository_config=_custom_config(),
    )

    assert status is JobStatus.SUCCEEDED, job.info()
    config = _repo(store).config
    assert config.inline_chunk_threshold_bytes == 7
    assert config.manifest is not None
    assert "time" in str(config.manifest.splitting.split_sizes)
    # the derived container is still merged in
    assert "https://example.com/" in (config.virtual_chunk_containers or {})
    assert _open(store).sizes["time"] == 1


def test_repository_config_does_not_mutate_the_caller_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = _custom_config()
    _request(
        monkeypatch,
        make_vds("2013-01-01"),
        store=str(tmp_path / "sst.icechunk"),
        concat_dim="time",
        repository_config=config,
    )

    assert config.inline_chunk_threshold_bytes == 7
    assert not (config.virtual_chunk_containers or {})


def test_repository_config_is_ignored_on_append(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    _request(monkeypatch, make_vds("2013-01-01"), store=store, concat_dim="time")

    with pytest.warns(UserWarning, match="repository_config"):
        job, status = _request(
            monkeypatch,
            make_vds("2013-01-02"),
            store=store,
            concat_dim="time",
            repository_config=_custom_config(),
        )

    assert status is JobStatus.SUCCEEDED, job.info()
    assert _open(store).sizes["time"] == 2
    # the stored configuration wins on append
    assert _repo(store).config.inline_chunk_threshold_bytes is None


def test_repository_config_requires_icechunk_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    captured_virtualize(monkeypatch, make_vds("2013-01-01"))
    with pytest.raises(ValueError, match="icechunk"):
        VDSPlugin(fake_context()).request(
            sample_result_set(),
            store=str(tmp_path / "refs.parquet"),
            format="kerchunk",
            repository_config=_custom_config(),
        )


# ---------------------------------------------------------------------------
# store=<icechunk.Repository>
# ---------------------------------------------------------------------------


def test_store_accepts_an_empty_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    repo = icechunk.Repository.create(_storage(store))

    job, status = _request(
        monkeypatch,
        make_vds("2013-01-01"),
        store=repo,
        href=store,
        concat_dim="time",
    )

    assert status is JobStatus.SUCCEEDED, job.info()
    (asset,) = job.results()
    assert asset.href == store
    assert _open(store).sizes["time"] == 1
    # containers are merged into the caller's repository
    assert "https://example.com/" in (
        _repo(store).config.virtual_chunk_containers or {}
    )


def test_store_repository_appends_to_existing_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    _request(monkeypatch, make_vds("2013-01-01"), store=store, concat_dim="time")

    job, status = _request(
        monkeypatch,
        make_vds("2013-01-02"),
        store=_repo(store),
        href=store,
        concat_dim="time",
    )

    assert status is JobStatus.SUCCEEDED, job.info()
    assert _open(store).sizes["time"] == 2


def test_store_repository_requires_href(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    captured_virtualize(monkeypatch, make_vds("2013-01-01"))
    repo = icechunk.Repository.create(_storage(str(tmp_path / "sst.icechunk")))

    with pytest.raises(ValueError, match="href"):
        VDSPlugin(fake_context()).request(
            sample_result_set(), store=repo, concat_dim="time"
        )


def test_store_repository_rejects_repository_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    captured_virtualize(monkeypatch, make_vds("2013-01-01"))
    store = str(tmp_path / "sst.icechunk")
    repo = icechunk.Repository.create(_storage(store))

    with pytest.raises(ValueError, match="repository_config"):
        VDSPlugin(fake_context()).request(
            sample_result_set(),
            store=repo,
            href=store,
            concat_dim="time",
            repository_config=_custom_config(),
        )


def test_store_accepts_an_icechunk_storage_with_href(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    job, status = _request(
        monkeypatch,
        make_vds("2013-01-01"),
        store=_storage(store),
        href=store,
        concat_dim="time",
    )

    assert status is JobStatus.SUCCEEDED, job.info()
    assert job.results()[0].href == store
    assert _open(store).sizes["time"] == 1


# ---------------------------------------------------------------------------
# virtual_chunk_containers=
# ---------------------------------------------------------------------------


def test_virtual_chunk_containers_are_merged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    job, status = _request(
        monkeypatch,
        make_vds("2013-01-01"),
        store=store,
        concat_dim="time",
        virtual_chunk_containers={
            "https://other.example.com/": icechunk.http_store(),
        },
    )

    assert status is JobStatus.SUCCEEDED, job.info()
    vccs = _repo(store).config.virtual_chunk_containers or {}
    assert set(vccs) == {"https://example.com/", "https://other.example.com/"}


def test_virtual_chunk_containers_override_derived(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    job, status = _request(
        monkeypatch,
        make_vds("2013-01-01"),
        store=store,
        concat_dim="time",
        virtual_chunk_containers={
            "https://example.com/": icechunk.http_store(headers={"X-Test": "1"}),
        },
    )

    assert status is JobStatus.SUCCEEDED, job.info()
    container = (_repo(store).config.virtual_chunk_containers or {})[
        "https://example.com/"
    ]
    assert "X-Test" in str(container)


def test_virtual_chunk_containers_are_merged_on_append(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    _request(monkeypatch, make_vds("2013-01-01"), store=store, concat_dim="time")

    job, status = _request(
        monkeypatch,
        make_vds("2013-01-02"),
        store=store,
        concat_dim="time",
        virtual_chunk_containers={
            "https://other.example.com/": icechunk.http_store(),
        },
    )

    assert status is JobStatus.SUCCEEDED, job.info()
    vccs = _repo(store).config.virtual_chunk_containers or {}
    assert set(vccs) == {"https://example.com/", "https://other.example.com/"}
    assert _open(store).sizes["time"] == 2
