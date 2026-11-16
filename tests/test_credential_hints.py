"""Credential-endpoint hints persisted in an icechunk repository's metadata.

A virtual store whose chunks live in a NASA S3 bucket needs the reader to
supply credentials. The writer records *where to get them* (a non-secret
endpoint + collection identity) under a canonical metadata key, so
``ea.open_virtual(asset.href)`` can authorize the containers without the caller
knowing the provider.
"""

from __future__ import annotations

from typing import Any

import icechunk
import pytest
from earthaccess.search import DataGranule
from earthaccess_plugin_spec import JobStatus
from earthaccess_plugin_spec.testing import fake_context

from earthaccess_vds import VDSPlugin
from earthaccess_vds.writer import (
    CREDENTIALS_METADATA_KEY,
    credential_hints,
    write_icechunk,
)
from tests.synthetic import captured_virtualize, make_vds

S3_URL = "s3://podaac-ops-cumulus-protected/a.nc"
OTHER_S3_URL = "s3://ges-disc-cumulus-protected/b.nc"
HTTPS_URL = "https://example.com/a.nc"
ENDPOINT = "https://archive.podaac.earthdatacloud.nasa.gov/s3credentials"


def _granule(
    *,
    provider: str = "POCLOUD",
    collection_id: str | None = "C1996881146-POCLOUD",
    s3: tuple[str, ...] = (S3_URL,),
    https: tuple[str, ...] = (),
    endpoint: str | None = ENDPOINT,
) -> DataGranule:
    urls = [{"URL": url, "Type": "GET DATA VIA DIRECT ACCESS"} for url in s3]
    urls += [{"URL": url, "Type": "GET DATA"} for url in https]
    if endpoint is not None:
        urls.append({"URL": endpoint, "Type": "GET DATA"})
    meta: dict[str, Any] = {"concept-id": "G1-POCLOUD", "provider-id": provider}
    if collection_id is not None:
        meta["collection-concept-id"] = collection_id
    return DataGranule(
        {"meta": meta, "umm": {"RelatedUrls": urls}},
        cloud_hosted=bool(s3),
    )


def _repo(href: str) -> Any:
    return icechunk.Repository.open(icechunk.Storage.new_local_filesystem(href))


def _hints_in(href: str) -> dict[str, Any]:
    spec = (_repo(href).metadata or {}).get(CREDENTIALS_METADATA_KEY) or {}
    return spec.get("containers") or {}


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def test_hints_are_derived_for_s3_prefixes() -> None:
    hints = credential_hints([_granule()], make_vds("2013-01-01", url=S3_URL))

    assert hints == {
        "s3://podaac-ops-cumulus-protected/": {
            "provider": "POCLOUD",
            "collection_concept_id": "C1996881146-POCLOUD",
            "credentials_endpoint": ENDPOINT,
            "region": "us-west-2",
        }
    }


def test_no_hints_for_https_chunks() -> None:
    hints = credential_hints(
        [_granule(s3=(), https=(HTTPS_URL,), endpoint=None)],
        make_vds("2013-01-01", url=HTTPS_URL),
    )

    assert hints == {}


def test_hints_are_per_bucket() -> None:
    vds = make_vds("2013-01-01", url=OTHER_S3_URL)

    hints = credential_hints([_granule()], vds)

    # the granule only carries links in the PODAAC bucket
    assert hints == {}


def test_hints_skip_granules_without_an_endpoint() -> None:
    hints = credential_hints(
        [_granule(endpoint=None, collection_id=None)],
        make_vds("2013-01-01", url=S3_URL),
    )

    assert hints == {}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_hints_are_written_to_repo_metadata(tmp_path: Any) -> None:
    store = str(tmp_path / "sst.icechunk")
    hints = credential_hints([_granule()], make_vds("2013-01-01", url=S3_URL))

    write_icechunk(
        make_vds("2013-01-01", url=S3_URL),
        store,
        concat_dim="time",
        credential_hints=hints,
    )

    spec = _repo(store).metadata[CREDENTIALS_METADATA_KEY]
    assert spec["version"] == 1
    assert spec["containers"] == hints


def test_hints_merge_on_append_and_keep_unrelated_metadata(tmp_path: Any) -> None:
    store = str(tmp_path / "sst.icechunk")
    first = {
        "s3://podaac-ops-cumulus-protected/": {
            "provider": "POCLOUD",
            "collection_concept_id": "C1996881146-POCLOUD",
            "credentials_endpoint": ENDPOINT,
            "region": "us-west-2",
        }
    }
    write_icechunk(
        make_vds("2013-01-01", url=S3_URL),
        store,
        concat_dim="time",
        credential_hints=first,
    )
    _repo(store).update_metadata({"custom": "kept"})

    second = {
        "s3://ges-disc-cumulus-protected/": {
            "provider": "GES_DISC",
            "collection_concept_id": "C1-GES_DISC",
            "credentials_endpoint": "https://goldsmr4.gesdisc.eosdis.nasa.gov/s3credentials",
            "region": "us-west-2",
        }
    }
    write_icechunk(
        make_vds("2013-01-02", url=S3_URL),
        store,
        append_dim="time",
        credential_hints=second,
    )

    metadata = _repo(store).metadata
    assert metadata["custom"] == "kept"
    assert set(_hints_in(store)) == set(first) | set(second)


def test_no_metadata_without_hints(tmp_path: Any) -> None:
    store = str(tmp_path / "sst.icechunk")

    write_icechunk(make_vds("2013-01-01"), store, concat_dim="time")

    assert CREDENTIALS_METADATA_KEY not in (_repo(store).metadata or {})


# ---------------------------------------------------------------------------
# Through the plugin
# ---------------------------------------------------------------------------


def test_request_persists_the_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    captured_virtualize(monkeypatch, make_vds("2013-01-01", url=S3_URL))
    granule = _granule(https=(HTTPS_URL,))

    job = VDSPlugin(fake_context()).request(
        [granule], store=store, concat_dim="time"
    )
    status = job.wait(timeout=60)

    assert status is JobStatus.SUCCEEDED, job.info()
    hints = _hints_in(store)
    assert "s3://podaac-ops-cumulus-protected/" in hints
    assert hints["s3://podaac-ops-cumulus-protected/"]["credentials_endpoint"] == ENDPOINT
