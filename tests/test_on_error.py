"""``on_error="skip"``: record a failed granule and ingest the rest."""

from __future__ import annotations

from typing import Any

import pytest
from earthaccess.search import DataGranule
from earthaccess_plugin_spec import JobStatus
from earthaccess_plugin_spec.testing import fake_context

from earthaccess_vds import VDSPlugin
from tests.synthetic import make_vds

GOOD = "https://a.example/good.nc"
BAD = "https://a.example/bad.nc"


def _granule(href: str, time: str, concept_id: str) -> DataGranule:
    return DataGranule(
        {
            "meta": {
                "concept-id": concept_id,
                "collection-concept-id": "C1-TEST",
                "provider-id": "POCLOUD",
            },
            "umm": {
                "TemporalExtent": {
                    "RangeDateTime": {
                        "BeginningDateTime": f"{time}.000Z",
                        "EndingDateTime": f"{time}.000Z",
                    },
                },
                "RelatedUrls": [{"URL": href, "Type": "GET DATA"}],
            },
        },
        cloud_hosted=False,
    )


@pytest.fixture
def failing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The batch open fails; per-granule opens fail only for the bad URL."""

    def _batch_fails(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("batch open failed")

    def _single(url: str, **kwargs: Any) -> Any:
        if url == BAD:
            raise FileNotFoundError(f"cannot open {url}")
        return make_vds("2013-01-01", url=url)

    monkeypatch.setattr("earthaccess_vds.engine._open_virtual_mfdataset", _batch_fails)
    monkeypatch.setattr(
        "earthaccess_vds.engine._open_virtual_dataset_single", _single
    )


def _request(granules: list[Any], **options: Any) -> Any:
    job = VDSPlugin(fake_context()).request(granules, **options)
    return job, job.wait(timeout=60)


def test_on_error_skip_ingests_the_survivors(failing_backend: None) -> None:
    granules = [
        _granule(GOOD, "2013-01-01", "G1"),
        _granule(BAD, "2013-01-02", "G2"),
    ]

    job, status = _request(
        granules, concat_dim="time", parser="HDFParser", on_error="skip"
    )

    assert status is JobStatus.PARTIAL, job.info()
    info = job.info()
    assert info["failed"] == 1
    assert [error["granule"] for error in info["errors"]] == ["G2"]
    assert "cannot open" in info["errors"][0]["error"]
    assert job.results().sizes["time"] == 1


def test_on_error_raise_fails_the_batch(failing_backend: None) -> None:
    granules = [
        _granule(GOOD, "2013-01-01", "G1"),
        _granule(BAD, "2013-01-02", "G2"),
    ]

    job, status = _request(granules, concat_dim="time", parser="HDFParser")

    assert status is JobStatus.FAILED
    assert "batch open failed" in job.info()["error"]


def test_on_error_skip_is_rejected_with_tree() -> None:
    with pytest.raises(ValueError, match="tree"):
        VDSPlugin(fake_context()).request(
            [_granule(GOOD, "2013-01-01", "G1")],
            concat_dim="time",
            tree=True,
            on_error="skip",
        )
