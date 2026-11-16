"""Idempotent ingestion: skip unchanged, upsert reprocessed, append new."""

from __future__ import annotations

from typing import Any

import icechunk
import pytest
import xarray as xr
from earthaccess.search import DataGranule
from earthaccess_plugin_spec import JobStatus
from earthaccess_plugin_spec.testing import fake_context

from earthaccess_vds import VDSPlugin
from earthaccess_vds.idempotency import (
    INGESTED_METADATA_KEY,
    contiguous_runs,
    granule_fingerprint,
    granule_key,
    plan_batch,
    same_fingerprint,
)
from earthaccess_vds.writer import read_icechunk_state
from tests.synthetic import make_vds

TIME = "2013-01-01T00:00:00"
TIME_2 = "2013-01-02T00:00:00"
HREF = "https://a.example/f1.nc"
HREF_V2 = "https://a.example/f2.nc"


def _granule(
    *,
    time: str = TIME,
    revision: str = "1",
    href: str = HREF,
    concept_id: str = "G1-POCLOUD",
) -> DataGranule:
    stamp = f"{time}.000Z"
    return DataGranule(
        {
            "meta": {
                "concept-id": concept_id,
                "collection-concept-id": "C1996881146-POCLOUD",
                "provider-id": "POCLOUD",
                "revision-id": revision,
            },
            "umm": {
                "TemporalExtent": {
                    "RangeDateTime": {
                        "BeginningDateTime": stamp,
                        "EndingDateTime": stamp,
                    },
                },
                "RelatedUrls": [{"URL": href, "Type": "GET DATA"}],
            },
        },
        cloud_hosted=False,
    )


def _fake_virtualize(granules: Any, *, concat_dim: str | None = None, **_: Any) -> Any:
    """A per-granule fake that keeps each granule's own href and time."""

    pieces = []
    for granule in granules:
        stamp = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        pieces.append(make_vds(stamp.split(".")[0], url=granule.data_links()[0]))
    if len(pieces) == 1:
        return pieces[0]
    return xr.combine_nested(pieces, concat_dim=concat_dim)


def _request(monkeypatch: pytest.MonkeyPatch, granules: Any, **options: Any) -> Any:
    monkeypatch.setattr("earthaccess_vds.engine.virtualize", _fake_virtualize)
    job = VDSPlugin(fake_context()).request(granules, **options)
    return job, job.wait(timeout=60)


def _paths(href: str) -> list[str]:
    import earthaccess as ea

    vds = ea.open_virtual(href, load=False)
    return [
        entry["path"] for entry in vds.sst.data.manifest.dict().values()
    ]


def _times(href: str) -> list[str]:
    import earthaccess as ea

    vds = ea.open_virtual(href, load=False)
    return [str(value) for value in vds.time.values]


def _ledger(href: str) -> dict[str, Any]:
    repo = icechunk.Repository.open(icechunk.Storage.new_local_filesystem(href))
    return ((repo.metadata or {}).get(INGESTED_METADATA_KEY) or {}).get("keys") or {}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_granule_key_normalizes_to_naive_utc() -> None:
    assert granule_key(_granule()) == TIME
    assert granule_key(_granule(), "time") == TIME
    assert granule_key(_granule(), "lat") is None


def test_granule_key_accepts_single_date_time() -> None:
    granule = _granule()
    granule["umm"]["TemporalExtent"] = {"SingleDateTime": "2013-01-01T00:00:00.000Z"}

    assert granule_key(granule) == TIME


def test_fingerprint_prefers_revision_then_href() -> None:
    first = granule_fingerprint(_granule())
    reprocessed = granule_fingerprint(_granule(revision="2", href=HREF_V2))
    no_revision = granule_fingerprint(_granule(revision="", href=HREF))

    assert same_fingerprint(first, first)
    assert not same_fingerprint(first, reprocessed)
    assert same_fingerprint(no_revision, no_revision)
    assert not same_fingerprint(first, None)


def test_plan_batch_classifies() -> None:
    time_3 = "2013-01-03T00:00:00"
    plan = plan_batch(
        [
            _granule(),                                            # unchanged
            _granule(time=TIME_2, revision="2", href=HREF_V2),     # reprocessed
            _granule(time=time_3, concept_id="G3"),                # new
            _granule(time="", href=""),                            # unkeyed
        ],
        key="time",
        index={TIME: 0, TIME_2: 1},
        ledger={
            TIME: granule_fingerprint(_granule()),
            TIME_2: granule_fingerprint(_granule(time=TIME_2)),
        },
    )

    assert len(plan.unchanged) == 1
    assert len(plan.reprocessed) == 1
    assert plan.reprocessed[0][3] == 1
    assert len(plan.new) == 1
    assert len(plan.unkeyed) == 1
    assert plan.summary() == {
        "appended": 1,
        "upserted": 1,
        "skipped": 1,
        "unkeyed": 1,
    }


def test_contiguous_runs_groups_consecutive_indices() -> None:
    items: list[tuple[Any, str, dict[str, Any], int]] = [
        (_granule(time=TIME), TIME, {}, 0),
        (_granule(time=TIME_2), TIME_2, {}, 1),
        (_granule(time="2013-01-05T00:00:00"), "k", {}, 4),
    ]

    runs = contiguous_runs(items)

    assert [len(run) for run in runs] == [2, 1]


# ---------------------------------------------------------------------------
# Through request()
# ---------------------------------------------------------------------------


def test_rerun_is_a_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    store = str(tmp_path / "sst.icechunk")
    _request(monkeypatch, [_granule()], store=store, concat_dim="time")

    job, status = _request(monkeypatch, [_granule()], store=store, concat_dim="time")

    assert status is JobStatus.SUCCEEDED, job.info()
    assert _times(store) == ["2013-01-01T00:00:00.000000000"]
    assert job.info()["summary"]["skipped"] == 1
    assert _paths(store) == [HREF]


def test_reprocessed_granule_is_upserted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    _request(monkeypatch, [_granule()], store=store, concat_dim="time")

    job, status = _request(
        monkeypatch,
        [_granule(revision="2", href=HREF_V2)],
        store=store,
        concat_dim="time",
    )

    assert status is JobStatus.SUCCEEDED, job.info()
    assert _times(store) == ["2013-01-01T00:00:00.000000000"]
    assert _paths(store) == [HREF_V2]
    assert job.info()["summary"]["upserted"] == 1


def test_new_time_is_appended(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    store = str(tmp_path / "sst.icechunk")
    _request(monkeypatch, [_granule()], store=store, concat_dim="time")

    job, status = _request(
        monkeypatch,
        [_granule(time=TIME_2, concept_id="G2", href="https://a.example/f3.nc")],
        store=store,
        concat_dim="time",
    )

    assert status is JobStatus.SUCCEEDED, job.info()
    assert len(_times(store)) == 2
    assert job.info()["summary"]["appended"] == 1


def test_mixed_batch_upserts_and_appends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    _request(monkeypatch, [_granule()], store=store, concat_dim="time")

    job, status = _request(
        monkeypatch,
        [
            _granule(revision="2", href=HREF_V2),
            _granule(time=TIME_2, concept_id="G2", href="https://a.example/f3.nc"),
        ],
        store=store,
        concat_dim="time",
    )

    assert status is JobStatus.SUCCEEDED, job.info()
    assert len(_times(store)) == 2
    assert _paths(store)[0] == HREF_V2
    assert job.info()["summary"] == {
        "appended": 1,
        "upserted": 1,
        "skipped": 0,
        "unkeyed": 0,
    }


def test_dedupe_skip_never_overwrites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    _request(monkeypatch, [_granule()], store=store, concat_dim="time")

    job, status = _request(
        monkeypatch,
        [_granule(revision="2", href=HREF_V2)],
        store=store,
        concat_dim="time",
        dedupe="skip",
    )

    assert status is JobStatus.SUCCEEDED, job.info()
    assert _paths(store) == [HREF]
    assert job.info()["summary"]["skipped"] == 1


def test_dedupe_off_keeps_the_old_append_behaviour(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    _request(monkeypatch, [_granule()], store=store, concat_dim="time")

    job, status = _request(
        monkeypatch, [_granule()], store=store, concat_dim="time", dedupe="off"
    )

    assert status is JobStatus.SUCCEEDED, job.info()
    assert len(_times(store)) == 2


def test_ledger_records_the_ingested_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    _request(monkeypatch, [_granule()], store=store, concat_dim="time")

    ledger = _ledger(store)

    assert ledger[TIME]["revision"] == "1"
    assert ledger[TIME]["href"] == HREF


def test_read_state_of_a_missing_store_is_empty(tmp_path: Any) -> None:
    index, ledger, resolved = read_icechunk_state(
        str(tmp_path / "missing.icechunk"), "time"
    )

    assert index == {} and ledger == {}
    assert resolved.access == "file"


def test_duplicate_keys_are_rejected_once_the_store_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    store = str(tmp_path / "sst.icechunk")
    first = [_granule(concept_id="G1"), _granule(concept_id="G2")]
    job, status = _request(monkeypatch, first, store=store, concat_dim="time")
    assert status is JobStatus.SUCCEEDED, job.info()

    job, status = _request(monkeypatch, first, store=store, concat_dim="time")

    assert status is JobStatus.FAILED
    assert "more than one granule" in job.info()["error"]


def test_ledger_key_without_an_index_is_skipped() -> None:
    plan = plan_batch(
        [_granule(revision="2", href=HREF_V2)],
        key="time",
        index={},  # the coordinate could not be read
        ledger={TIME: granule_fingerprint(_granule())},
    )

    assert len(plan.unchanged) == 1
    assert plan.writes == 0
