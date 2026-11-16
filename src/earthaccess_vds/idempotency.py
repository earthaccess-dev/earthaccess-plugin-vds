"""Idempotent ingestion: keys, fingerprints, and the in-store ledger.

A virtual store is append-only, so re-running a batch duplicates its time steps
and a reprocessed granule cannot replace its predecessor. This module provides
the pieces the plugin needs to make ``request()`` idempotent:

- the **key** is the concat-coordinate value (a time step) derived from the
  granule's UMM metadata, so no granule has to be opened to classify it;
- the **fingerprint** is a cheap, non-secret identity (CMR ``revision-id``,
  falling back to the first data link) used to tell an unchanged granule from a
  reprocessed one;
- the **ledger** records ``key -> fingerprint`` in the icechunk repository
  metadata, so a later run can skip unchanged keys and upsert reprocessed ones
  without a sidecar file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

INGESTED_METADATA_KEY = "earthaccess:vds/ingested"
"""Repository metadata key recording the ingested keys and their fingerprints."""


def granule_key(granule: Any, key: str | None = None) -> str | None:
    """Return the ISO time step of *granule* (the dedupe key), or ``None``.

    Only the time coordinate can be derived from CMR metadata without opening
    the granule; any other ``key`` disables dedupe for that batch.
    """

    if key not in (None, "time"):
        return None

    umm = granule.get("umm", {}) if hasattr(granule, "get") else {}
    extent = umm.get("TemporalExtent") or {}
    range_dt = extent.get("RangeDateTime") or {}
    stamp = range_dt.get("BeginningDateTime") or extent.get("SingleDateTime")
    if not stamp:
        return None

    try:
        import pandas as pd

        moment = pd.Timestamp(stamp)
        if moment.tzinfo is not None:
            # CMR stamps are UTC; the store's coordinate is naive datetime64
            moment = moment.tz_convert("UTC").tz_localize(None)
        return moment.isoformat()
    except (TypeError, ValueError):
        return None


def granule_fingerprint(granule: Any) -> dict[str, Any]:
    """A cheap identity for detecting a reprocessed granule."""

    meta = granule.get("meta", {}) if hasattr(granule, "get") else {}
    umm = granule.get("umm", {}) if hasattr(granule, "get") else {}
    production = (umm.get("DataGranule") or {}).get("ProductionDateTime")
    href = None
    try:
        links = granule.data_links()
        href = links[0] if links else None
    except Exception:  # noqa: BLE001 - a fingerprint must never fail a request
        href = None
    return {
        "granule": meta.get("concept-id"),
        "revision": meta.get("revision-id"),
        "production": production,
        "href": href,
    }


def same_fingerprint(
    recorded: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> bool:
    """Whether two fingerprints describe the same source bytes.

    CMR bumps ``revision-id`` when a granule is reprocessed, which is the
    authoritative signal. Without it, fall back to the data link.
    """

    if not recorded or not current:
        return False
    if recorded.get("revision") and current.get("revision"):
        if (
            recorded["revision"] != current["revision"]
            or recorded.get("granule") != current.get("granule")
        ):
            return False
        if recorded.get("production") and current.get("production"):
            return recorded["production"] == current["production"]
        return True
    if recorded.get("production") and current.get("production"):
        return (
            recorded["production"] == current["production"]
            and recorded.get("granule") == current.get("granule")
        )
    if recorded.get("etag") and current.get("etag"):
        return recorded["etag"] == current["etag"]
    return bool(recorded.get("href")) and recorded.get("href") == current.get("href")


@dataclass
class BatchPlan:
    """How a batch splits against what the store already holds.

    Items are ``(granule, key, fingerprint)`` tuples; reprocessed items also
    carry the store index they must overwrite: ``(granule, key, fingerprint,
    index)``.
    """

    new: list[tuple[Any, str, dict[str, Any]]] = field(default_factory=list)
    reprocessed: list[tuple[Any, str, dict[str, Any], int]] = field(
        default_factory=list
    )
    unchanged: list[tuple[Any, str, dict[str, Any]]] = field(default_factory=list)
    unkeyed: list[Any] = field(default_factory=list)

    @property
    def writes(self) -> int:
        return len(self.new) + len(self.reprocessed)

    def summary(self) -> dict[str, int]:
        return {
            "appended": len(self.new),
            "upserted": len(self.reprocessed),
            "skipped": len(self.unchanged),
            "unkeyed": len(self.unkeyed),
        }


def plan_batch(
    granules: list[Any],
    *,
    key: str | None,
    index: dict[str, int],
    ledger: dict[str, dict[str, Any]],
    dedupe: str = "upsert",
) -> BatchPlan:
    """Classify *granules* against the store's key index and ledger.

    ``dedupe="skip"`` never overwrites: a key that already exists is skipped
    even when its fingerprint changed.
    """

    plan = BatchPlan()
    known_store = bool(index or ledger)
    seen: set[str] = set()
    for granule in granules:
        k = granule_key(granule, key)
        fingerprint = granule_fingerprint(granule)
        if k is None:
            plan.unkeyed.append(granule)
            continue
        if known_store and k in seen:
            msg = (
                f"the batch has more than one granule for key {k!r}; "
                "dedupe needs one granule per key. Pass key=<coordinate> to "
                "identify a slot, or dedupe='off' to append them all."
            )
            raise ValueError(msg)
        seen.add(k)
        if k not in index:
            if k in ledger:
                # the ledger knows this key but the store index is unreadable:
                # skip rather than append a duplicate
                plan.unchanged.append((granule, k, fingerprint))
            else:
                plan.new.append((granule, k, fingerprint))
            continue
        if dedupe == "skip" or same_fingerprint(ledger.get(k), fingerprint):
            plan.unchanged.append((granule, k, fingerprint))
            continue
        plan.reprocessed.append((granule, k, fingerprint, index[k]))
    return plan


def contiguous_runs(
    items: list[tuple[Any, str, dict[str, Any], int]],
) -> list[list[tuple[Any, str, dict[str, Any], int]]]:
    """Group reprocessed items into runs of consecutive store indices."""

    runs: list[list[tuple[Any, str, dict[str, Any], int]]] = []
    for item in sorted(items, key=lambda item: item[3]):
        if runs and item[3] == runs[-1][-1][3] + 1:
            runs[-1].append(item)
        else:
            runs.append([item])
    return runs


def ledger_entries(
    plan: BatchPlan,
    *,
    include_unchanged: bool = False,
) -> dict[str, dict[str, Any]]:
    """The ledger updates implied by a plan.

    Written keys always update the ledger; with ``include_unchanged`` the keys
    that were probed but did not change are recorded too (so an ETag baseline
    is kept without rewriting their references).
    """

    entries: dict[str, dict[str, Any]] = {}
    for _granule, key, fingerprint in plan.new:
        entries[key] = fingerprint
    for _granule, key, fingerprint, _index in plan.reprocessed:
        entries[key] = fingerprint
    if include_unchanged:
        for _granule, key, fingerprint in plan.unchanged:
            if fingerprint.get("etag"):
                entries[key] = fingerprint
    return entries


def apply_etags(
    plan: BatchPlan,
    etags: dict[str, str | None],
    *,
    index: dict[str, int],
    ledger: dict[str, dict[str, Any]],
) -> int:
    """Fold probed ETags into *plan*; return the number of unverified items.

    A key whose recorded ETag differs is promoted to ``reprocessed`` (it was
    replaced in place, without a CMR revision bump). A key without a recorded
    ETag keeps its references but records the baseline. A failed probe leaves
    the item unchanged and counts as unverified.
    """

    unverified = 0
    remaining: list[tuple[Any, str, dict[str, Any]]] = []
    for granule, key, fingerprint in plan.unchanged:
        probed = etags.get(key)
        if probed is None:
            unverified += 1
            remaining.append((granule, key, fingerprint))
            continue
        updated = {**fingerprint, "etag": probed}
        recorded = (ledger.get(key) or {}).get("etag")
        if recorded and recorded != probed and key in index:
            plan.reprocessed.append((granule, key, updated, index[key]))
        else:
            remaining.append((granule, key, updated))
    plan.unchanged = remaining

    plan.new = [
        (granule, key, {**fingerprint, "etag": etags[key]})
        if etags.get(key)
        else (granule, key, fingerprint)
        for granule, key, fingerprint in plan.new
    ]
    plan.reprocessed = [
        (granule, key, {**fingerprint, "etag": etags[key]}, position)
        if etags.get(key)
        else (granule, key, fingerprint, position)
        for granule, key, fingerprint, position in plan.reprocessed
    ]
    return unverified
