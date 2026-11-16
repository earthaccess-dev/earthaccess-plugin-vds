"""Remote change probes for idempotent ingestion (opt-in ``verify="etag"``).

Metadata fingerprints catch the normal reprocessing path (CMR bumps
``revision-id``). They cannot catch a file replaced in place without a CMR
update — for that the writer has to ask the origin. This module performs the
cheap HTTP probe: a ``HEAD`` with the EDL session, following the DAAC's 303
redirect to the object store, and reads the ``ETag`` (the origin's, e.g. an S3
multipart ETag). It is not a content hash, but it changes when the object
changes, which is what the ledger needs.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from earthaccess_plugin_spec import AuthHandle

logger = logging.getLogger(__name__)

DEFAULT_PROBE_WORKERS = 8


def probe_url(granule: Any) -> str | None:
    """The URL to probe for an ETag: an HTTPS data link (S3 needs an S3 HEAD)."""

    try:
        links = granule.data_links(access="https")
    except Exception:  # noqa: BLE001 - a probe must never fail the request
        return None
    for link in links:
        if urlparse(link).scheme in ("http", "https"):
            return link
    return None


def head_etag(session: Any, url: str, *, timeout: float = 60) -> str | None:
    """Return the origin ETag for *url*, following DAAC redirects."""

    response = session.head(url, allow_redirects=True, timeout=timeout)
    if response.status_code >= 400:
        return None
    return response.headers.get("ETag")


def probe_etags(
    targets: dict[str, Any],
    *,
    auth: AuthHandle | None,
    max_workers: int = DEFAULT_PROBE_WORKERS,
) -> dict[str, str | None]:
    """Probe one ETag per key; failures yield ``None`` (unverified)."""

    if not targets:
        return {}
    session = auth.session() if auth is not None else None
    if session is None:
        return dict.fromkeys(targets)

    def _one(item: tuple[str, Any]) -> tuple[str, str | None]:
        key, granule = item
        url = probe_url(granule)
        if url is None:
            return key, None
        try:
            return key, head_etag(session, url)
        except Exception as exc:  # noqa: BLE001 - a probe must never fail the job
            logger.warning("ETag probe failed for %s: %r", key, exc)
            return key, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(_one, targets.items()))
