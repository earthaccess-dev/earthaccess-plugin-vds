"""HTTP ETag probes (opt-in ``verify="etag"``)."""

from __future__ import annotations

from typing import Any

from earthaccess.search import DataGranule

from earthaccess_vds import probes


class _Response:
    def __init__(self, status_code: int, headers: dict[str, str]) -> None:
        self.status_code = status_code
        self.headers = headers


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def head(self, url: str, **kwargs: Any) -> _Response:
        self.kwargs = kwargs
        return self.response


def _granule(*, https: tuple[str, ...] = (), s3: tuple[str, ...] = ()) -> DataGranule:
    urls = [{"URL": url, "Type": "GET DATA"} for url in https] + [
        {"URL": url, "Type": "GET DATA VIA DIRECT ACCESS"} for url in s3
    ]
    return DataGranule(
        {"meta": {"concept-id": "G1"}, "umm": {"RelatedUrls": urls}},
        cloud_hosted=bool(s3),
    )


def test_head_etag_follows_redirects() -> None:
    session = _Session(_Response(200, {"ETag": '"abc-3"'}))

    assert probes.head_etag(session, "https://example.com/a.nc") == '"abc-3"'
    assert session.kwargs["allow_redirects"] is True


def test_head_etag_returns_none_on_error() -> None:
    session = _Session(_Response(403, {}))

    assert probes.head_etag(session, "https://example.com/a.nc") is None


def test_probe_url_prefers_https() -> None:
    granule = _granule(
        https=("https://example.com/a.nc",),
        s3=("s3://bucket/a.nc",),
    )

    assert probes.probe_url(granule) == "https://example.com/a.nc"


def test_probe_url_without_https_is_none() -> None:
    assert probes.probe_url(_granule(s3=("s3://bucket/a.nc",))) is None


def test_probe_etags_maps_every_key() -> None:
    granule = _granule(https=("https://example.com/a.nc",))
    session = _Session(_Response(200, {"ETag": '"x"'}))

    class _Auth:
        def session(self) -> _Session:
            return session

        def token(self) -> str:
            return "token"

        def basic_auth(self) -> tuple[str, str] | None:
            return None

    result = probes.probe_etags({"k1": granule, "k2": granule}, auth=_Auth())

    assert result == {"k1": '"x"', "k2": '"x"'}
