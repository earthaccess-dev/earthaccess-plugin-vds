"""Region defaults for S3 / icechunk targets."""

from __future__ import annotations

from typing import Any

import pytest
from earthaccess_plugin_spec.testing import fake_context

from earthaccess_vds import VDSPlugin
from earthaccess_vds.regions import NASA_REGION
from earthaccess_vds.storage import _from_s3_store, resolve_icechunk_target
from earthaccess_vds.writer import _region_hint


class _FakeS3Store:
    """The attributes ``_from_s3_store`` reads from an obstore store."""

    config = {"bucket": "bucket"}
    prefix = "prefix"
    client_options: dict[str, Any] = {}
    credential_provider = None


def test_s3_uri_defaults_to_us_west_2() -> None:
    resolved = resolve_icechunk_target("s3://bucket/prefix")

    assert "region: us-west-2" in str(resolved.storage)


def test_s3_uri_region_can_be_overridden() -> None:
    resolved = resolve_icechunk_target("s3://bucket/prefix", region="eu-west-1")

    assert "region: eu-west-1" in str(resolved.storage)


def test_s3_store_without_a_region_gets_the_default() -> None:
    resolved = _from_s3_store(_FakeS3Store(), href=None, prefix=None)

    assert "region: us-west-2" in str(resolved.storage)


def test_s3_store_region_wins_over_the_default() -> None:
    class _RegionalStore(_FakeS3Store):
        config = {"bucket": "bucket", "region": "ap-south-1"}

    resolved = _from_s3_store(_RegionalStore(), href=None, prefix=None)

    assert "region: ap-south-1" in str(resolved.storage)


def test_region_hint_for_a_redirect_error() -> None:
    hint = _region_hint(RuntimeError("301 PermanentRedirect"), "us-west-2")

    assert isinstance(hint, ValueError)
    assert "region=" in str(hint)


def test_region_hint_ignores_unrelated_errors() -> None:
    assert _region_hint(RuntimeError("disk full"), "us-west-2") is None


def test_options_default_region_is_us_west_2() -> None:
    assert VDSPlugin.Options().region == NASA_REGION


def test_request_accepts_a_custom_region() -> None:
    plugin = VDSPlugin(fake_context())

    with pytest.raises(ValueError, match="no input"):
        plugin.request(None, region="eu-west-1")
