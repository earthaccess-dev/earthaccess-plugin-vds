"""Region defaults for S3 and icechunk targets.

NASA Earthdata S3 buckets live in ``us-west-2`` and direct S3 access is only
allowed from inside that region: ``access="direct"`` needs the workers there.
HTTPS access (``access="indirect"``, the default) works from anywhere, and the
icechunk store write only needs the *target* bucket's region for signing.

Whether a process actually runs in ``us-west-2`` is not detectable reliably
(instance metadata may be blocked by policy, and serverless runtimes do not
expose it), so this module only provides the default — the constraint is
documented rather than enforced.
"""

from __future__ import annotations

NASA_REGION = "us-west-2"
"""Default region for S3 targets and direct NASA S3 access."""
