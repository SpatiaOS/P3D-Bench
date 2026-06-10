"""Valid bucket — the executable-validity gate.

valid == an STL was produced (the program compiled, built a solid, and exported
a non-empty mesh). A non-empty ``errors`` list does NOT by itself invalidate a
case (e.g. a failed per-part export still leaves the union valid). Reported
separately from the four scored buckets; predictions that fail it are worst-filled
in the scored buckets (see metrics.base).
"""

from __future__ import annotations

from .base import MetricBucket, ScoreContext


def classify_invalid(compiled: dict) -> str:
    """Coarse reason for an invalid case (model-output vs local-CAD failure)."""
    errors = compiled.get("errors") or []
    blob = " ".join(str(e) for e in errors).lower()
    if not compiled.get("step") and not compiled.get("stl"):
        if "no code" in blob or "empty" in blob:
            return "no_code"
        if "json" in blob:
            return "json_parse"
        if "syntax" in blob or "undefined" in blob or "nameerror" in blob:
            return "code_error"
        return "build_failed"
    # STEP produced but no STL -> tessellation failed.
    if compiled.get("step") and not compiled.get("stl"):
        return "mesh_failed"
    return "unknown"


class _ValidBucket(MetricBucket):
    bucket = "valid"
    requires: set[str] = set()

    def score(self, ctx: ScoreContext) -> dict:
        valid = bool(ctx.compiled.get("stl"))
        return {"valid": valid, "invalid_reason": None if valid else classify_invalid(ctx.compiled)}


BUCKET = _ValidBucket()
