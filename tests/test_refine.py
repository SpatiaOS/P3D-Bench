"""Hermetic tests for the transport-retry + invalid->refine machinery.

No network, no API keys, no external binaries (openscad/node/Blender), no local
data — everything is faked or mocked, so this runs anywhere (incl. a fresh CI
checkout) in well under a second and never skips. It guards the behaviour ported
from cadbenchmark:

  * ``models/_retry.post_with_retry`` — exponential backoff on 429/5xx, transient
    transport errors, and HTTP-200 provider-error bodies; budget exhaustion raises.
  * ``pipeline._infer_with_refine`` — compile-check-retry that feeds the compile
    error back and regenerates (image-/assembly-3d); Text-to-3D stays single-shot.
"""

from __future__ import annotations

import re
from unittest import mock

import pytest

import p3dbench.pipeline as P
from p3dbench.formats.base import CompileResult
from p3dbench.models import _retry
from p3dbench.models.base import ModelResponse


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------
class _Resp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, ok, status, body):
        self.ok = ok
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


class FakeClient:
    """Returns a queued response per call; records the prompt it was given."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.prompts = []

    def generate(self, prompt, *, images=None, system=None):
        self.prompts.append(prompt)
        text = self._texts[len(self.prompts) - 1]
        return ModelResponse(text=text, model="fake", usage={"completion_tokens": 10})


class FakeFormat:
    """Format stub: extract the first ```fence``` block; compile by sentinel.

    ``compile`` returns valid iff the code contains ``GOOD`` — no real geometry
    engine involved, so the test is independent of openscad/cadquery/node.
    """

    slug = "openscad"
    display_name = "OpenSCAD"
    extension = ".scad"

    def extract_code(self, raw_text):
        m = re.findall(r"```(?:\w+)?\s*\n(.*?)```", raw_text, re.DOTALL)
        return (m[0].strip() if m else raw_text.strip())

    def compile(self, code, output_dir):
        if "GOOD" in code:
            return CompileResult(valid=True, stl=str(output_dir) + "/model.stl")
        return CompileResult(valid=False, errors=["boom: bad code"],
                             error_details=[{"stage": "execute", "line_number": 2}])


class Bundle:
    system = "SYS"
    user = "Draw a 10mm cube."
    images = []


def _bad(tag="x"):
    return f"```scad\nBAD {tag}\n```"


def _good():
    return "```scad\nGOOD cube(10);\n```"


# --------------------------------------------------------------------------
# transport retry
# --------------------------------------------------------------------------
def test_retry_backoff_and_provider_error_body():
    """503 -> backoff, HTTP-200 provider-error body -> backoff, then success."""
    seq = [
        _Resp(False, 503, {"error": "down"}),
        _Resp(True, 200, {"error": {"code": 522, "message": "timeout"}}),  # 200 but error
        _Resp(True, 200, {"choices": [{"message": {"content": "hi"}}]}),
    ]
    calls = {"i": 0}

    def fake_post(url, headers, json, timeout):
        r = seq[calls["i"]]
        calls["i"] += 1
        return r

    slept = []
    with mock.patch("p3dbench.models._retry.requests.post", side_effect=fake_post), \
         mock.patch("p3dbench.models._retry.time.sleep", side_effect=slept.append):
        resp = _retry.post_with_retry(
            "u", headers={}, json_body={}, timeout=10, max_retries=5,
            retryable_body=_retry.openrouter_error_reason,
        )
    assert resp.json()["choices"]          # returned the good response
    assert calls["i"] == 3                 # took all three
    assert slept == [5, 10]                # exponential backoff 5, 10


def test_retry_exhaustion_raises():
    """All attempts 5xx -> raises after the budget, never returns a bad response."""
    def always_500(url, headers, json, timeout):
        return _Resp(False, 500, {})

    with mock.patch("p3dbench.models._retry.requests.post", side_effect=always_500), \
         mock.patch("p3dbench.models._retry.time.sleep"):
        with pytest.raises(Exception):
            _retry.post_with_retry("u", headers={}, json_body={}, timeout=10, max_retries=2)


def test_retry_no_retry_budget_one():
    """max_retries=1 means a single attempt, no sleep."""
    def good(url, headers, json, timeout):
        return _Resp(True, 200, {"choices": [1]})

    slept = []
    with mock.patch("p3dbench.models._retry.requests.post", side_effect=good), \
         mock.patch("p3dbench.models._retry.time.sleep", side_effect=slept.append):
        _retry.post_with_retry("u", headers={}, json_body={}, timeout=10, max_retries=1)
    assert slept == []


def test_openrouter_error_reason():
    assert _retry.openrouter_error_reason({"error": {"code": 522, "message": "x"}}) == "code=522 x"
    assert _retry.openrouter_error_reason({"choices": [{}]}) is None
    assert _retry.openrouter_error_reason({"error": "flat"}) == "flat"


# --------------------------------------------------------------------------
# refine gating
# --------------------------------------------------------------------------
def test_effective_attempts_gating():
    assert P._effective_attempts("text-to-3d", None) == 1      # text: always single-shot
    assert P._effective_attempts("text-to-3d", 5) == 1         # even if asked
    assert P._effective_attempts("image-to-3d", None) == P.DEFAULT_REFINE_ATTEMPTS
    assert P._effective_attempts("assembly-3d", 1) == 1        # explicit disable
    assert P._effective_attempts("image-to-3d", 4) == 4


# --------------------------------------------------------------------------
# refine loop
# --------------------------------------------------------------------------
def test_refine_recovers_on_feedback():
    """Invalid attempt 1 -> valid attempt 2; the retry prompt carries the error."""
    fc = FakeClient([_bad("1"), _good()])
    row = {"id": "c"}
    P._infer_with_refine(fc, FakeFormat(), Bundle(), row, max_attempts=3)
    assert row["error"] is None
    assert row["attempts"] == 2
    assert "GOOD" in row["code"]
    assert row["failure_class"] is None
    # second call must include the fed-back error + previous code
    assert "failed to compile" in fc.prompts[1]
    assert "BAD 1" in fc.prompts[1]
    assert "boom: bad code" in fc.prompts[1]


def test_refine_exhausts_all_invalid():
    fc = FakeClient([_bad("1"), _bad("2"), _bad("3")])
    row = {"id": "c"}
    P._infer_with_refine(fc, FakeFormat(), Bundle(), row, max_attempts=3)
    assert row["error"] is not None
    assert row["attempts"] == 3
    assert len(fc.prompts) == 3
    assert row["failure_class"] == P.FAILURE_GENERATION_INVALID


def test_refine_escalates_on_repeated_error():
    """Same error twice -> the third prompt carries the escalation note."""
    fc = FakeClient([_bad("same"), _bad("same"), _good()])
    row = {"id": "c"}
    P._infer_with_refine(fc, FakeFormat(), Bundle(), row, max_attempts=3)
    # attempt 2's signature repeats attempt 1 -> prompt built for attempt 3 escalates
    assert "times in a row" in fc.prompts[2]


def test_refine_empty_extraction_stops_immediately():
    fc = FakeClient(["```scad\n\n```"] * 3)   # empty fenced block
    row = {"id": "c"}
    P._infer_with_refine(fc, FakeFormat(), Bundle(), row, max_attempts=3)
    assert row["error"] == "empty code extraction"
    assert row["failure_class"] == P.FAILURE_GENERATION_INVALID
    assert len(fc.prompts) == 1               # no wasted retries


def test_refine_llm_failure_stops_immediately():
    class BoomClient:
        def __init__(self):
            self.n = 0

        def generate(self, prompt, *, images=None, system=None):
            self.n += 1
            raise RuntimeError("api exploded")

    bc = BoomClient()
    row = {"id": "c"}
    P._infer_with_refine(bc, FakeFormat(), Bundle(), row, max_attempts=3)
    assert "api exploded" in row["error"]
    assert bc.n == 1                          # LLM-side failure: no feedback retry
    assert row["failure_class"] == P.FAILURE_INFERENCE_GAP


def test_refine_usage_accumulates():
    fc = FakeClient([_bad("1"), _good()])
    row = {"id": "c"}
    P._infer_with_refine(fc, FakeFormat(), Bundle(), row, max_attempts=3)
    assert row["usage"]["completion_tokens"] == 20   # 10 + 10 across two attempts


def test_single_shot_path_sets_no_attempt_history():
    fc = FakeClient([_good()])
    # `infer` seeds these base fields before calling; single-shot only overwrites
    # `error` on failure, so reproduce that contract here.
    row = {"id": "c", "raw_text": None, "code": None, "usage": {}, "error": None}
    P._infer_single_shot(fc, FakeFormat(), Bundle(), row)
    assert row["error"] is None
    assert row["failure_class"] is None
    assert "GOOD" in row["code"]
    assert "attempt_history" not in row      # single-shot carries no refine record


# --------------------------------------------------------------------------
# refine helpers
# --------------------------------------------------------------------------
def test_count_repeats():
    sig = "execute|2|boom"
    hist = [{"valid": False, "error_signature": sig}]
    assert P._count_repeats(hist, sig) == 2
    assert P._count_repeats(hist, "other|None|x") == 1
    assert P._count_repeats([{"valid": True, "error_signature": sig}], sig) == 1


def test_is_export_timeout_only():
    assert P._is_export_timeout_only(["CadQuery export timed out after 60s"]) is True
    assert P._is_export_timeout_only(["syntax error", "timed out"]) is False
    assert P._is_export_timeout_only([]) is False


def test_merge_usage_nested():
    total = {}
    P._merge_usage(total, {"prompt_tokens": 10, "is_byok": False, "d": {"x": 1}})
    P._merge_usage(total, {"prompt_tokens": 5, "d": {"x": 2, "y": 3}})
    assert total["prompt_tokens"] == 15
    assert total["d"] == {"x": 3, "y": 3}
    assert total["is_byok"] is False         # bool is set-once, not summed


def test_build_refine_prompt_contents():
    p = P._build_refine_prompt(
        "ORIGINAL TASK", "BAD code", ["boom"], FakeFormat(),
        has_images=True, error_detail={"stage": "execute", "line_number": 2}, repeat_count=2,
    )
    assert "ORIGINAL TASK" in p              # original task preserved
    assert "BAD code" in p                   # previous code echoed
    assert "boom" in p                       # error fed back
    assert "image(s)" in p                   # image hint when has_images
    assert "Failing line number: 2" in p     # diagnostics
    assert "times in a row" in p             # escalation at repeat>=2
