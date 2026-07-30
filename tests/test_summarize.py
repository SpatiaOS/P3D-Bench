"""summarize() denominators: untested vs invalid.

An API failure means the case was never tested — it must not be scored as if the
model had produced a bad answer, and it must not sit in the Valid denominator
either. A model that answered with uncompilable code is the opposite: a genuine
model failure that stays worst-filled.
"""

from __future__ import annotations

import json

from p3dbench.pipeline import summarize
from p3dbench.utils import write_jsonl


def _row(case_id: str, *, valid: bool, llm_failed: bool = False, **metrics) -> dict:
    return {
        "id": case_id,
        "task": "image-to-3d",
        "format": "openscad",
        "model": "m",
        "split": "demo",
        "text_mode": "parametric",
        "valid": valid,
        "llm_failed": llm_failed,
        "buckets": ["valid", "topology"],
        "raw_metrics": metrics,
    }


_PERFECT = {"no_open_edge": 1.0, "inverted_normal_ratio": 0.0,
            "non_manifold_edge_ratio": 0.0}


def _summarize(tmp_path, rows) -> dict:
    metrics = tmp_path / "metrics.jsonl"
    out = tmp_path / "summary.json"
    write_jsonl(metrics, rows)
    summarize(metrics, out=out)
    return json.loads(out.read_text())["groups"][0]


def test_untested_case_leaves_every_denominator(tmp_path):
    group = _summarize(tmp_path, [
        _row("a", valid=True, **_PERFECT),
        _row("b", valid=False, llm_failed=True),
    ])
    assert group["n_cases"] == 2
    assert group["n_tested"] == 1
    assert group["llm_fail_cases"] == 1
    assert group["llm_fail_rate"] == 0.5
    # Valid is 1/1 tested, not 1/2 total; and the untested case does not drag
    # Topology down to 0.5.
    assert group["valid_rate"] == 1.0
    assert group["buckets"]["topology"] == 1.0


def test_answered_but_invalid_case_is_worst_filled(tmp_path):
    group = _summarize(tmp_path, [
        _row("a", valid=True, **_PERFECT),
        _row("b", valid=False),
    ])
    assert group["n_tested"] == 2
    assert group["llm_fail_cases"] == 0
    assert group["valid_rate"] == 0.5
    assert group["buckets"]["topology"] == 0.5   # (1.0 + 0.0) / 2


def test_all_untested_group_reports_no_score(tmp_path):
    group = _summarize(tmp_path, [
        _row("a", valid=False, llm_failed=True),
        _row("b", valid=False, llm_failed=True),
    ])
    assert group["n_tested"] == 0
    assert group["llm_fail_rate"] == 1.0
    assert group["valid_rate"] is None
    assert group["buckets"] == {}
    assert group["score"] is None


def test_rows_without_the_flag_behave_as_before(tmp_path):
    """Metrics files written before the flag existed must still summarize."""
    rows = [_row("a", valid=True, **_PERFECT), _row("b", valid=False)]
    for row in rows:
        row.pop("llm_failed")
    group = _summarize(tmp_path, rows)
    assert group["n_tested"] == 2
    assert group["llm_fail_cases"] == 0
    assert group["valid_rate"] == 0.5
