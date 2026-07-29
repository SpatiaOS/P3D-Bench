"""Text-to-3D scoring contract: QA weighting + the shared text condition.

Both are places where a plausible-looking simplification silently changes what
the benchmark reports, so pin them.
"""

from __future__ import annotations

import pytest

from p3dbench.data.schema import Case
from p3dbench.metrics.base import bucket_score_for_case
from p3dbench.tasks.text_to_3d import TASK
from p3dbench.text_condition import resolve_text_condition


def _case(text: str = "parametric text", **metadata) -> Case:
    return Case.from_dict({
        "id": "c",
        "task": "text-to-3d",
        "split": "demo",
        "input": {"text": text},
        "target": {},
        "metadata": metadata,
    })


# -- QA weighting -----------------------------------------------------------
def test_param_judge_is_question_level_micro_average():
    """4 QA-S + 8 QA-P questions -> (QA-S + 2*QA-P)/3, not mean(QA-S, QA-P)."""
    buckets = bucket_score_for_case(
        "text-to-3d",
        {"qa_semantic": 1.0, "qa_param": 0.25},
        valid=True,
        text_mode="parametric",
    )
    assert buckets["judge"] == pytest.approx((1.0 + 2 * 0.25) / 3)
    # The macro mean would be 0.625 — a 4-question split must not outweigh an
    # 8-question one.
    assert buckets["judge"] != pytest.approx(0.625)


def test_descriptive_judge_stays_an_equal_weight_mean():
    buckets = bucket_score_for_case(
        "text-to-3d",
        {"qa_semantic": 1.0, "judge_semantic": 10},
        valid=True,
        text_mode="descriptive",
    )
    assert buckets["judge"] == pytest.approx(1.0)


def test_invalid_case_is_worst_filled_in_both_modes():
    for mode in ("parametric", "descriptive"):
        buckets = bucket_score_for_case("text-to-3d", {}, valid=False, text_mode=mode)
        assert buckets["judge"] == 0.0


# -- shared text condition --------------------------------------------------
def test_descriptive_uses_text_desc_and_falls_back():
    case = _case(text_desc="a natural-language description")
    assert resolve_text_condition(case, "descriptive") == "a natural-language description"
    assert resolve_text_condition(case, "parametric") == "parametric text"
    # Demo split carries no text_desc.
    assert resolve_text_condition(_case(), "descriptive") == "parametric text"


def test_generation_prompt_uses_the_same_resolver():
    """The J-Sem judge scores against this exact string, so they must agree."""
    from p3dbench.formats import get_format

    case = _case(text_desc="a natural-language description")
    bundle = TASK.build_prompt(get_format("openscad"), case, [], text_mode="descriptive")
    assert resolve_text_condition(case, "descriptive") in bundle.user
    assert "parametric text" not in bundle.user


def test_unknown_text_mode_is_rejected():
    with pytest.raises(ValueError):
        resolve_text_condition(_case(), "detailed")
