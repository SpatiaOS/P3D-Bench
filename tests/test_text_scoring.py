"""Text-to-3D scoring contract: QA weighting + the shared text condition.

Both are places where a plausible-looking simplification silently changes what
the benchmark reports, so pin them.
"""

from __future__ import annotations

from pathlib import Path

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


# -- GT judge views ---------------------------------------------------------
class _FakeResolved:
    """Minimal ResolvedCase stand-in for the GT-view helper."""

    def __init__(self, case, gt_mesh=None, gt_renders=()):
        self.case = case
        self.gt_mesh = gt_mesh
        self.gt_renders = list(gt_renders)

    @property
    def id(self):
        return self.case.id


def _ctx(resolved):
    from p3dbench.metrics.base import ScoreContext

    return ScoreContext(
        case=resolved, task="text-to-3d", fmt="openscad",
        compiled={}, work_dir=None,
    )


def test_shipped_four_gt_views_are_used_as_is(tmp_path):
    from p3dbench.metrics.judge import _gt_judge_views_or_render

    shipped = []
    for i in range(4):
        p = tmp_path / f"view_{i:03d}.png"
        p.write_bytes(b"")
        shipped.append(p)
    views = _gt_judge_views_or_render(_ctx(_FakeResolved(_case(), gt_renders=shipped)))
    assert views == [str(p) for p in shipped]


def test_gt_views_are_rendered_from_the_mesh_and_cached(tmp_path, monkeypatch):
    """Text-to-3D ships no GT views, so they get rendered once and reused."""
    import p3dbench.metrics.judge as J

    mesh = tmp_path / "gt.stl"
    mesh.write_bytes(b"")
    calls = []

    def fake_render(source, out_dir, n_views=4, **kwargs):
        calls.append(str(source))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for i in range(n_views):
            p = out_dir / f"view_{i:03d}.png"
            p.write_bytes(b"")
            written.append(str(p))
        return written

    monkeypatch.setattr(J, "_render_pred_multiview", fake_render)
    ctx = _ctx(_FakeResolved(_case(), gt_mesh=mesh))

    first = J._gt_judge_views_or_render(ctx)
    assert len(first) == 4
    assert calls == [str(mesh)]
    # Ordered by view index, so PRED view i pairs with GT view i.
    assert [Path(p).name for p in first] == [f"view_{i:03d}.png" for i in range(4)]

    second = J._gt_judge_views_or_render(ctx)
    assert second == first
    assert calls == [str(mesh)]        # cached: rendered once per case, not per run


def test_missing_gt_mesh_skips_instead_of_raising(tmp_path):
    from p3dbench.metrics.judge import _gt_judge_views_or_render

    assert _gt_judge_views_or_render(_ctx(_FakeResolved(_case()))) == []
