"""Regression cases for the 2026-09-16 aggregation revision."""

import json

import pytest

from p3dbench.metrics.base import bucket_score_for_case
from p3dbench.pipeline import summarize
from p3dbench.utils import write_jsonl


@pytest.mark.parametrize("task", ["text-to-3d", "image-to-3d", "assembly-3d"])
def test_auxiliary_f005_cannot_change_geo(task):
    raw = {"chamfer_distance": .005, "f_score_001": .2,
           "normal_consistency": .8, "iou": .1}
    for f005 in (None, 0., 1.):
        actual = bucket_score_for_case(task, {**raw, "f_score_005": f005}, True)
        assert actual["geometry"] == pytest.approx(.4)


def test_unavailable_iou_differs_from_measured_zero_and_invalid_output():
    raw = {"chamfer_distance": 0., "f_score_001": 1., "normal_consistency": 1.}
    assert bucket_score_for_case("text-to-3d", {**raw, "iou": None}, True)["geometry"] == 1.
    assert bucket_score_for_case("text-to-3d", {**raw, "iou": 0.}, True)["geometry"] == .75
    assert bucket_score_for_case("text-to-3d", raw, False)["geometry"] == 0.


def _row(task="image-to-3d", mode="parametric", **raw):
    return {"id": "case", "task": task, "format": "openscad", "model": "m",
            "text_mode": mode, "valid": True, "llm_failed": False,
            "buckets": ["valid", "geometry", "topology", "judge", "part"],
            "raw_metrics": raw}


def _summary(tmp_path, rows):
    source, target = tmp_path / "metrics.jsonl", tmp_path / "summary.json"
    write_jsonl(source, rows)
    summarize(source, out=target)
    return json.loads(target.read_text())


@pytest.mark.parametrize("task, expected", [("image-to-3d", 30.), ("assembly-3d", 40.)])
def test_headline_excludes_topology_but_reports_it(tmp_path, task, expected):
    raw = {"chamfer_distance": .008, "f_score_001": .2,
           "normal_consistency": .2, "iou": .2,
           "judge_geometry": 4.6, "judge_semantic": 4.6, "judge_aesthetics": 4.6,
           "part_match_f1": .4, "part_fs": .8, "part_fs_normalized": 0.}
    for no_open_edge, error in [(1., 0.), (0., 1.)]:
        summary = _summary(tmp_path, [_row(task, **raw, no_open_edge=no_open_edge,
            inverted_normal_ratio=error, non_manifold_edge_ratio=error)])
        group = summary["groups"][0]
        assert group["score"] == expected
        assert group["buckets"]["topology"] == no_open_edge
        assert summary["aggregation_revision"] == "p3d-scoring-20260916"


def test_topology_only_invalid_case_does_not_invent_headline_buckets(tmp_path):
    row = _row(no_open_edge=1., inverted_normal_ratio=0., non_manifold_edge_ratio=0.)
    row["buckets"] = ["valid", "topology"]
    failed = {**row, "id": "invalid", "valid": False, "raw_metrics": {}}
    group = _summary(tmp_path, [row, failed])["groups"][0]
    assert group["buckets"] == {"topology": .5}
    assert group["score"] is None
    assert group["valid_rate"] == .5


def test_text_modes_are_separate_and_legacy_rows_default_to_parametric(tmp_path):
    param = _row("text-to-3d", chamfer_distance=.008, f_score_001=.2,
                 normal_consistency=.2, iou=.2, qa_semantic=.4, qa_param=.4)
    param.pop("text_mode")
    desc = _row("text-to-3d", "descriptive", qa_semantic=1., judge_semantic=10.)
    groups = {g["text_mode"]: g for g in _summary(tmp_path, [param, desc])["groups"]}
    assert groups["parametric"]["n_cases"] == groups["descriptive"]["n_cases"] == 1
    assert groups["parametric"]["score"] == 30.
    assert groups["descriptive"]["score"] == 100.
    assert set(groups["descriptive"]["buckets"]) == {"judge"}
