from __future__ import annotations

import json
import os
import copy
from dataclasses import dataclass
from pathlib import Path

import pytest

import p3dbench.data.full_builder as full_builder
from p3dbench.data.full_builder import _write_qa_bank
from p3dbench.data.loader import ResolvedCase
from p3dbench.data.schema import Case
from p3dbench.formats import get_format
from p3dbench.metrics.base import (
    PART_PROTOCOL_STATUS_KEY,
    PART_STATUS_DECOMPOSITION_UNUSABLE,
    PART_STATUS_EVALUATOR_GAP,
    PART_STATUS_FIDELITY_REJECTED,
    PART_STATUS_FIDELITY_UNAVAILABLE,
    PART_STATUS_GENERATION_INVALID,
    PART_STATUS_MEASURED,
    PART_STATUS_UNCLASSIFIED_MISSING,
    ScoreContext,
    bucket_score_for_case,
    bucket_membership,
    missing_required_metrics,
    part_required_status,
)
from p3dbench.metrics.judge import (
    QA_ANSWERER_SYSTEM_PROMPT,
    _JudgeBucket,
    _render_pred_multiview,
    answer_qa_bank,
    llm_judge_score,
)
from p3dbench.metrics.part import _compile_failure_status
from p3dbench.metrics.geometry import iou_csg
from p3dbench.pipeline import (
    FAILURE_GENERATION_INVALID,
    FAILURE_INFERENCE_GAP,
    _formal_expected_case_ids,
    _validate_formal_compiled_rows,
    score,
    summarize,
)
from p3dbench.protocol import (
    PAPER_PROMPT_GOLDEN_SHA256,
    PAPER_PROTOCOL_ID,
    file_fingerprint,
    model_call_trace,
    paper_dataset_contract,
    paper_manifest_provenance,
    qa_dataset_content_sha256,
    sha256_text,
    validate_paper_materialized_manifest,
    validate_paper_source_ids,
)
from p3dbench.text_condition import resolve_text_condition
from p3dbench.tasks.assembly_3d import TASK as ASSEMBLY_TASK


@dataclass
class _Cfg:
    name: str = "judge"
    provider: str = "openai_compatible"
    model: str = "judge-requested"
    base_url: str = "https://example.invalid/v1"


@dataclass
class _Response:
    text: str
    model: str = "judge-requested"
    usage: dict | None = None
    finish_reason: str = "stop"


class _Client:
    cfg = _Cfg()

    def __init__(self, text: str):
        self.text = text
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return _Response(self.text)


def _images(tmp_path: Path, prefix: str, n: int) -> list[str]:
    paths = []
    for index in range(n):
        path = tmp_path / f"{prefix}_{index}.png"
        path.write_bytes(f"{prefix}-{index}".encode())
        paths.append(str(path))
    return paths


def test_textdesc_uses_semantic_only_prompt_and_exact_schema(tmp_path):
    client = _Client('{"reason":"right category","semantic":8}')
    pred = _images(tmp_path, "pred", 4)
    gt = _images(tmp_path, "gt", 4)
    trace = {}
    result = llm_judge_score(
        client,
        pred,
        gt,
        condition_text="a mounting bracket",
        semantic_only=True,
        protocol_id=PAPER_PROTOCOL_ID,
        trace_out=trace,
    )
    assert result == {"semantic": 8, "reason": "right category", "error": None}
    prompt, kwargs = client.calls[0]
    assert "Your ONLY task" in prompt
    assert "Do NOT penalize PRED for geometric differences" in prompt
    assert "a mounting bracket" in prompt
    assert len(kwargs["images"]) == 8
    assert sha256_text(prompt) == PAPER_PROMPT_GOLDEN_SHA256[
        "judge_semantic_only_fixture_v1"
    ]
    assert trace["kind"] == "judge_semantic_only"
    assert trace["response"]["model_identity_matches"] is True


def test_paper_judge_rejects_partial_views_before_call(tmp_path):
    client = _Client('{"reason":"x","semantic":8}')
    result = llm_judge_score(
        client,
        _images(tmp_path, "pred", 3),
        _images(tmp_path, "gt", 4),
        semantic_only=True,
        protocol_id=PAPER_PROTOCOL_ID,
    )
    assert result["error"] and "exactly 4 PRED views" in result["error"]
    assert not client.calls


def test_semantic_only_requires_score_and_nonempty_reason(tmp_path):
    for payload, fragment in (
        ('{"semantic": 8}', "reason"),
        ('{"semantic": true, "reason": "x"}', "semantic"),
        ('{"semantic": 11, "reason": "x"}', "semantic"),
    ):
        result = llm_judge_score(
            _Client(payload),
            _images(tmp_path, f"strict_sem_pred_{fragment}", 4),
            _images(tmp_path, f"strict_sem_gt_{fragment}", 4),
            semantic_only=True,
            protocol_id=PAPER_PROTOCOL_ID,
        )
        assert result["semantic"] is None
        assert fragment in result["error"]


def test_visual_judge_includes_original_condition_image(tmp_path):
    client = _Client('{"reason":"ok","geometry":7,"semantic":8,"aesthetics":6}')
    condition = _images(tmp_path, "condition", 1)
    result = llm_judge_score(
        client,
        _images(tmp_path, "pred", 4),
        _images(tmp_path, "gt", 4),
        condition_text="assembly caption",
        condition_image_paths=condition,
        require_condition_image=True,
        enable_semantic=True,
        protocol_id=PAPER_PROTOCOL_ID,
    )
    assert result == {
        "geometry": 7,
        "semantic": 8,
        "aesthetics": 6,
        "reason": "ok",
        "error": None,
    }
    prompt, kwargs = client.calls[0]
    assert "original model-visible condition image" in prompt
    assert "Images 2..5 are PRED" in prompt
    assert "Images 6..9 are GT" in prompt
    assert "PRED view 1 vs GT view 1" in prompt
    assert "image 1 PRED" not in prompt
    assert kwargs["images"][0] == condition[0]
    assert len(kwargs["images"]) == 9
    assert sha256_text(prompt) == PAPER_PROMPT_GOLDEN_SHA256[
        "judge_visual_fixture_v1"
    ]

    missing_client = _Client(
        '{"reason":"ok","geometry":7,"semantic":8,"aesthetics":6}'
    )
    missing = llm_judge_score(
        missing_client,
        _images(tmp_path, "pred_missing_condition", 4),
        _images(tmp_path, "gt_missing_condition", 4),
        require_condition_image=True,
        protocol_id=PAPER_PROTOCOL_ID,
    )
    assert "condition image" in missing["error"]
    assert not missing_client.calls


def test_paper_qa_requires_source_bbox_four_renders_and_prompt_e(
    tmp_path, monkeypatch
):
    bank = {
        "questions": [{
            "qid": "semantic_1",
            "split": "semantic",
            "category": "features",
            "question": "Question 1?",
            "options": ["a1", "b1", "c1", "d1"],
            "answer": "A",
        }],
        "_paper_contract": {
            "qa_bank_version": 9,
            "text_mode": "parametric",
            "format": "openscad",
            "bank_sha256": "a" * 64,
        },
    }
    client = _Client('{"answers":[{"qid":"semantic_1","answer":"A"}]}')
    model_stl = tmp_path / "model.stl"
    model_stl.write_bytes(b"mesh")
    bbox_text = (
        "PREDICTION MESH BOUNDING BOX:\n"
        "  Bounding box extents: X=1.0000, Y=2.0000, Z=3.0000"
    )
    monkeypatch.setattr(
        "p3dbench.metrics.judge._extract_pred_mesh_summary",
        lambda _path: bbox_text,
    )
    try:
        answer_qa_bank(
            client,
            bank,
            _images(tmp_path, "qa", 1),
            "openscad",
            "cube();",
            pred_stl_path=model_stl,
            protocol_id=PAPER_PROTOCOL_ID,
        )
    except ValueError as exc:
        assert "exactly 4" in str(exc)
    else:
        raise AssertionError("paper QA must reject a single view")
    assert not client.calls

    renders = _images(tmp_path, "qa4", 4)
    trace = {}
    answer_qa_bank(
        client,
        bank,
        renders,
        "openscad",
        "cube();",
        artifact_label="OpenSCAD",
        artifact_name="generated.scad",
        pred_stl_path=model_stl,
        protocol_id=PAPER_PROTOCOL_ID,
        trace_out=trace,
    )
    prompt, kwargs = client.calls[0]
    assert "E. None of the above" in prompt
    assert bbox_text in prompt
    assert len(bank["questions"][0]["options"]) == 4
    assert len(kwargs["images"]) == 4
    assert trace["evidence"]["prediction_source"]["sha256"]
    assert trace["evidence"]["prediction_bbox"]["present"] is True
    assert trace["evidence"]["prediction_bbox"]["sha256"]
    assert trace["evidence"]["prediction_bbox"]["source_mesh"]["sha256"]
    assert trace["evidence"]["prediction_render_count"] == 4
    assert trace["evidence"]["qa_bank"]["bank_sha256"] == "a" * 64
    assert sha256_text(prompt) == PAPER_PROMPT_GOLDEN_SHA256[
        "qa_answer_fixture_v1"
    ]
    assert sha256_text(QA_ANSWERER_SYSTEM_PROMPT) == (
        PAPER_PROMPT_GOLDEN_SHA256["qa_answer_system_v1"]
    )

    wrong_version = {
        **bank,
        "_paper_contract": {
            **bank["_paper_contract"],
            "qa_bank_version": 8,
        },
    }
    with pytest.raises(ValueError, match="bank v9"):
        answer_qa_bank(
            client,
            wrong_version,
            _images(tmp_path, "qa_wrong_version", 4),
            "openscad",
            "cube();",
            pred_stl_path=model_stl,
            protocol_id=PAPER_PROTOCOL_ID,
        )


def test_required_metric_gap_is_not_partial_average():
    raw = {
        "chamfer_distance": 0.001,
        "f_score_005": 0.8,
        "f_score_001": 0.5,
        "normal_consistency": 0.9,
        # required IoU intentionally missing
    }
    scores = bucket_score_for_case(
        "image-to-3d", raw, True, required_buckets={"geometry"}
    )
    assert scores["geometry"] is None
    assert missing_required_metrics(
        "image-to-3d", raw, True, required_buckets={"geometry"}
    ) == {"geometry": ["iou_applicability"]}

    invalid_scores = bucket_score_for_case(
        "image-to-3d", {}, False, required_buckets={"geometry"}
    )
    assert invalid_scores["geometry"] == 0.0
    assert missing_required_metrics(
        "image-to-3d", {}, False, required_buckets={"geometry"}
    ) == {}


@pytest.mark.parametrize(
    ("status", "expected_score", "expected_missing"),
    [
        (PART_STATUS_MEASURED, 0.7, {}),
        (PART_STATUS_DECOMPOSITION_UNUSABLE, 0.0, {}),
        (PART_STATUS_FIDELITY_REJECTED, 0.0, {}),
        (
            PART_STATUS_FIDELITY_UNAVAILABLE,
            None,
            {"part": [f"{PART_PROTOCOL_STATUS_KEY}:fidelity_unavailable"]},
        ),
        (
            PART_STATUS_EVALUATOR_GAP,
            None,
            {"part": [f"{PART_PROTOCOL_STATUS_KEY}:evaluator_gap"]},
        ),
        (
            PART_STATUS_UNCLASSIFIED_MISSING,
            None,
            {"part": [f"{PART_PROTOCOL_STATUS_KEY}:unclassified_missing"]},
        ),
    ],
)
def test_formal_part_status_controls_fixed_denominator(
    status, expected_score, expected_missing
):
    raw = {
        "part_match_f1": 0.8 if status == PART_STATUS_MEASURED else None,
        "part_fs": 0.6 if status == PART_STATUS_MEASURED else None,
        PART_PROTOCOL_STATUS_KEY: status,
    }
    if status in {
        PART_STATUS_DECOMPOSITION_UNUSABLE,
        PART_STATUS_FIDELITY_REJECTED,
    }:
        raw["part_note"] = status
    score = bucket_score_for_case(
        "assembly-3d",
        raw,
        True,
        required_buckets={"part"},
    )["part"]
    if expected_score is None:
        assert score is None
    else:
        assert score == pytest.approx(expected_score)
    assert missing_required_metrics(
        "assembly-3d",
        raw,
        True,
        required_buckets={"part"},
    ) == expected_missing

    invalid = bucket_score_for_case(
        "assembly-3d",
        {},
        False,
        required_buckets={"part"},
    )
    assert invalid["part"] == 0.0


def test_textparam_panel_is_geometry5_topology3():
    membership = bucket_membership("text-to-3d", "parametric")
    assert membership["geometry"] == [
        "chamfer_distance",
        "f_score_005",
        "f_score_001",
        "normal_consistency",
        "iou",
    ]
    assert membership["topology"] == [
        "no_open_edge",
        "inverted_normal_ratio",
        "non_manifold_edge_ratio",
    ]
    assert len(membership["geometry"]) == 5
    assert len(membership["topology"]) == 3


def test_legacy_part_missing_is_unclassified_but_complete_pair_is_measured():
    assert part_required_status({
        "part_match_f1": 0.8,
        "part_fs": 0.6,
    }) == PART_STATUS_MEASURED
    assert part_required_status({
        "part_match_f1": 0.8,
        "part_fs": None,
    }) == PART_STATUS_UNCLASSIFIED_MISSING
    assert part_required_status({
        PART_PROTOCOL_STATUS_KEY: PART_STATUS_DECOMPOSITION_UNUSABLE,
    }) == PART_STATUS_UNCLASSIFIED_MISSING
    assert _compile_failure_status(["OpenSCAD timed out"]) == (
        PART_STATUS_EVALUATOR_GAP
    )
    assert _compile_failure_status(["parse failed"]) == (
        PART_STATUS_DECOMPOSITION_UNUSABLE
    )


def test_formal_part_summary_uses_full_denominator_or_blocks(
    tmp_path, monkeypatch
):
    expected_ids = ["case-1", "case-2", "case-3", "case-4"]
    contract = {
        "task_profile": "assembly-3d",
        "split": "full",
        "expected_count": len(expected_ids),
        "ordered_ids_sha256": sha256_text("\n".join(expected_ids)),
    }
    monkeypatch.setattr(
        "p3dbench.pipeline._formal_expected_case_ids",
        lambda _task, _split: (expected_ids, contract),
    )
    common = {
        "chamfer_distance": 0.001,
        "f_score_005": 0.8,
        "f_score_001": 0.5,
        "normal_consistency": 0.9,
        "iou": 0.7,
        "pred_open_edge_ratio": 0.0,
        "gt_open_edge_ratio": 0.0,
        "no_open_edge": 1.0,
        "inverted_normal_ratio": 0.0,
        "non_manifold_edge_ratio": 0.0,
        "judge_semantic": 8,
        "judge_geometry": 7,
        "judge_aesthetics": 6,
    }

    def row(case_id, *, valid=True, status=None, match_f1=None, part_fs=None):
        raw = dict(common)
        if status is not None:
            raw[PART_PROTOCOL_STATUS_KEY] = status
        raw["part_match_f1"] = match_f1
        raw["part_fs"] = part_fs
        if status in {
            PART_STATUS_DECOMPOSITION_UNUSABLE,
            PART_STATUS_FIDELITY_REJECTED,
        }:
            raw["part_note"] = status
        return {
            "id": case_id,
            "task": "assembly-3d",
            "format": "openscad",
            "model": "gpt",
            "split": "full",
            "text_mode": "parametric",
            "valid": valid,
            "failure_class": (
                None if valid else FAILURE_GENERATION_INVALID
            ),
            "buckets": [
                "valid", "geometry", "topology", "judge", "part",
            ],
            "raw_metrics": raw,
            "protocol_id": PAPER_PROTOCOL_ID,
            "case_universe": contract,
        }

    rows = [
        row(
            "case-1",
            status=PART_STATUS_MEASURED,
            match_f1=0.8,
            part_fs=0.6,
        ),
        row("case-2", valid=False),
        row("case-3", status=PART_STATUS_DECOMPOSITION_UNUSABLE),
        row("case-4", status=PART_STATUS_FIDELITY_REJECTED),
    ]
    metrics = tmp_path / "part_metrics.jsonl"
    output = tmp_path / "part_summary.json"
    _write_jsonl(metrics, rows)
    summarize(metrics, out=output)
    group = json.loads(output.read_text(encoding="utf-8"))["groups"][0]
    assert group["promotion_ready"] is True
    assert group["buckets"]["part"] == pytest.approx(0.175)
    assert group["metric_coverage"]["part"]["denominator_count"] == 4
    assert group["metric_coverage"]["part"]["gap_count"] == 0
    assert group["metric_coverage"]["part"]["status_counts"] == {
        PART_STATUS_DECOMPOSITION_UNUSABLE: 1,
        PART_STATUS_FIDELITY_REJECTED: 1,
        PART_STATUS_GENERATION_INVALID: 1,
        PART_STATUS_MEASURED: 1,
    }

    rows[2] = row(
        "case-3",
        status=PART_STATUS_UNCLASSIFIED_MISSING,
    )
    _write_jsonl(metrics, rows)
    summarize(metrics, out=output)
    blocked = json.loads(
        output.read_text(encoding="utf-8")
    )["groups"][0]
    assert blocked["promotion_ready"] is False
    assert "required_metric_gap" in blocked["promotion_blockers"]
    assert "part" not in blocked["buckets"]
    assert blocked["metric_coverage"]["part"]["denominator_count"] == 3
    assert blocked["metric_coverage"]["part"]["gap_count"] == 1


def test_iou_is_inapplicable_for_open_edges_but_missing_when_eligible():
    common = {
        "chamfer_distance": 0.001,
        "f_score_005": 0.8,
        "f_score_001": 0.5,
        "normal_consistency": 0.9,
    }
    open_edge = {
        **common,
        "pred_open_edge_ratio": 0.1,
        "gt_open_edge_ratio": 0.0,
        "iou": None,
    }
    scores = bucket_score_for_case(
        "image-to-3d", open_edge, True, required_buckets={"geometry"}
    )
    assert scores["geometry"] is not None
    assert missing_required_metrics(
        "image-to-3d", open_edge, True, required_buckets={"geometry"}
    ) == {}

    one_known_open = dict(open_edge)
    one_known_open.pop("pred_open_edge_ratio")
    one_known_open["gt_open_edge_ratio"] = 0.2
    assert missing_required_metrics(
        "image-to-3d", one_known_open, True,
        required_buckets={"geometry"},
    ) == {}

    eligible = {
        **common,
        "pred_open_edge_ratio": 0.0,
        "gt_open_edge_ratio": 0.0,
        "iou": None,
    }
    assert bucket_score_for_case(
        "image-to-3d", eligible, True, required_buckets={"geometry"}
    )["geometry"] is None
    assert missing_required_metrics(
        "image-to-3d", eligible, True, required_buckets={"geometry"}
    ) == {"geometry": ["iou"]}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _formal_metric_row(case_id: str, contract: dict, *, open_edge: bool) -> dict:
    return {
        "id": case_id,
        "task": "image-to-3d",
        "format": "openscad",
        "model": "gpt",
        "split": "full",
        "text_mode": "parametric",
        "valid": True,
        "buckets": ["valid", "geometry", "topology", "judge"],
        "raw_metrics": {
            "chamfer_distance": 0.001,
            "f_score_005": 0.8,
            "f_score_001": 0.5,
            "normal_consistency": 0.9,
            "iou": None if open_edge else 0.7,
            "pred_open_edge_ratio": 0.1 if open_edge else 0.0,
            "gt_open_edge_ratio": 0.0,
            "no_open_edge": 0.0 if open_edge else 1.0,
            "inverted_normal_ratio": 0.0,
            "non_manifold_edge_ratio": 0.0,
            "judge_semantic": 8,
            "judge_geometry": 7,
            "judge_aesthetics": 6,
        },
        "protocol_id": PAPER_PROTOCOL_ID,
        "case_universe": contract,
    }


def test_formal_summary_audits_universe_and_iou_eligibility(
    tmp_path, monkeypatch
):
    expected_ids = ["case-1", "case-2"]
    contract = {
        "task_profile": "image-to-3d",
        "split": "full",
        "expected_count": 2,
        "ordered_ids_sha256": sha256_text("\n".join(expected_ids)),
    }
    monkeypatch.setattr(
        "p3dbench.pipeline._formal_expected_case_ids",
        lambda _task, _split: (expected_ids, contract),
    )
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [
        _formal_metric_row("case-1", contract, open_edge=False),
        _formal_metric_row("case-2", contract, open_edge=True),
    ])
    out = tmp_path / "summary.json"
    summarize(metrics, out=out)
    group = json.loads(out.read_text())["groups"][0]
    assert group["promotion_ready"] is True
    assert group["case_universe"]["complete"] is True
    assert group["metric_coverage"]["iou"] == {
        "eligible_cases": 1,
        "measured_cases": 1,
        "inapplicable_cases": 1,
        "generation_invalid_cases": 0,
        "gap_count": 0,
        "gaps": [],
    }

    _write_jsonl(metrics, [
        _formal_metric_row("case-1", contract, open_edge=False),
        _formal_metric_row("case-1", contract, open_edge=True),
    ])
    summarize(metrics, out=out)
    blocked = json.loads(out.read_text())["groups"][0]
    assert blocked["promotion_ready"] is False
    assert "case_universe_mismatch" in blocked["promotion_blockers"]
    assert blocked["case_universe"]["duplicate_ids"] == ["case-1"]
    assert blocked["case_universe"]["missing_ids"] == ["case-2"]

    mixed_a = _formal_metric_row("case-1", contract, open_edge=False)
    mixed_b = _formal_metric_row("case-2", contract, open_edge=True)
    mixed_b["text_mode"] = "descriptive"
    mixed_b["buckets"] = ["valid", "geometry", "topology"]
    _write_jsonl(metrics, [mixed_a, mixed_b])
    summarize(metrics, out=out)
    mixed = json.loads(out.read_text())["groups"][0]
    assert mixed["promotion_ready"] is False
    assert "mixed_text_mode" in mixed["promotion_blockers"]
    assert "mixed_metric_panel" in mixed["promotion_blockers"]


def test_formal_text_manifest_requires_exactly_400_cases(tmp_path, monkeypatch):
    manifest = tmp_path / "text_to_3d_full.jsonl"
    _write_jsonl(manifest, [{"id": f"case-{index}"} for index in range(399)])
    monkeypatch.setattr(
        "p3dbench.pipeline.manifest_path",
        lambda _task, _split: manifest,
    )
    with pytest.raises(ValueError, match="exactly 400"):
        _formal_expected_case_ids("text-to-3d", "full")


def test_dataset_contract_matches_all_three_local_uid_releases():
    root = Path(
        os.environ.get(
            "P3DBENCH_DATASET_FIXTURE",
            str(Path(__file__).resolve().parents[2] / "P3D-Bench"),
        )
    )
    manifest = root / "DATASET_MANIFEST.json"
    if not manifest.is_file():
        pytest.skip("local release fixture unavailable")
    contract = paper_dataset_contract()
    assert file_fingerprint(str(manifest))["sha256"] == (
        contract["dataset_manifest_sha256"]
    )
    for task, token, expected in (
        ("text-to-3d", "text_to_3d", 400),
        ("image-to-3d", "image_to_3d", 400),
        ("assembly-3d", "assembly_3d", 203),
    ):
        rows = [
            json.loads(line)
            for line in (root / "data" / token / "uids.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        result = validate_paper_source_ids(
            task, [row["uid"] for row in rows]
        )
        assert result["expected_count"] == expected
    qa_path = root / contract["qa"]["path"]
    assert file_fingerprint(str(qa_path))["sha256"] == (
        contract["qa"]["source_sha256"]
    )
    qa_rows = [
        json.loads(line)
        for line in qa_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert sum(len(row["questions"]) for row in qa_rows) == 12800
    assert qa_dataset_content_sha256(qa_rows) == (
        contract["qa"]["normalized_content_sha256"]
    )

    image_ids = [
        json.loads(line)["uid"]
        for line in (root / "data/image_to_3d/uids.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    image_rows = [
        {
            "id": f"p3d_image-to-3d_{index:06d}",
            "task": "image-to-3d",
            "split": "full",
            "metadata": {
                "source_id": uid,
                "paper_dataset_contract": paper_manifest_provenance(
                    "image-to-3d"
                ),
            },
        }
        for index, uid in enumerate(image_ids)
    ]
    validated = validate_paper_materialized_manifest(
        "image-to-3d", image_rows, split="full"
    )
    assert validated["expected_count"] == 400
    reordered = list(image_rows)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="case ID/order"):
        validate_paper_materialized_manifest(
            "image-to-3d", reordered, split="full"
        )
    wrong_sources = json.loads(json.dumps(image_rows))
    wrong_sources[0]["metadata"]["source_id"], wrong_sources[1]["metadata"][
        "source_id"
    ] = (
        wrong_sources[1]["metadata"]["source_id"],
        wrong_sources[0]["metadata"]["source_id"],
    )
    with pytest.raises(ValueError, match="source-ID order/content digest"):
        validate_paper_materialized_manifest(
            "image-to-3d", wrong_sources, split="full"
        )


def test_shared_descriptive_condition_is_exact_generation_condition():
    case = Case.from_dict({
        "id": "case",
        "task": "text-to-3d",
        "split": "full",
        "input": {"text": "PARAMETRIC CONDITION"},
        "target": {},
        "metadata": {"text_desc": "  DESCRIPTIVE CONDITION  "},
    })
    assert resolve_text_condition(case, "parametric") == "PARAMETRIC CONDITION"
    assert resolve_text_condition(case, "descriptive") == "DESCRIPTIVE CONDITION"


def test_trace_recursively_removes_endpoints_and_credentials():
    client = _Client("{}")
    response = _Response("{}")
    trace = model_call_trace(
        kind="fixture",
        client=client,
        response=response,
        prompt="prompt",
        images=[],
        request_overrides={
            "max_tokens": 4096,
            "safe": {"temperature": 0},
            "nested": {
                "endpoint": "https://private.invalid",
                "credentials": {"api_key": "do-not-write"},
                "headers": {"Authorization": "Bearer secret"},
            },
        },
    )
    serialized = json.dumps(trace)
    assert "private.invalid" not in serialized
    assert "do-not-write" not in serialized
    assert "Bearer secret" not in serialized
    assert trace["request"]["safe"]["temperature"] == 0
    assert trace["request"]["max_tokens"] == 4096


def test_textparam_judge_is_question_level_micro_average():
    score = bucket_score_for_case(
        "text-to-3d",
        {"qa_semantic": 0.25, "qa_param": 0.75},
        True,
        "parametric",
        required_buckets={"judge"},
    )
    assert score["judge"] == pytest.approx((0.25 + 2 * 0.75) / 3)


def test_formal_renderer_uses_only_frozen_blender_profile(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        "p3dbench.render.occ.render_multiview",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("formal renderer must not use OCC")
        ),
    )

    def fake_blender(*args, **kwargs):
        calls.append(kwargs)
        return [str(tmp_path / f"view_{index:03d}.png") for index in range(4)]

    monkeypatch.setattr(
        "p3dbench.render.blender.render_multiview", fake_blender
    )
    views = _render_pred_multiview(
        "model.stl",
        tmp_path,
        protocol_id=PAPER_PROTOCOL_ID,
    )
    assert len(views) == 4
    assert calls == [{
        "n_views": 4,
        "resolution": 768,
        "samples": 128,
        "seed": 42,
    }]


@pytest.mark.parametrize("fmt", ["cadquery", "openscad"])
def test_part_decomposition_prompt_matches_frozen_golden(fmt):
    prompt = ASSEMBLY_TASK.build_decompose_prompt(
        get_format(fmt),
        "FIXTURE_CODE",
        True,
    )
    assert sha256_text(prompt) == PAPER_PROMPT_GOLDEN_SHA256[
        f"part_decompose_{fmt}_fixture_v1"
    ]


def test_judge_bucket_preserves_scientific_and_qa_evidence(
    tmp_path, monkeypatch
):
    questions = [
        {
            "text_mode": "descriptive",
            "format": "json",
            "qid": f"semantic_{index}",
            "question": f"question {index}",
            "options": ["a", "b", "c", "d"],
            "answer": "A",
        }
        for index in range(1, 5)
    ]
    bank_path = tmp_path / "qa.json"
    _write_qa_bank("0000/00000000", questions, bank_path, overwrite=True)
    stl = tmp_path / "pred.stl"
    stl.write_bytes(b"mesh")
    case = Case.from_dict({
        "id": "case",
        "task": "text-to-3d",
        "split": "full",
        "input": {"text": "parametric"},
        "target": {"qa_bank_path": "qa.json"},
        "metadata": {
            "source_id": "0000/00000000",
            "text_desc": "descriptive",
        },
    })
    ctx = ScoreContext(
        task="text-to-3d",
        fmt="minimal-json",
        case=ResolvedCase(case, tmp_path),
        compiled={"stl": str(stl)},
        work_dir=tmp_path / "work",
        judge_client=object(),
        protocol_id=PAPER_PROTOCOL_ID,
        shared={"stage1_code": "{}", "text_mode": "descriptive"},
    )
    payload = {
        "answers": [
            {"qid": f"semantic_{index}", "answer": "A"}
            for index in range(1, 5)
        ]
    }
    monkeypatch.setattr(
        "p3dbench.metrics.judge._render_pred_multiview",
        lambda *args, **kwargs: [
            str(tmp_path / f"view_{index:03d}.png") for index in range(4)
        ],
    )
    monkeypatch.setattr(
        "p3dbench.metrics.judge.answer_qa_bank",
        lambda *args, **kwargs: payload,
    )
    bucket = _JudgeBucket()
    monkeypatch.setattr(
        bucket,
        "_descriptive_judge_semantic",
        lambda *_args: {
            "semantic": 8,
            "reason": "correct category",
            "error": None,
        },
    )
    raw = bucket.score(ctx)
    assert raw["judge_semantic_result"] == {
        "semantic": 8,
        "reason": "correct category",
        "error": None,
    }
    assert raw["qa_answer_payload"] == payload
    assert len(raw["qa_results"]) == 4
    assert raw["qa_scoring"]["semantic_correct"] == 4


def test_summary_task_profile_equal_means_supported_formats(tmp_path):
    common = {
        "chamfer_distance": 0.001,
        "f_score_005": 0.8,
        "f_score_001": 0.5,
        "normal_consistency": 0.9,
        "pred_open_edge_ratio": 0.0,
        "gt_open_edge_ratio": 0.0,
        "iou": 0.7,
        "no_open_edge": 1.0,
        "inverted_normal_ratio": 0.0,
        "non_manifold_edge_ratio": 0.0,
    }
    rows = []
    for fmt, qa_s, qa_p in (
        ("minimal-json", 0.25, 0.75),
        ("openscad", 0.75, 0.25),
    ):
        rows.append({
            "id": fmt,
            "task": "text-to-3d",
            "format": fmt,
            "model": "gpt",
            "split": "demo",
            "text_mode": "parametric",
            "valid": True,
            "buckets": ["valid", "geometry", "topology", "judge"],
            "raw_metrics": {
                **common,
                "qa_semantic": qa_s,
                "qa_param": qa_p,
            },
            "protocol_id": None,
        })
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, rows)
    output = tmp_path / "summary.json"
    summarize(metrics, out=output)
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert len(summary["groups"]) == 2
    profile = summary["task_profiles"][0]
    expected = (
        (0.25 + 2 * 0.75) / 3
        + (0.75 + 2 * 0.25) / 3
    ) / 2
    assert profile["format_aggregation"] == "equal_mean"
    assert profile["complete_formats"] is True
    assert profile["buckets"]["judge"] == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda case: case["metadata"].__setitem__("source_id", "tampered"),
        lambda case: case["input"].__setitem__("text", "tampered"),
        lambda case: case["target"].__setitem__(
            "mesh_path", "targets/mesh/tampered.stl"
        ),
    ],
)
def test_formal_compiled_case_binding_rejects_any_embedded_case_tamper(
    monkeypatch, mutate
):
    expected_case = Case.from_dict({
        "id": "case-1",
        "task": "image-to-3d",
        "split": "full",
        "input": {"text": "", "image_paths": ["inputs/case-1.png"]},
        "target": {
            "format": "step",
            "step_path": "targets/step/case-1.step",
            "mesh_path": "targets/mesh/case-1.stl",
        },
        "metadata": {"source_id": "source-1"},
    }).to_dict()
    contract = {"expected_count": 1}
    monkeypatch.setattr(
        "p3dbench.pipeline._formal_expected_case_rows",
        lambda _task, _split: ([expected_case], contract),
    )
    row = {
        "id": "case-1",
        "task": "image-to-3d",
        "format": "openscad",
        "model": "gpt",
        "split": "full",
        "text_mode": "parametric",
        "case": copy.deepcopy(expected_case),
        "valid": True,
        "failure_class": None,
    }
    mutate(row["case"])
    with pytest.raises(ValueError, match="case/manifest binding"):
        _validate_formal_compiled_rows(
            [row], ["valid", "geometry", "topology", "judge"]
        )


def test_formal_inference_gap_blocks_without_worst_fill(
    tmp_path, monkeypatch
):
    expected_ids = ["case-1"]
    contract = {
        "task_profile": "image-to-3d",
        "split": "full",
        "expected_count": 1,
        "ordered_ids_sha256": sha256_text("case-1"),
    }
    monkeypatch.setattr(
        "p3dbench.pipeline._formal_expected_case_ids",
        lambda _task, _split: (expected_ids, contract),
    )
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [{
        "id": "case-1",
        "task": "image-to-3d",
        "format": "openscad",
        "model": "gpt",
        "split": "full",
        "text_mode": "parametric",
        "valid": False,
        "failure_class": FAILURE_INFERENCE_GAP,
        "buckets": ["valid", "geometry", "topology", "judge"],
        "raw_metrics": {"_pipeline_error": "inference_gap: timeout"},
        "protocol_id": PAPER_PROTOCOL_ID,
        "case_universe": contract,
    }])
    output = tmp_path / "summary.json"
    summarize(metrics, out=output)
    group = json.loads(output.read_text(encoding="utf-8"))["groups"][0]
    assert "inference_or_evaluator_gap" in group["promotion_blockers"]
    assert group["failure_gap_count"] == 1
    assert group["valid_rate"] is None
    assert group["buckets"] == {}
    assert group["score"] is None
    assert group["metric_coverage"]["iou"]["generation_invalid_cases"] == 0


def test_text_gt_builder_materializes_frozen_four_view_profile(
    tmp_path, monkeypatch
):
    root = tmp_path / "full"
    mesh = tmp_path / "gt.stl"
    mesh.write_bytes(b"mesh")
    calls = []

    def fake_render(_mesh, output_dir, **kwargs):
        calls.append(kwargs)
        paths = []
        for index in range(4):
            path = Path(output_dir) / f"view_{index:03d}.png"
            path.write_bytes(f"view-{index}".encode())
            paths.append(str(path))
        return paths

    monkeypatch.setattr(full_builder, "FULL_ROOT", root)
    monkeypatch.setattr(
        "p3dbench.render.blender.render_multiview", fake_render
    )
    monkeypatch.setattr(
        full_builder, "_image_size", lambda _path: (768, 768)
    )
    paths, contract, error = full_builder._materialize_text_gt_renders(
        mesh, "case", overwrite=False
    )
    assert error is None
    assert len(paths) == 4
    assert calls == [{
        "n_views": 4,
        "resolution": 768,
        "samples": 128,
        "seed": 42,
    }]
    assert contract["renderer"] == "blender_clay"
    assert contract["source_mesh_sha256"] == file_fingerprint(
        str(mesh)
    )["sha256"]
    assert all((root / path).is_file() for path in paths)


def test_iou_csg_exception_is_unavailable_not_zero(monkeypatch):
    mesh = type("_Mesh", (), {"is_volume": True})()
    monkeypatch.setattr(
        "p3dbench.metrics.geometry._to_manifold",
        lambda _mesh: (_ for _ in ()).throw(RuntimeError("evaluator failed")),
    )
    value, pred_fixed, gt_fixed = iou_csg(
        mesh, mesh, skip_normalize=True
    )
    assert value is None
    assert pred_fixed is False and gt_fixed is False


@pytest.mark.parametrize(
    "payload,error_fragment",
    [
        (
            '{"reason":"x","geometry":true,"semantic":8,"aesthetics":6}',
            "geometry",
        ),
        (
            '{"reason":"x","geometry":7,"semantic":8}',
            "aesthetics",
        ),
        (
            '{"reason":"x","geometry":7,"semantic":11,"aesthetics":6}',
            "semantic",
        ),
        (
            '{"geometry":7,"semantic":8,"aesthetics":6}',
            "reason",
        ),
    ],
)
def test_joint_visual_judge_requires_every_enabled_axis_and_reason(
    tmp_path, payload, error_fragment
):
    client = _Client(payload)
    result = llm_judge_score(
        client,
        _images(tmp_path, "strict_pred", 4),
        _images(tmp_path, "strict_gt", 4),
        condition_image_paths=_images(tmp_path, "strict_condition", 1),
        require_condition_image=True,
        enable_semantic=True,
        protocol_id=PAPER_PROTOCOL_ID,
    )
    assert error_fragment in result["error"]
    assert result["geometry"] is None
    assert result["aesthetics"] is None
    assert result["semantic"] is None


def test_evaluation_meta_path_is_always_relative(tmp_path, monkeypatch):
    case = Case.from_dict({
        "id": "case",
        "task": "image-to-3d",
        "split": "demo",
        "input": {},
        "target": {},
        "metadata": {},
    }).to_dict()
    compiled = tmp_path / "compiled.jsonl"
    _write_jsonl(compiled, [{
        "id": "case",
        "task": "image-to-3d",
        "format": "openscad",
        "model": "gpt",
        "split": "demo",
        "text_mode": "parametric",
        "case": case,
        "code": "cube();",
        "compile": {"valid": True},
        "valid": True,
        "failure_class": None,
    }])

    class _TraceBucket:
        def score(self, ctx):
            ctx.shared["evaluation_calls"] = [{"kind": "fixture"}]
            return {}

    monkeypatch.setattr(
        "p3dbench.pipeline.get_metric_bucket", lambda _name: _TraceBucket()
    )
    output = tmp_path / "out" / "metrics.jsonl"
    score(
        compiled,
        "valid",
        out=output,
        work_dir=tmp_path / "unrelated-absolute-work",
    )
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    meta_path = Path(row["evaluation_meta_path"])
    assert not meta_path.is_absolute()
    assert (output.parent / meta_path).is_file()
