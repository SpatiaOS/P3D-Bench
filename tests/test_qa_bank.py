"""Text-to-3D QA bank: Hub-variant selection + the demo single-variant bank.

Light, dependency-free checks (no LLM, no geometry): they exercise the eval-time
``select_bank_questions`` selection and the ``score_qa_results`` scorer, plus the
build-time ``_write_qa_bank`` normalization. The Hub ``qa.jsonl`` packs every
text_mode x format variant into one ``questions`` list; the demo/research bank is
a single variant already carrying ``split``.
"""

from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from p3dbench.data.full_builder import _write_qa_bank
from p3dbench.metrics.judge import (
    score_qa_results,
    select_bank_questions,
    validate_paper_qa_bank,
)
from p3dbench.protocol import (
    PAPER_QA_BANK_VERSION,
    qa_questions_content_sha256,
)

REPO = Path(__file__).resolve().parents[1]

# A minimal two-variant Hub-style bank: parametric (4 sem + 8 param) and
# descriptive (4 sem only), each in the "json" format slug.
_HUB_QUESTIONS = (
    [{"text_mode": "parametric", "format": "json", "qid": f"semantic_{i}",
      "split": "semantic", "source_text_level": "detailed",
      "question": "q", "options": ["a", "b", "c", "d"], "answer": "A"} for i in range(1, 5)]
    + [{"text_mode": "parametric", "format": "json", "qid": f"param_{i}",
        "split": "param", "source_text_level": "parametric_detail",
        "question": "q", "options": ["a", "b", "c", "d"], "answer": "B"} for i in range(1, 9)]
    + [{"text_mode": "descriptive", "format": "json", "qid": f"semantic_{i}",
        "split": "semantic", "source_text_level": "detailed",
        "question": "q", "options": ["a", "b", "c", "d"], "answer": "C"} for i in range(1, 5)]
)
_HUB_BANK = {
    "uid": "0000/00000000",
    "qa_bank_version": PAPER_QA_BANK_VERSION,
    "questions": _HUB_QUESTIONS,
}


def _materialized_bank(tmp_path: Path, questions=None) -> dict:
    dst = tmp_path / "paper-bank.json"
    _write_qa_bank(
        "0000/00000000",
        questions or _HUB_QUESTIONS,
        dst,
        overwrite=True,
    )
    return json.loads(dst.read_text(encoding="utf-8"))


def test_hub_variant_selection():
    # parametric/minimal-json -> full 12-question variant (minimal-json maps to json).
    q = select_bank_questions(_HUB_BANK, "parametric", "minimal-json")
    assert len(q) == 12
    assert sum(x["split"] == "semantic" for x in q) == 4
    assert sum(x["split"] == "param" for x in q) == 8

    # descriptive -> 4 semantic-only.
    qd = select_bank_questions(_HUB_BANK, "descriptive", "minimal-json")
    assert len(qd) == 4 and all(x["split"] == "semantic" for x in qd)

    # A format with no variant (text-to-3d never runs it) -> empty => clean skip,
    # never a mix of variants.
    assert select_bank_questions(_HUB_BANK, "parametric", "cadquery") == []


def test_demo_single_variant_passthrough():
    bank = json.loads((REPO / "data/demo/targets/qa/p3d_text-to-3d_000000.json").read_text())
    q = select_bank_questions(bank, "parametric", "minimal-json")
    assert len(q) == 12 and all("split" in x for x in q)


def test_score_qa_results_all_correct():
    q = select_bank_questions(_HUB_BANK, "parametric", "minimal-json")
    payload = {"answers": [{"qid": x["qid"], "answer": x["answer"]} for x in q]}
    _rows, metrics = score_qa_results({"questions": q}, payload)
    assert metrics["semantic_accuracy"] == 1.0
    assert metrics["param_accuracy"] == 1.0
    assert metrics["overall_accuracy"] == 1.0


def test_write_qa_bank_fills_split(tmp_path):
    dst = tmp_path / "bank.json"
    _write_qa_bank("0000/00000000", _HUB_QUESTIONS, dst, overwrite=False)
    written = json.loads(dst.read_text())
    assert written["uid"] == "0000/00000000"
    assert written["qa_bank_version"] == PAPER_QA_BANK_VERSION
    assert all("split" in q for q in written["questions"])
    assert all("source_text_level" in q for q in written["questions"])
    assert written["source_contract"]["source_sha256"]
    assert written["content_sha256"] == qa_questions_content_sha256(
        written["questions"]
    )
    # qid prefix drives the split.
    by_qid = {q["qid"]: q["split"] for q in written["questions"]}
    assert by_qid["param_1"] == "param" and by_qid["semantic_1"] == "semantic"


def test_paper_bank_validation_is_v9_and_strict(tmp_path):
    bank = _materialized_bank(tmp_path)
    questions = validate_paper_qa_bank(
        bank,
        text_mode="parametric",
        fmt="minimal-json",
        expected_uid="0000/00000000",
    )
    assert len(questions) == 12

    stale = dict(bank, qa_bank_version=8)
    try:
        validate_paper_qa_bank(stale, text_mode="parametric", fmt="minimal-json")
    except ValueError as exc:
        assert "v9" in str(exc)
    else:
        raise AssertionError("stale bank must be rejected")


def test_paper_bank_rejects_duplicate_options_and_e_ground_truth(tmp_path):
    bad_questions = [dict(q) for q in _HUB_QUESTIONS]
    bad_questions[0] = dict(
        bad_questions[0],
        options=["same", "same", "c", "d"],
        answer="E",
    )
    bad = _materialized_bank(tmp_path)
    bad["questions"] = bad_questions
    bad["content_sha256"] = qa_questions_content_sha256(bad_questions)
    try:
        validate_paper_qa_bank(bad, text_mode="parametric", fmt="minimal-json")
    except ValueError as exc:
        assert "distinct" in str(exc) or "A-D" in str(exc)
    else:
        raise AssertionError("invalid A-D bank must be rejected")


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda questions: questions.__setitem__(
                slice(0, 2), list(reversed(questions[:2]))
            ),
            "qid order",
        ),
        (
            lambda questions: questions[0].__setitem__(
                "source_text_level", "abstract"
            ),
            "source_text_level",
        ),
        (
            lambda questions: questions[0].__setitem__(
                "text_mode", "descriptive"
            ),
            "questions",
        ),
        (
            lambda questions: questions[0].__setitem__("format", "openscad"),
            "questions",
        ),
    ],
)
def test_paper_bank_rejects_wrong_order_source_mode_or_format(
    tmp_path, mutator, message
):
    questions = copy.deepcopy(_HUB_QUESTIONS)
    mutator(questions)
    bank = _materialized_bank(tmp_path)
    bank["questions"] = questions
    bank["content_sha256"] = qa_questions_content_sha256(questions)
    with pytest.raises(ValueError, match=message):
        validate_paper_qa_bank(
            bank, text_mode="parametric", fmt="minimal-json"
        )
