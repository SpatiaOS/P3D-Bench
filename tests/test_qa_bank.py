"""Text-to-3D QA bank: Hub-variant selection + the demo single-variant bank.

Light, dependency-free checks (no LLM, no geometry): they exercise the eval-time
``select_bank_questions`` selection and the ``score_qa_results`` scorer, plus the
build-time ``_write_qa_bank`` normalization. The Hub ``qa.jsonl`` packs every
text_mode x format variant into one ``questions`` list; the demo/research bank is
a single variant already carrying ``split``.
"""

from __future__ import annotations

import json
from pathlib import Path

from p3dbench.data.full_builder import _write_qa_bank
from p3dbench.metrics.judge import score_qa_results, select_bank_questions

REPO = Path(__file__).resolve().parents[1]

# A minimal two-variant Hub-style bank: parametric (4 sem + 8 param) and
# descriptive (4 sem only), each in the "json" format slug.
_HUB_QUESTIONS = (
    [{"text_mode": "parametric", "format": "json", "qid": f"semantic_{i}",
      "question": "q", "options": ["a", "b", "c", "d"], "answer": "A"} for i in range(1, 5)]
    + [{"text_mode": "parametric", "format": "json", "qid": f"param_{i}",
        "question": "q", "options": ["a", "b", "c", "d"], "answer": "B"} for i in range(1, 9)]
    + [{"text_mode": "descriptive", "format": "json", "qid": f"semantic_{i}",
        "question": "q", "options": ["a", "b", "c", "d"], "answer": "C"} for i in range(1, 5)]
)
_HUB_BANK = {"uid": "0000/00000000", "questions": _HUB_QUESTIONS}


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
    assert all("split" in q for q in written["questions"])
    # qid prefix drives the split.
    by_qid = {q["qid"]: q["split"] for q in written["questions"]}
    assert by_qid["param_1"] == "param" and by_qid["semantic_1"] == "semantic"
