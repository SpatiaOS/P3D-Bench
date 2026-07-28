"""Judge bucket — VLM-as-Judge visual panel + Text-to-3D QA.

Two evaluation modes, picked from ``ctx.task`` inside :class:`_JudgeBucket`:

* **Visual judge** (image-to-3d / assembly-3d): render 4 tetrahedral views of the
  predicted mesh and pair them index-for-index against the case's 4 GT renders,
  then ask the judge for J-Geo / J-Sem / J-Aes on a 1-10 scale in a single call
  (PRED views first, GT views second). Strict pairing: if either side does not
  have exactly 4 views, the judge is skipped (all judge keys -> ``None``) rather
  than run partially — a partial set would silently mis-pair viewpoints.

* **QA** (text-to-3d): a frozen v9 MCQ bank ships with each case. The Hub bank
  (``data/text_to_3d/qa.jsonl``) packs every text_mode×format variant; the
  eval-time path selects the active run's variant via
  :func:`select_bank_questions` — 12 questions (4 semantic + 8 param) in
  parametric mode, 4 semantic-only in descriptive mode. The predicted source,
  measured bounding box, and exactly 4 canonical predicted renders are handed
  to the answerer (an "option E / none-of-the-above" is appended at answer time
  only). Accuracy over the semantic and param splits gives QA-S / QA-P.

The judge / answerer / scoring algorithms, prompts, rubrics, thresholds and the
JSON parser are reproduced verbatim from the research evaluation code. All
generation/verification of QA banks is intentionally NOT included — banks are
pre-built and shipped alongside the data (Hub ``qa.jsonl`` or a local prebuilt
bank); this module is the eval-time path only.

Heavy/optional dependencies (trimesh for the bbox summary, the render backends)
are imported lazily so the module imports cleanly without them.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..protocol import (
    PAPER_CANONICAL_VIEW_COUNT,
    PAPER_PROTOCOL_ID,
    PAPER_QA_BANK_VERSION,
    PAPER_RENDER_RESOLUTION,
    PAPER_RENDER_SAMPLES,
    PAPER_RENDER_SEED,
    append_call_trace,
    file_fingerprint,
    model_call_trace,
    sha256_text,
    validate_materialized_qa_bank_contract,
)
from ..text_condition import resolve_text_condition
from .base import MetricBucket, ScoreContext

logger = logging.getLogger(__name__)


# ==========================================================================
# Shared JSON extraction
# (the richer 3-strategy text-to-3d QA parser; the llm_judge brace-scan is a
#  strict subset of strategy 3, so this single parser serves both paths.)
# ==========================================================================
def extract_json_object(text: str) -> Optional[dict]:
    """Extract the first valid JSON object from a model response.

    Tries three strategies in order:
    1. Strip markdown fences and ``json.loads`` the full content.
    2. Regex-extract the first ```json ... ``` block.
    3. Fall back to brace-depth scanning.
    """
    # Strategy 1: strip markdown fences and try full parse
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), count=1)
    stripped = re.sub(r"\n?```\s*$", "", stripped.strip(), count=1)
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: extract from ```json ... ``` block
    fence_match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: brace-depth scanning (original llm_judge fallback)
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    break
    return None


def _generate_text(client, prompt: str, *, images=None, system=None,
                   temperature=None, max_tokens=None, timeout=None,
                   trace_kind: Optional[str] = None,
                   trace_out: Optional[dict] = None) -> str:
    """Call a :class:`p3dbench.models.ModelClient` and return its response text."""
    image_list = list(images) if images else []
    resp = client.generate(
        prompt,
        images=image_list or None,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if trace_out is not None and trace_kind:
        trace_out.update(model_call_trace(
            kind=trace_kind,
            client=client,
            response=resp,
            prompt=prompt,
            images=image_list,
            system=system,
            request_overrides={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_s": timeout,
                "image_count": len(image_list),
            },
        ))
    return getattr(resp, "text", "") or ""


# ==========================================================================
# 1. VISUAL JUDGE  (J-Geo / J-Sem / J-Aes)
# ==========================================================================
def judge_default_result(error: Optional[str] = None) -> dict:
    """Schema returned by :func:`llm_judge_score` (also used when skipping)."""
    return {
        "geometry": None,
        "semantic": None,
        "aesthetics": None,
        "reason": "",
        "error": error,
    }


def semantic_judge_default_result(error: Optional[str] = None) -> dict:
    """Schema for the descriptive Text-to-3D semantic-only judge."""
    return {"semantic": None, "reason": "", "error": error}


def _strict_judge_axis(scores: dict, axis: str) -> tuple[Optional[float], Optional[str]]:
    value = scores.get(axis)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        return None, f"{axis} score missing or non-numeric"
    numeric = float(value)
    if not 1.0 <= numeric <= 10.0:
        return None, f"{axis} score out of range: {value}"
    return value, None


def _strict_judge_reason(scores: dict) -> tuple[str, Optional[str]]:
    reason = scores.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return "", "reason missing or non-string"
    return reason, None


def _normalize_image_list(paths: Union[str, List[str], None]) -> List[str]:
    """Coerce str|list|None to a list of existing file paths."""
    if paths is None:
        return []
    if isinstance(paths, str):
        paths = [paths]
    return [p for p in paths if p and Path(p).exists()]


def llm_judge_score(llm_client,
                    pred_image_paths: Union[str, List[str]],
                    gt_image_paths: Union[str, List[str]],
                    condition_text: str = "",
                    enable_semantic: bool = False,
                    geometry_only: bool = False,
                    semantic_only: bool = False,
                    condition_image_paths: Union[str, List[str], None] = None,
                    require_condition_image: bool = False,
                    protocol_id: Optional[str] = None,
                    trace_out: Optional[dict] = None) -> dict:
    """Use a VLM to score the similarity between predicted and GT CAD models.

    Args:
        llm_client: A :class:`p3dbench.models.ModelClient`.
        pred_image_paths: One or more PNGs of the (aligned) predicted mesh, in the
            canonical VIEW_ANGLES order: top-front-right diagonal (down),
            top-back-left diagonal (down), bottom-back-right diagonal (up),
            bottom-front-left diagonal (up).
        gt_image_paths: One or more PNGs of the GT mesh, same order.
        condition_text: Original condition (text prompt, etc.).
        enable_semantic: Enable semantic similarity scoring. Ignored when
            ``geometry_only=True``.
        geometry_only: When True the judge scores ONLY geometry — aesthetics +
            semantic are dropped from both the prompt and the output dict (used
            for text-to-3d). Output dict in this mode: ``{geometry, reason, error}``.
        semantic_only: When True, use the frozen descriptive Text-to-3D prompt
            and return exactly ``{semantic, reason, error}``.
        condition_image_paths: Original model-visible image condition for
            Image-/Assembly-3D. It is sent before PRED/GT multiviews.
        require_condition_image: Fail before the model call when the formal
            visual protocol has no original model-visible image.

    Returns:
        dict with geometry score; plus aesthetics + optional semantic + avg unless
        ``geometry_only=True``. Never raises — always returns the dict schema.
    """
    pred_list = _normalize_image_list(pred_image_paths)
    gt_list = _normalize_image_list(gt_image_paths)
    condition_images = _normalize_image_list(condition_image_paths)
    default_result = semantic_judge_default_result if semantic_only else judge_default_result
    if not pred_list:
        return default_result(f"Pred image(s) missing: {pred_image_paths}")
    if not gt_list:
        return default_result(f"GT image(s) missing: {gt_image_paths}")

    if protocol_id == PAPER_PROTOCOL_ID:
        if len(pred_list) != PAPER_CANONICAL_VIEW_COUNT:
            return default_result(
                f"Paper protocol requires exactly {PAPER_CANONICAL_VIEW_COUNT} PRED views; "
                f"got {len(pred_list)}"
            )
        if len(gt_list) != PAPER_CANONICAL_VIEW_COUNT:
            return default_result(
                f"Paper protocol requires exactly {PAPER_CANONICAL_VIEW_COUNT} GT views; "
                f"got {len(gt_list)}"
            )
        if require_condition_image and len(condition_images) != 1:
            return default_result(
                "Paper visual judge requires exactly one original "
                f"model-visible condition image; got {len(condition_images)}"
            )

    n_pred, n_gt = len(pred_list), len(gt_list)
    view_order = (
        "(1: top-front-right diagonal looking down, "
        "2: top-back-left diagonal looking down, "
        "3: bottom-back-right diagonal looking up, "
        "4: bottom-front-left diagonal looking up — all at ±30° elevation)"
    )
    offset = len(condition_images)
    condition_note = (
        f"Image 1 is the original model-visible condition image.\n"
        if offset == 1 else
        (f"Images 1..{offset} are the original model-visible condition images.\n"
         if offset else "")
    )
    multiview_note = (
        f"{condition_note}"
        f"Images {offset + 1}..{offset + n_pred} are PRED rendered from {view_order}.\n"
        f"Images {offset + n_pred + 1}..{offset + n_pred + n_gt} are GT rendered from the same views."
        if n_pred > 1 or n_gt > 1 else
        f"{condition_note}Image {offset + 1} is PRED, Image {offset + 2} is GT."
    )

    if semantic_only:
        geometry_only = False
        enable_semantic = True

    # geometry_only takes precedence over enable_semantic — when True,
    # neither aesthetics nor semantic is asked for or returned. Used for
    # text-to-3d where the panel is a single geometry axis.
    if geometry_only:
        enable_semantic = False

    aesthetics_section = ""
    aesthetics_json_field = ""
    if not geometry_only:
        if enable_semantic:
            # Semantic-gated rubric: aes is capped at 3 when PRED is the wrong
            # kind of object (semantic < 4), and otherwise scales with PRED's
            # own detail richness without requiring feature alignment to GT.
            # Decide your `semantic` score first in `reason`, then apply the cap.
            aesthetics_section = """
### Aesthetics & Detail (1-10, gated by Semantic):
  Aes captures "is PRED a well-made object of the right kind?" — it
  combines a semantic-category gate (looking at GT only to identify
  what kind of object it is) with PRED's own detail richness and
  visual quality. Within the same category, PRED does NOT need to
  match GT's specific features/proportions/alignment to score high.

  HARD GATE: Decide your `semantic` score below first (think it through
  in `reason`). Then:
    - If `semantic` < 4 (PRED is the wrong kind of object — different
      category or only vaguely similar), aesthetics MUST be in [1, 3]:
        1 = bare primitive (cylinder/cube/ring/slab) or completely
            unrelated category — no shared visual cues with GT
        2 = wrong category, but PRED is itself a clean simple object
        3 = wrong category, but PRED has some sub-features of its own
    - If `semantic` ≥ 4 (PRED is at least loosely the same kind of
      object as GT), score aesthetics in [2, 10] by PRED's own detail
      richness and visual quality (do NOT require feature-level
      alignment to GT — a different bracket that's well-detailed
      still scores high):
        10 = Production-quality: many meaningful sub-features
             (fillets, chamfers, bosses, ribs, holes, varied
             thicknesses), polished surfaces, every feature looks
             intentional and engineered
        8-9 = Clean and richly detailed; one or two sub-features
              missing or slightly rough, but clearly an engineered
              part of the right kind
        6-7 = Right kind but coarse — few sub-features, plain blocky
              execution, or noticeably awkward proportions
        4-5 = Right kind but minimum-viable detail — recognizable
              mainly by silhouette; surfaces and edges feel
              unconsidered
        2-3 = Right kind but essentially bare — almost no sub-features
              beyond the basic shape, no engineering polish; the
              category guess is the only thing carrying the score

  A clean primitive (e.g. a polished cylinder) for an unrelated GT is
  ALWAYS gated to ≤3, no matter how visually pleasing the primitive
  looks on its own.
"""
        else:
            # Fallback when semantic isn't asked for (semantic gate impossible):
            # use the original PRED-absolute rubric so the axis still produces
            # something meaningful, just without the wrong-category cap.
            aesthetics_section = """
### Aesthetics & Detail (ABSOLUTE quality of PRED, 1-10):
  Score PRED's own quality as a CAD object. This is NOT a similarity-to-GT
  score — do NOT penalize PRED for being cleaner or more detailed than GT;
  GT only sets the expected level of detail for the task category.
  - Visual appeal: harmonious proportions, intentional symmetry, smooth curved
    surfaces where expected, clean silhouettes, no broken/artifact geometry.
  - Detail richness: meaningful sub-features and sub-parts (fillets, chamfers,
    bosses, ribs, holes, varied thicknesses) — not just blocky primitives.
  - Reasonableness: every feature looks like it belongs on a real engineered
    part — feature placement, sizes, and thicknesses are plausible; no
    arbitrary, decorative-only, or nonsensical geometry; sub-features are
    consistent with the object's apparent function.
  10 = Production-quality: visually polished, richly detailed, every feature
       sensibly placed and reasonably proportioned
  8-9 = Clean and pleasing; mostly detailed and reasonable; minor sub-features
        missing, slightly rough, or one questionable design choice
  6-7 = Recognizable form but coarse or plain; several sub-features missing,
        some unrefined regions, or noticeably awkward proportions
  4-5 = Rough primitive-level geometry; significant detail missing; design
        feels arbitrary or unconsidered
  2-3 = Crude blocks, broken surfaces, visible artifacts, or implausible
        / nonsensical geometry
  1 = Unrecognizable or fundamentally broken
"""
        aesthetics_json_field = ', "aesthetics": <1-10>'

    semantic_section = ""
    semantic_json_field = ""
    if enable_semantic:
        semantic_section = """
### Semantic (Semantic Similarity, 1-10):
  - Are both objects the same type/category? (e.g., both gears, both scissors, both brackets)
  - Do they serve the same functional purpose?
  - Would a human recognize them as the same kind of object?
  10 = Identical semantic category and function
  8-9 = Same category with minor functional variations
  6-7 = Related categories but different subtypes
  4-5 = Loosely related categories
  2-3 = Different categories with vague similarity
  1 = Completely different semantic categories
"""
        semantic_json_field = ', "semantic": <1-10>'

    condition_line = f"Original text description: {condition_text}\n" if condition_text else ""

    if semantic_only:
        prompt = f"""You are evaluating whether a text-to-CAD model produced the correct TYPE of object.

{multiview_note}
The GT images show what a ground-truth model of the described object looks like, but the PRED model was generated from TEXT ONLY — it never saw the GT images. Do NOT penalize PRED for geometric differences from GT. Only judge whether PRED is the right SEMANTIC category of object.

{condition_line}## Task

Look at the PRED images and the GT images. The GT shows what a correct model of the described object looks like. The PRED was generated from the text above, without seeing the GT.

Your ONLY task: does the PRED model belong to the same semantic category as what the text describes?

## Scoring rubric — Semantic Only (1-10)

### Semantic (Semantic Category Match, 1-10):
  - Is PRED the same type/category of object that the text describes?
  - Would a human, after reading the text, say "yes, the PRED is that kind of object"?
  - Ignore all geometric differences: wrong shape, wrong size, missing features — these are FORMAT/GEOMETRY issues, not semantic issues.
  - Focus ONLY on: does PRED look like it belongs to the category described in the text?

  10 = PRED is clearly the same category of object described by the text (e.g. bracket, enclosure, gear, mount)
  8-9 = PRED is the right category but one sub-feature is categorically wrong (e.g. a hole where a slot should be)
  6-7 = PRED is a related category (e.g. a mounting bracket vs a structural bracket)
  4-5 = PRED is a loosely related mechanical part (e.g. both are "mechanical parts" but different functions)
  2-3 = PRED is a different category with only vague resemblance
  1 = PRED is completely unrelated to the described object

Respond in EXACTLY this JSON format, nothing else:
{{"reason": "<justify the semantic score>", "semantic": <1-10>}}"""
    else:
        prompt = f"""You are a strict CAD model evaluator comparing a generated model (PRED) against a ground-truth model (GT).

{multiview_note}
The models have been approximately aligned, so ignore small pose offsets that persist across all views — focus on shape, features, proportions, and detail.

{condition_line}## Evaluation procedure

Step 1 — Cross-check every view pair by local view index (PRED view 1 vs GT view 1,
PRED view 2 vs GT view 2, PRED view 3 vs GT view 3, and PRED view 4 vs GT view 4
— each pair is the same viewpoint):
  - Views 1 & 2 look DOWN from above at opposite diagonals: use them to check
    top surfaces and top-facing features (bosses, through-holes from the top,
    upper fillets/chamfers).
  - Views 3 & 4 look UP from below at opposite diagonals: use them to check
    bottom surfaces, through-holes exiting the bottom, base geometry,
    underside pockets.
  - Across all 4 views:
    - Aspect ratios and proportions (e.g. length-to-width ratio, head-to-body ratio)
    - Absolute and relative sizes of each sub-feature
    - Relative positioning and alignment between parts
    - Presence or absence of features: holes, slots, fillets, chamfers, ribs, bosses
    - Shape of cross-sections (circular vs rectangular, sharp vs rounded edges)
    - Number of distinct parts / connected components

Step 2 — Score using the rubric below.

## Scoring rubric (1-10)

### Geometry (Geometric Shape Similarity, 1-10):
  - Aspect ratios and proportions matching
  - Feature positions and quantities
  - Overall contour similarity across all views
  - Size and scale of components
  10 = Geometrically identical across every view — all proportions, sizes, and features match exactly
  8-9 = Minor proportion differences (<10%) or one small feature slightly off
  6-7 = Noticeable proportion errors (10-25%) or 1-2 missing/extra features
  4-5 = Significant shape or proportion errors (>25%), or multiple missing features
  2-3 = Major structural differences; only vaguely similar geometry
  1 = Completely different geometric shape

{aesthetics_section}{semantic_section}
IMPORTANT: A score of 10 means VISUALLY INDISTINGUISHABLE across every view. If you can see ANY difference in proportions, features, or shape, geometry must be below 10. Be precise and critical.

Respond in EXACTLY this JSON format, nothing else:
{{"reason": "<per-view differences, then justify each score>", "geometry": <1-10>{aesthetics_json_field}{semantic_json_field}}}"""

    try:
        response_text = _generate_text(
            llm_client, prompt, images=condition_images + pred_list + gt_list, timeout=240,
            trace_kind=("judge_semantic_only" if semantic_only else "judge_visual"),
            trace_out=trace_out,
        )

        scores = extract_json_object(response_text)
        if not scores:
            return default_result(f"Could not parse LLM response: {response_text[:200]}")

        if semantic_only:
            semantic_score, axis_error = _strict_judge_axis(
                scores, "semantic"
            )
            reason, reason_error = _strict_judge_reason(scores)
            if axis_error or reason_error:
                return semantic_judge_default_result(
                    axis_error or reason_error
                )
            return {
                "semantic": semantic_score,
                "reason": reason,
                "error": None,
            }

        reason, reason_error = _strict_judge_reason(scores)
        geometry_score, geometry_error = _strict_judge_axis(
            scores, "geometry"
        )
        if reason_error or geometry_error:
            return judge_default_result(reason_error or geometry_error)

        if geometry_only:
            return {
                "geometry": geometry_score,
                "reason": reason,
                "error": None,
            }

        aesthetics_score, aesthetics_error = _strict_judge_axis(
            scores, "aesthetics"
        )
        if aesthetics_error:
            return judge_default_result(aesthetics_error)
        semantic_score = None
        if enable_semantic:
            semantic_score, semantic_error = _strict_judge_axis(
                scores, "semantic"
            )
            if semantic_error:
                return judge_default_result(semantic_error)
        return {
            "geometry": geometry_score,
            "semantic": semantic_score,
            "aesthetics": aesthetics_score,
            "reason": reason,
            "error": None,
        }
    except Exception as e:
        return default_result(f"LLM judge failed: {e}")


# ==========================================================================
# 2. QA  (QA-S / QA-P — EVAL-TIME ONLY, banks ship with the data)
#    Bank generation + verification are NOT ported (banks ship with data).
# ==========================================================================
SEMANTIC_QA_COUNT = 4
PARAM_QA_COUNT = 8
TOTAL_QA_COUNT = SEMANTIC_QA_COUNT + PARAM_QA_COUNT

# Warn (but do not truncate) when an input file is unusually large.
LARGE_INPUT_WARN_BYTES = 512_000

OPTION_E_TEXT = "None of the above — the predicted object does not match any of the provided options"

QA_ANSWERER_SYSTEM_PROMPT = (
    "You are a strict CAD QA answerer.\n\n"
    "REASONING APPROACH BY QUESTION TYPE:\n\n"
    "For SEMANTIC questions (shape, orientation, feature presence/location):\n"
    "  - Use the RENDER IMAGE as your primary evidence. Directly observe the "
    "shape, spatial relationships, and feature positions visible in the render.\n"
    "  - You may cross-reference the artifact code for confirmation, but if you "
    "do, you must trace the full construction logic — do not just read variable "
    "names or declarations.\n\n"
    "For PARAMETER questions (dimensions, counts, numeric values):\n"
    "  - Use the ARTIFACT CODE as your primary evidence, but you MUST trace "
    "the actual construction logic to determine the final geometry.\n"
    "  - CRITICAL WARNING — the 'variable declaration trap': source code often "
    "declares parameters like `cylinder_height = 0.5` or `depth = 0.3` at the "
    "top of the file, but the construction logic below may apply scale(), "
    "translate(), rotate(), or other transforms that change the final dimensions. "
    "Variables may also be overridden, ignored, or incorrectly wired into the "
    "construction calls. Do NOT treat variable declarations as facts about the "
    "final object. Instead, trace how each value flows through the construction "
    "logic to determine the actual dimensions of the finished CAD model.\n"
    "  - Use the render image to sanity-check proportions and relative sizes.\n\n"
    "For DERIVED questions (differences, ratios, angles, comparisons):\n"
    "  - Identify ALL individual measurements referenced in the question.\n"
    "  - Extract each measurement separately by tracing the construction logic.\n"
    "  - Then perform the required calculation (subtraction, division, angle "
    "formula, etc.).\n"
    "  - Show your intermediate values before giving the final answer.\n"
    "  - Do NOT guess the derived value from the render alone — compute it from "
    "the artifact code.\n\n"
    "For EXISTENCE TRAP questions (one option says 'does not exist' or "
    "describes a different feature):\n"
    "  - FIRST determine whether the described feature actually exists in the "
    "artifact by examining BOTH the code and the render.\n"
    "  - If the feature exists as described: ignore the negation option and "
    "compute the answer normally.\n"
    "  - If the feature does NOT exist or is described incorrectly (e.g., "
    "'triangular chamfer' but it is actually a circular arc): select the "
    "negation/correction option.\n"
    "  - Do NOT assume features exist just because the question asks about "
    "them — the question may be deliberately testing whether you can detect "
    "a geometric mismatch.\n\n"
    "OPTION E — 'None of the above':\n"
    "  Every question includes an option E: 'None of the above — the predicted "
    "object does not match any of the provided options.' Select E when:\n"
    "  - Your computed value does NOT match ANY of the A-D options (not even "
    "approximately). For example, if you compute a diameter of 0.56 but the "
    "options are 0.25, 0.39, 0.195, 0.57, pick the closest match; but if the "
    "options are 0.75, 0.36, 0.90, 1.10 and your value is 0.56, pick E.\n"
    "  - The question describes a feature type (e.g. 'cylindrical') that does "
    "NOT match what you observe in the render (e.g. you see a hexagonal or "
    "square cross-section, not a smooth cylinder). Trust what you SEE in the "
    "render over what the source code declares.\n"
    "  - The spatial relationship described in all A-D options is wrong for the "
    "predicted object (e.g. all options assume concentricity but the parts are "
    "clearly offset).\n"
    "  Do NOT select E just because you are uncertain. Only select E when you "
    "have positive evidence that none of A-D is correct for the PREDICTED "
    "object.\n\n"
    "Never guess or use any prior knowledge about what the ground-truth object "
    "might look like."
)

# Format slug -> human label for the predicted artifact shown to the answerer.
# (Ported from ARTIFACT_SPECS labels, keyed by the new P3D format slugs.)
ARTIFACT_LABELS = {
    "minimal-json": "Text2CAD prediction JSON",
    "json": "Text2CAD prediction JSON",
    "cadquery": "CadQuery Python source",
    "threejs": "Three.js source",
    "openscad": "OpenSCAD source",
}


def _read_text(path: Path) -> str:
    """Read a file in full. Warn (but never truncate) if unusually large."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > LARGE_INPUT_WARN_BYTES:
        logger.warning(
            "QA metric input %s is %d chars (>%d); passing full content to LLM.",
            path, len(text), LARGE_INPUT_WARN_BYTES,
        )
    return text


def _normalize_answer_letter(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D", "E"}:
        return text
    return ""


def mcq_accuracy(predictions: List[str], ground_truths: List[str]) -> dict:
    """Case-insensitive stripped letter match. (Ported from qa_metrics.py.)"""
    correct = sum(1 for p, g in zip(predictions, ground_truths)
                  if p.strip().upper() == g.strip().upper())
    total = len(predictions)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }


def _bank_questions(qa_bank: dict) -> List[dict]:
    """Return the flat question list from a bank (concatenating splits if needed)."""
    return qa_bank.get("questions") or (qa_bank.get("semantic", []) + qa_bank.get("param", []))


# HF qa.jsonl tags each question with a short format slug; the Text-to-3D run
# formats map onto it 1:1 (the task supports only these two).
_HF_QA_FORMAT = {"minimal-json": "json", "openscad": "openscad", "json": "json"}


def _question_split(q: dict) -> str:
    """Semantic vs param split — an explicit field, else the qid prefix."""
    return q.get("split") or ("param" if str(q.get("qid", "")).startswith("param") else "semantic")


def select_bank_questions(qa_bank: dict, text_mode: Optional[str] = None,
                          fmt: Optional[str] = None) -> List[dict]:
    """Flat, normalized question list for one ``(text_mode, format)`` variant.

    The Hub bank (``data/text_to_3d/qa.jsonl``) packs every text_mode×format
    variant into one ``questions`` list, each tagged with ``text_mode``/
    ``format``; the demo/research bank ships a single variant already carrying
    ``split``/``category``. This returns the matching subset (or the whole bank
    when it is not variant-tagged), with ``split`` filled in from the qid where
    absent so the scorer can separate QA-S from QA-P.
    """
    questions = _bank_questions(qa_bank)
    tagged = [q for q in questions if "text_mode" in q or "format" in q]
    if tagged:
        # Variant-tagged (Hub) bank: keep only the active run's variant. A
        # missing selector or no match yields [] so the caller cleanly skips
        # rather than scoring against a mix of variants.
        hf_fmt = _HF_QA_FORMAT.get(fmt, fmt) if fmt else None
        questions = [q for q in tagged
                     if q.get("text_mode") == text_mode and q.get("format") == hf_fmt]
    out = []
    for q in questions:
        q = dict(q)
        q["split"] = _question_split(q)
        out.append(q)
    return out


def validate_paper_qa_bank(
    qa_bank: dict,
    *,
    text_mode: str,
    fmt: str,
    expected_uid: Optional[str] = None,
) -> List[dict]:
    """Validate and select one frozen v9 QA-bank variant without regenerating it."""
    if text_mode not in {"parametric", "descriptive"}:
        raise ValueError(f"Invalid paper QA text mode: {text_mode!r}")
    hf_fmt = _HF_QA_FORMAT.get(fmt)
    if hf_fmt is None:
        raise ValueError(f"Unsupported paper QA prediction format: {fmt!r}")
    version = qa_bank.get("qa_bank_version")
    if version != PAPER_QA_BANK_VERSION:
        raise ValueError(
            f"Paper protocol requires QA bank v{PAPER_QA_BANK_VERSION}; got {version!r}"
        )
    bank_uid = str(qa_bank.get("uid") or qa_bank.get("case_id") or "").strip()
    if not bank_uid:
        raise ValueError("Paper QA bank must carry its source UID")
    validate_materialized_qa_bank_contract(
        qa_bank,
        expected_uid=expected_uid or bank_uid,
    )

    bank_mode = qa_bank.get("text_mode")
    if bank_mode is not None and bank_mode != text_mode:
        raise ValueError(
            f"QA bank mode mismatch: expected {text_mode}, got {bank_mode}"
        )
    bank_format = qa_bank.get("format")
    if bank_format is not None and bank_format != hf_fmt:
        raise ValueError(
            f"QA bank format mismatch: expected {hf_fmt}, got {bank_format}"
        )
    if (
        qa_bank.get("semantic_source_text_level") is not None
        and qa_bank.get("semantic_source_text_level") != "detailed"
    ):
        raise ValueError("QA bank semantic source must be detailed")
    expected_param_source = (
        "parametric_detail" if text_mode == "parametric" else ""
    )
    if (
        qa_bank.get("param_source_text_level") is not None
        and str(qa_bank.get("param_source_text_level") or "")
        != expected_param_source
    ):
        raise ValueError(
            f"QA bank param source must be {expected_param_source!r}"
        )

    questions = [
        dict(question)
        for question in _bank_questions(qa_bank)
        if question.get("text_mode") == text_mode
        and question.get("format") == hf_fmt
    ]
    expected_semantic = SEMANTIC_QA_COUNT
    expected_param = 0 if text_mode == "descriptive" else PARAM_QA_COUNT
    expected_total = expected_semantic + expected_param
    if len(questions) != expected_total:
        raise ValueError(
            f"{text_mode}/{fmt} QA bank must contain {expected_total} questions; "
            f"got {len(questions)}"
        )

    expected_qids = (
        [f"semantic_{index}" for index in range(1, expected_semantic + 1)]
        + [f"param_{index}" for index in range(1, expected_param + 1)]
    )
    actual_qids = [str(question.get("qid") or "").strip() for question in questions]
    if actual_qids != expected_qids:
        raise ValueError(
            f"QA qid order must be {expected_qids}; got {actual_qids}"
        )

    semantic_n = param_n = 0
    for index, question in enumerate(questions, start=1):
        qid = str(question.get("qid", "")).strip()
        if question.get("text_mode") != text_mode:
            raise ValueError(
                f"QA question {qid} mode must be {text_mode!r}"
            )
        if question.get("format") != hf_fmt:
            raise ValueError(
                f"QA question {qid} format must be {hf_fmt!r}"
            )
        split = question.get("split")
        if split == "semantic":
            semantic_n += 1
            expected_source = "detailed"
        elif split == "param":
            param_n += 1
            expected_source = "parametric_detail"
        else:
            raise ValueError(f"QA question {qid} has invalid split {split!r}")
        if question.get("source_text_level") != expected_source:
            raise ValueError(
                f"QA question {qid} source_text_level must be "
                f"{expected_source!r}"
            )
        options = question.get("options")
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(f"QA question {qid} must have exactly four A-D options")
        normalized_options = [str(option).strip() for option in options]
        if any(not option for option in normalized_options) or len(set(normalized_options)) != 4:
            raise ValueError(f"QA question {qid} must have four non-empty distinct options")
        if str(question.get("answer", "")).strip().upper() not in {"A", "B", "C", "D"}:
            raise ValueError(f"QA question {qid} ground truth must be A-D")

    if semantic_n != expected_semantic or param_n != expected_param:
        raise ValueError(
            f"QA split mismatch: semantic={semantic_n}/{expected_semantic}, "
            f"param={param_n}/{expected_param}"
        )
    return questions


def _format_question_block(questions: List[dict]) -> str:
    blocks = []
    for question in questions:
        options = question["options"]
        category = question.get("category", "")
        header = f"[{question['qid']}] ({question['split']} / {category}) {question['question']}"
        blocks.append(
            f"{header}\n"
            f"A. {options[0]}\n"
            f"B. {options[1]}\n"
            f"C. {options[2]}\n"
            f"D. {options[3]}\n"
            f"E. {OPTION_E_TEXT}"
        )
    return "\n\n".join(blocks)


def _extract_pred_mesh_summary(stl_path: Optional[Union[str, Path]]) -> str:
    """Measured bounding-box block for the predicted mesh, or "" on any failure.

    Loads the STL via :func:`p3dbench.compile.step_mesh.load_mesh` (trimesh-backed).
    Optional: degrades to "" if trimesh / the mesh isn't available, since the bbox
    is only a supplementary sanity-check for the answerer.
    """
    if not stl_path:
        return ""
    stl_path = Path(stl_path)
    if not stl_path.exists():
        return ""
    try:
        from ..compile.step_mesh import load_mesh
        mesh = load_mesh(str(stl_path))
        if mesh is None or mesh.is_empty:
            return ""
        extents = mesh.extents
        bounds = mesh.bounds
        lines = [
            "PREDICTION MESH BOUNDING BOX (extracted from the exported STL for reference — note that mesh tessellation and export may introduce small rounding errors vs. the source code):",
            f"  Bounding box extents: X={extents[0]:.4f}, Y={extents[1]:.4f}, Z={extents[2]:.4f}",
            f"  Bounding box min: [{bounds[0][0]:.4f}, {bounds[0][1]:.4f}, {bounds[0][2]:.4f}]",
            f"  Bounding box max: [{bounds[1][0]:.4f}, {bounds[1][1]:.4f}, {bounds[1][2]:.4f}]",
        ]
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("Could not extract mesh summary from %s: %s", stl_path, exc)
        return ""


def answer_qa_bank(
    llm_client,
    qa_bank: dict,
    pred_render_paths: Union[str, List[str]],
    fmt_slug: str,
    artifact_text: str,
    artifact_label: Optional[str] = None,
    artifact_name: str = "prediction",
    pred_stl_path: Optional[Union[str, Path]] = None,
    protocol_id: Optional[str] = None,
    trace_out: Optional[dict] = None,
) -> dict:
    """Answer a fixed QA bank using prediction renders and artifact text.

    Args:
        llm_client: a :class:`p3dbench.models.ModelClient`.
        qa_bank: the shipped bank dict (``semantic`` + ``param`` splits).
        pred_render_paths: canonical prediction render image paths.
        fmt_slug: prediction format slug (e.g. ``"minimal-json"``).
        artifact_text: the predicted source artifact, in full.
        artifact_label: human label for the artifact; defaults from ``fmt_slug``.
        artifact_name: basename shown to the answerer (diagnostic only).
        pred_stl_path: optional predicted STL for the measured-bbox sanity block.

    Returns the raw answer payload ``{"answers": [...]}``. Raises ``ValueError``
    on an unparseable answer payload (single-shot: no retry).
    """
    render_values = (
        [pred_render_paths] if isinstance(pred_render_paths, str)
        else list(pred_render_paths or [])
    )
    pred_renders = [Path(path) for path in render_values if path and Path(path).exists()]
    if protocol_id == PAPER_PROTOCOL_ID and len(pred_renders) != PAPER_CANONICAL_VIEW_COUNT:
        raise ValueError(
            f"Paper QA requires exactly {PAPER_CANONICAL_VIEW_COUNT} canonical "
            f"prediction renders; got {len(pred_renders)}"
        )
    if not pred_renders:
        raise FileNotFoundError("No prediction renders found")

    questions = _bank_questions(qa_bank)
    if not questions:
        raise ValueError("QA bank has no questions")
    paper_contract = qa_bank.get("_paper_contract")
    if protocol_id == PAPER_PROTOCOL_ID:
        if not isinstance(paper_contract, dict):
            raise ValueError("Paper QA requires a validated frozen-bank contract")
        if paper_contract.get("format") != fmt_slug:
            raise ValueError("Paper QA bank format does not match prediction format")
        if not paper_contract.get("text_mode"):
            raise ValueError("Paper QA bank contract is missing text mode")
        bank_sha = str(paper_contract.get("bank_sha256") or "")
        if len(bank_sha) != 64:
            raise ValueError("Paper QA bank contract is missing SHA-256 provenance")

    if artifact_label is None:
        artifact_label = ARTIFACT_LABELS.get(fmt_slug.lower(), fmt_slug)

    mesh_summary = _extract_pred_mesh_summary(pred_stl_path)
    if protocol_id == PAPER_PROTOCOL_ID and not mesh_summary:
        raise ValueError(
            "Paper QA requires measured prediction mesh bounding-box metadata"
        )
    mesh_block = f"\n\n{mesh_summary}\n" if mesh_summary else ""

    multiview = len(pred_renders) == 4
    if multiview:
        view_desc = (
            "The attached images are 4 rendered views of the predicted CAD object "
            "from a tetrahedral coverage:\n"
            "  Image 1: top-front-right diagonal looking down\n"
            "  Image 2: top-back-left diagonal looking down\n"
            "  Image 3: bottom-back-right diagonal looking up\n"
            "  Image 4: bottom-front-left diagonal looking up\n"
        )
        image_note = "the attached predicted render images (4 views)"
    else:
        view_desc = ""
        image_note = "the attached predicted render image"

    prompt = f"""Answer the following multiple-choice CAD evaluation questions.

You have access to ONLY these sources — use nothing else:
1. {image_note} of the predicted CAD object
2. The prediction artifact text below (the source code or JSON that defines the predicted object)
3. The measured mesh dimensions below (programmatically extracted from the actual built model){mesh_block}
{view_desc}
Prediction format: {fmt_slug}
Prediction artifact path: {artifact_name}
Prediction artifact type: {artifact_label}

Prediction artifact:
```text
{artifact_text}
```

Questions:
{_format_question_block(questions)}

IMPORTANT — follow this reasoning approach for each question:

For SEMANTIC questions (marked as "semantic" split — about shape, orientation, features):
  1. FIRST examine the renders: describe the overall shape, visible features, and spatial relationships.
  2. Use what you observe in the renders as your primary basis for the answer.
  3. You may reference the artifact code for confirmation, but if you do, trace the full construction logic — do not just read variable names.

For PARAMETER questions (marked as "param" split — about dimensions, counts, operations):
  1. Read the artifact code carefully. Do NOT just look at variable declarations at the top of the file.
  2. TRACE the actual construction logic: follow how declared values are used in the construction calls. Check for scale(), translate(), rotate(), or other transforms that change final dimensions. A variable named `height = 0.5` does NOT mean the final object is 0.5 units tall if the code later applies `scale(2)` or uses a different variable.
  3. You may use the mesh bounding box (if provided) as a supplementary reference to sanity-check your code analysis, but note that mesh export can introduce small rounding errors.
  4. Use the renders to sanity-check proportions.

Every question has an option E ("None of the above"). Select E when you have
positive evidence that none of A-D correctly describes the predicted object.
Do NOT select E merely because you are unsure.

Return exactly this JSON object and nothing else:
{{
  "answers": [
    {{"qid": "semantic_1", "reasoning": "Brief 1-2 sentence explanation of your reasoning", "answer": "A"}}
  ]
}}
Valid answer values: A, B, C, D, or E."""

    raw_text = _generate_text(
        llm_client, prompt, images=[str(path) for path in pred_renders],
        temperature=0.1, max_tokens=32768,
        system=QA_ANSWERER_SYSTEM_PROMPT, timeout=180,
        trace_kind="qa_answer",
        trace_out=trace_out,
    )
    if trace_out is not None:
        source_mesh = (
            file_fingerprint(str(pred_stl_path))
            if pred_stl_path and Path(pred_stl_path).is_file()
            else None
        )
        trace_out["evidence"] = {
            "prediction_source": {
                "name": artifact_name,
                "sha256": sha256_text(artifact_text),
                "bytes": len(artifact_text.encode("utf-8")),
            },
            "prediction_bbox": {
                "present": bool(mesh_summary),
                "sha256": (
                    sha256_text(mesh_summary) if mesh_summary else None
                ),
                "source_mesh": source_mesh,
            },
            "prediction_render_count": len(pred_renders),
            "qa_bank": dict(paper_contract) if paper_contract else None,
        }
    payload = extract_json_object(raw_text)
    if not payload or not isinstance(payload.get("answers"), list):
        logger.error(
            "Failed to parse QA answer payload. Raw response length=%d, first 1500 chars:\n%s",
            len(raw_text), raw_text[:1500],
        )
        raise ValueError("Failed to parse QA answer payload")
    return payload


def score_qa_results(qa_bank: dict, answer_payload: dict) -> Tuple[List[dict], dict]:
    """Score the answer payload against the fixed QA bank.

    Returns ``(per_question_rows, metrics)`` where ``metrics`` carries
    ``semantic_accuracy`` (QA-S), ``param_accuracy`` (QA-P), ``overall_accuracy``
    (all rounded to 4dp), plus diagnostic counts. ``E`` is never a GT answer, so
    choosing E always scores wrong (it is a diagnostic, counted separately).
    """
    questions = _bank_questions(qa_bank)
    answer_map: Dict[str, dict] = {}
    for item in answer_payload.get("answers", []):
        if not isinstance(item, dict):
            continue
        qid = str(item.get("qid", "")).strip()
        if not qid:
            continue
        answer_map[qid] = {
            "answer": _normalize_answer_letter(item.get("answer")),
            "reasoning": str(item.get("reasoning", "")).strip(),
        }

    qa_results: List[dict] = []
    semantic_preds: List[str] = []
    semantic_gts: List[str] = []
    param_preds: List[str] = []
    param_gts: List[str] = []
    overall_preds: List[str] = []
    overall_gts: List[str] = []
    category_buckets: Dict[str, List[Tuple[str, str]]] = {}

    skipped_count = 0
    none_of_above_count = 0
    for question in questions:
        entry = answer_map.get(question["qid"], {})
        predicted = entry.get("answer", "")
        reasoning = entry.get("reasoning", "")
        ground_truth = question["answer"]
        is_correct = predicted == ground_truth
        chose_none = predicted == "E"
        category = question.get("category", "")

        if not predicted:
            skipped_count += 1
        if chose_none:
            none_of_above_count += 1

        qa_results.append({
            "qid": question["qid"],
            "split": question["split"],
            "category": category,
            "question": question["question"],
            "predicted": predicted,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "chose_none_of_above": chose_none,
            "reasoning": reasoning,
        })

        overall_preds.append(predicted)
        overall_gts.append(ground_truth)
        if question["split"] == "semantic":
            semantic_preds.append(predicted)
            semantic_gts.append(ground_truth)
        else:
            param_preds.append(predicted)
            param_gts.append(ground_truth)

        if category:
            bucket_key = f"{question['split']}.{category}"
            category_buckets.setdefault(bucket_key, []).append((predicted, ground_truth))

    if skipped_count > 0:
        logger.warning("QA answerer skipped %d/%d questions", skipped_count, len(questions))

    semantic_stats = mcq_accuracy(semantic_preds, semantic_gts)
    param_stats = mcq_accuracy(param_preds, param_gts)
    overall_stats = mcq_accuracy(overall_preds, overall_gts)

    per_category_accuracy: Dict[str, dict] = {}
    for key, pairs in sorted(category_buckets.items()):
        preds = [p for p, _ in pairs]
        gts = [g for _, g in pairs]
        stats = mcq_accuracy(preds, gts)
        per_category_accuracy[key] = {
            "accuracy": round(stats["accuracy"], 4),
            "total": stats["total"],
            "correct": stats["correct"],
        }

    metrics = {
        "semantic_accuracy": round(semantic_stats["accuracy"], 4),
        "param_accuracy": round(param_stats["accuracy"], 4),
        "overall_accuracy": round(overall_stats["accuracy"], 4),
        "total_questions": overall_stats["total"],
        "total_correct": overall_stats["correct"],
        "semantic_total": semantic_stats["total"],
        "semantic_correct": semantic_stats["correct"],
        "param_total": param_stats["total"],
        "param_correct": param_stats["correct"],
        "none_of_above_count": none_of_above_count,
        "per_category_accuracy": per_category_accuracy,
    }
    return qa_results, metrics


# ==========================================================================
# 3. Rendering helpers (strict-pairing constants + multiview dispatch)
# ==========================================================================
# 4 tetrahedral judge views (elevation_deg, azimuth_deg): 2 top diagonals looking
# down + 2 bottom diagonals looking up at ±30°. The judge prompt assumes PRED
# view i is the SAME viewpoint as GT view i, so both sides must have exactly 4.
VIEW_ANGLES = [
    (30, 45),    # 0: top-front-right diagonal (above, looking down)
    (30, 225),   # 1: top-back-left diagonal (above, looking down)
    (-30, 135),  # 2: bottom-back-right diagonal (below, looking up)
    (-30, 315),  # 3: bottom-front-left diagonal (below, looking up)
]
JUDGE_VIEW_INDICES = [0, 1, 2, 3]
N_JUDGE_VIEWS = len(JUDGE_VIEW_INDICES)


def _render_pred_multiview(mesh_or_step_path: str, output_dir: Path,
                           n_views: int = N_JUDGE_VIEWS,
                           protocol_id: Optional[str] = None,
                           allow_paper_single_view: bool = False) -> List[str]:
    """Render ``n_views`` of the prediction, preferring occ/pyrender then blender.

    Returns the list of view PNG paths, or ``[]`` on any failure (so the judge
    cleanly skips when no render backend is installed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Import the render backends lazily so this module imports without them.
    try:
        from ..render import occ, blender
    except Exception as exc:
        logger.debug("render backends unavailable: %s", exc)
        return []

    if protocol_id == PAPER_PROTOCOL_ID:
        expected_views = 1 if allow_paper_single_view else PAPER_CANONICAL_VIEW_COUNT
        if n_views != expected_views:
            logger.warning(
                "paper renderer requires exactly %d views; got %d",
                expected_views,
                n_views,
            )
            return []
        try:
            views = blender.render_multiview(
                mesh_or_step_path,
                str(output_dir),
                n_views=n_views,
                resolution=PAPER_RENDER_RESOLUTION,
                samples=PAPER_RENDER_SAMPLES,
                seed=PAPER_RENDER_SEED,
            )
        except Exception as exc:
            logger.debug("formal Blender render failed: %s", exc)
            return []
        return list(views) if len(views or []) == expected_views else []

    for backend in (occ, blender):
        renderer = getattr(backend, "render_multiview", None)
        if renderer is None:
            continue
        try:
            views = renderer(mesh_or_step_path, str(output_dir), n_views=n_views)
            if views:
                return list(views)
        except Exception as exc:
            logger.debug("%s.render_multiview failed: %s",
                         getattr(backend, "__name__", backend), exc)
    return []


# ==========================================================================
# 4. Bucket
# ==========================================================================
_VISUAL_JUDGE_KEYS = ("judge_semantic", "judge_geometry", "judge_aesthetics")


def _resolve_text_mode(ctx: ScoreContext) -> str:
    """Best-effort text_mode lookup (not carried on ScoreContext directly).

    text_mode lives on the compiled row / CLI flag, not on ctx, so check the
    cross-bucket cache then the case metadata; default ``"parametric"``.
    """
    mode = ctx.shared.get("text_mode")
    if not mode:
        meta = getattr(getattr(ctx.case, "case", None), "metadata", None) or {}
        mode = meta.get("text_mode")
    return mode or "parametric"


def _aligned_pred_source(ctx: ScoreContext) -> Optional[str]:
    """Path/mesh for the predicted geometry to render.

    Prefer the geometry-bucket aligned mesh (paper: the judge sees the ALIGNED
    pred so pose offsets don't dominate). The aligned mesh is a live trimesh in
    ``ctx.shared['geometry']`` — write it out so a render backend can load it.
    Fall back to the compiled STL.
    """
    geom = ctx.shared.get("geometry") or {}
    aligned = geom.get("aligned_pred")
    if aligned is not None:
        try:
            ctx.work_dir.mkdir(parents=True, exist_ok=True)
            out = ctx.work_dir / "model_aligned.stl"
            aligned.export(str(out))
            return str(out)
        except Exception as exc:
            logger.debug("could not export aligned pred mesh: %s", exc)
    stl = ctx.compiled.get("stl")
    return str(stl) if stl else None


class _JudgeBucket(MetricBucket):
    bucket = "judge"
    requires: set[str] = {"render", "judge_model"}

    # -- public entry ---------------------------------------------------
    def score(self, ctx: ScoreContext) -> dict:
        if ctx.task == "text-to-3d":
            return self._score_qa(ctx)
        if ctx.task in ("image-to-3d", "assembly-3d"):
            return self._score_visual(ctx)
        raise ValueError(f"Judge bucket does not support task '{ctx.task}'")

    # -- Text-to-3D : QA -------------------------------------------------
    def _score_qa(self, ctx: ScoreContext) -> dict:
        text_mode = _resolve_text_mode(ctx)
        # Descriptive mode reports {qa_semantic, judge_semantic}; parametric
        # reports {qa_semantic, qa_param}. (metrics.base.bucket_membership picks
        # the consumed subset; we populate everything we can compute.)
        out: dict = {"qa_semantic": None, "qa_param": None}
        if text_mode == "descriptive":
            out["judge_semantic"] = None

        qa_path = ctx.case.qa_bank
        if qa_path is None or ctx.judge_client is None:
            return out  # clean skip — no bank or no judge client

        try:
            raw_bank = json.loads(Path(qa_path).read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("could not load QA bank %s: %s", qa_path, exc)
            return out

        # The Hub bank packs every text_mode×format variant; pick the active run's
        # subset (the demo/research single-variant bank passes through unchanged).
        try:
            if ctx.protocol_id == PAPER_PROTOCOL_ID:
                expected_uid = str(
                    (getattr(getattr(ctx.case, "case", None), "metadata", None) or {})
                    .get("source_id") or ""
                ) or None
                questions = validate_paper_qa_bank(
                    raw_bank,
                    text_mode=text_mode,
                    fmt=ctx.fmt,
                    expected_uid=expected_uid,
                )
            else:
                questions = select_bank_questions(
                    raw_bank, text_mode=text_mode, fmt=ctx.fmt
                )
                if not questions:
                    raise ValueError(f"no {text_mode}/{ctx.fmt} variant")
        except ValueError as exc:
            logger.warning("QA bank %s rejected: %s", qa_path, exc)
            out["_judge_error"] = f"qa_bank_invalid: {exc}"
            return out
        bank_fingerprint = file_fingerprint(str(qa_path))
        qa_bank = {
            "questions": questions,
            "_paper_contract": {
                "qa_bank_version": PAPER_QA_BANK_VERSION,
                "text_mode": text_mode,
                "format": ctx.fmt,
                "bank_sha256": bank_fingerprint["sha256"],
            },
        }

        pred_stl = ctx.compiled.get("stl")
        if not pred_stl:
            return out  # invalid prediction: nothing to render / answer about

        # Paper protocol: source + bbox + exactly four aligned canonical renders.
        render_dir = ctx.work_dir / "qa_render"
        pred_source = _aligned_pred_source(ctx) or str(pred_stl)
        view_count = (
            PAPER_CANONICAL_VIEW_COUNT
            if ctx.protocol_id == PAPER_PROTOCOL_ID else 1
        )
        views = _render_pred_multiview(
            pred_source,
            render_dir,
            n_views=view_count,
            protocol_id=ctx.protocol_id,
        )
        if len(views) != view_count:
            out["_judge_error"] = (
                f"qa_render_gap: expected {view_count} prediction views, got {len(views)}"
            )
            return out

        artifact_text = ctx.shared.get("stage1_code") or ""

        try:
            trace = {} if ctx.protocol_id == PAPER_PROTOCOL_ID else None
            payload = answer_qa_bank(
                ctx.judge_client, qa_bank, views,
                fmt_slug=ctx.fmt, artifact_text=artifact_text,
                artifact_name=f"{ctx.case.id}.{ctx.fmt}",
                pred_stl_path=pred_stl,
                protocol_id=ctx.protocol_id,
                trace_out=trace,
            )
            if trace is not None and trace:
                append_call_trace(ctx.shared, trace)
            qa_rows, qa_metrics = score_qa_results(qa_bank, payload)
        except Exception as exc:
            logger.warning("QA answering/scoring failed for %s: %s", ctx.case.id, exc)
            out["_judge_error"] = f"qa_answer_gap: {type(exc).__name__}: {exc}"
            return out

        # Keep the scientific response and question-level scoring evidence.
        # Aggregation consumes only the canonical numeric keys below.
        out["qa_answer_payload"] = payload
        out["qa_results"] = qa_rows
        out["qa_scoring"] = qa_metrics
        out["qa_semantic"] = qa_metrics["semantic_accuracy"]
        if text_mode == "descriptive":
            # Descriptive: param fidelity is not scored; add a single semantic
            # judge axis (visual J-Sem) instead.
            result = self._descriptive_judge_semantic(ctx, pred_stl)
            out["judge_semantic_result"] = result
            if result.get("error"):
                out["_judge_error"] = result["error"]
            else:
                out["judge_semantic"] = result.get("semantic")
        else:
            out["qa_param"] = qa_metrics["param_accuracy"]
        return out

    def _descriptive_judge_semantic(self, ctx: ScoreContext, pred_stl) -> dict:
        """Single semantic axis for descriptive Text-to-3D (strict 4v pairing)."""
        gt_renders = [str(p) for p in (ctx.case.gt_renders or []) if p]
        if len(gt_renders) != N_JUDGE_VIEWS:
            return semantic_judge_default_result(
                f"expected {N_JUDGE_VIEWS} GT views, got {len(gt_renders)}"
            )
        pred_src = _aligned_pred_source(ctx) or str(pred_stl)
        pred_views = _render_pred_multiview(
            pred_src,
            ctx.work_dir / "multiview",
            protocol_id=ctx.protocol_id,
        )
        if len(pred_views) != N_JUDGE_VIEWS:
            return semantic_judge_default_result(
                f"expected {N_JUDGE_VIEWS} PRED views, got {len(pred_views)}"
            )
        condition_text = resolve_text_condition(ctx.case.case, "descriptive")
        trace = {} if ctx.protocol_id == PAPER_PROTOCOL_ID else None
        result = llm_judge_score(
            ctx.judge_client, pred_views, gt_renders,
            condition_text=condition_text,
            enable_semantic=True, geometry_only=False, semantic_only=True,
            protocol_id=ctx.protocol_id,
            trace_out=trace,
        )
        if trace is not None and trace:
            append_call_trace(ctx.shared, trace)
        return result

    # -- Image-to-3D / Assembly-3D : visual judge -----------------------
    def _score_visual(self, ctx: ScoreContext) -> dict:
        out = {k: None for k in _VISUAL_JUDGE_KEYS}
        out["judge_visual_result"] = None

        if ctx.judge_client is None:
            return out  # clean skip

        # STRICT pairing: GT must have exactly N views.
        gt_renders = [str(p) for p in (ctx.case.gt_renders or []) if p and Path(p).exists()]
        if len(gt_renders) != N_JUDGE_VIEWS:
            return out

        pred_src = _aligned_pred_source(ctx)
        if not pred_src:
            return out  # invalid prediction

        pred_views = _render_pred_multiview(
            pred_src,
            ctx.work_dir / "multiview",
            protocol_id=ctx.protocol_id,
        )
        # STRICT pairing: PRED must also have exactly N views (no mis-pairing).
        if len(pred_views) != N_JUDGE_VIEWS:
            return out

        condition = getattr(getattr(ctx.case, "case", None), "input", None)
        condition_text = getattr(condition, "text", "") or ""
        condition_images = [
            str(path) for path in (getattr(ctx.case, "image_paths", None) or [])
            if path and Path(path).exists()
        ][:1]

        trace = {} if ctx.protocol_id == PAPER_PROTOCOL_ID else None
        result = llm_judge_score(
            ctx.judge_client, pred_views, gt_renders,
            condition_text=condition_text,
            enable_semantic=True, geometry_only=False,
            condition_image_paths=condition_images,
            require_condition_image=True,
            protocol_id=ctx.protocol_id,
            trace_out=trace,
        )
        out["judge_visual_result"] = result
        if trace is not None and trace:
            append_call_trace(ctx.shared, trace)
        if result.get("error"):
            # The judge could not run (API/parse failure) — an evaluation gap,
            # not a model failure. Leave required metrics missing so summarize
            # blocks promotion instead of averaging a partial subset.
            logger.warning("judge skipped for %s: %s", ctx.case.id, result["error"])
            out["_judge_error"] = result["error"]
            return out
        out["judge_geometry"] = result.get("geometry")
        out["judge_semantic"] = result.get("semantic")
        out["judge_aesthetics"] = result.get("aesthetics")
        return out


BUCKET = _JudgeBucket()
