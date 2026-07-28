"""The four staged functions: infer -> compile -> score -> summarize.

Each stage reads and writes a plain JSONL/JSON artifact and carries no hidden
state, so the same predictions can be re-scored under a different metric without
re-running inference. Records are self-contained (they embed the case) so a
downstream stage never has to re-read the manifest.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from .config import DEFAULT_CONFIG_DIR, load_judge_config
from .data.loader import ResolvedCase, data_root, load_cases, manifest_path
from .data.schema import Case
from .metrics.base import (
    SCORE_BUCKETS,
    ScoreContext,
    bucket_score_for_case,
    bucket_membership,
    iou_applicability,
    missing_required_metrics,
    normalize_value,
)
from .models import get_client
from .registry import (
    get_metric_bucket,
    resolve_format,
    resolve_metric_buckets,
    resolve_task,
)
from .utils import read_jsonl, write_json, write_jsonl
from .protocol import (
    PAPER_PROTOCOL_ID,
    file_fingerprint,
    validate_paper_materialized_manifest,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Stage 1: infer
# --------------------------------------------------------------------------
# Tasks that get the compile-check-retry error-feedback loop (cadbenchmark port).
# Text-to-3D stays single-shot by design.
REFINE_TASKS = ("image-to-3d", "assembly-3d")
DEFAULT_REFINE_ATTEMPTS = 3
FAILURE_GENERATION_INVALID = "generation_invalid"
FAILURE_INFERENCE_GAP = "inference_gap"
FAILURE_EVALUATOR_GAP = "evaluator_gap"
GAP_FAILURE_CLASSES = {FAILURE_INFERENCE_GAP, FAILURE_EVALUATOR_GAP}
ALL_FAILURE_CLASSES = {FAILURE_GENERATION_INVALID, *GAP_FAILURE_CLASSES}


def _duplicate_ids(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _formal_expected_case_rows(task: str, split: str) -> tuple[list[dict], dict]:
    """Load and validate the canonical materialized rows for one paper task."""
    if split != "full":
        raise ValueError("Paper protocol requires split='full'")
    path = manifest_path(task, split)
    if not path.is_file():
        raise ValueError(f"Paper protocol manifest is missing: {path}")
    manifest_rows = list(read_jsonl(path))
    contract = validate_paper_materialized_manifest(
        task,
        manifest_rows,
        split=split,
        root=data_root(split),
    )
    contract["manifest"] = file_fingerprint(str(path))
    canonical_rows = [Case.from_dict(row).to_dict() for row in manifest_rows]
    return canonical_rows, contract


def _formal_expected_case_ids(task: str, split: str) -> tuple[list[str], dict]:
    """Load the task-declared full case universe in its canonical order."""
    rows, contract = _formal_expected_case_rows(task, split)
    return [str(row["id"]) for row in rows], contract


def _validate_formal_compiled_rows(
    rows: list[dict],
    buckets: list[str],
) -> dict:
    """Fail closed before any evaluator-model calls on a partial paper run."""
    if not rows:
        raise ValueError("Paper protocol cannot score an empty compiled artifact")
    for field in ("task", "format", "model", "split", "text_mode"):
        values = {str(row.get(field)) for row in rows}
        if len(values) != 1:
            raise ValueError(
                f"Paper protocol forbids mixed {field} values: {sorted(values)}"
            )
    task = str(rows[0]["task"])
    text_mode = str(rows[0].get("text_mode"))
    if task == "text-to-3d" and text_mode not in {
        "parametric", "descriptive"
    }:
        raise ValueError(
            f"Paper protocol has invalid Text-to-3D mode: {text_mode!r}"
        )
    expected_panel = set(resolve_metric_buckets("all", task))
    if set(buckets) != expected_panel:
        raise ValueError(
            "Paper protocol requires the complete metric panel "
            f"{sorted(expected_panel)}; got {sorted(set(buckets))}"
        )
    expected_cases, contract = _formal_expected_case_rows(
        task, str(rows[0]["split"])
    )
    expected_ids = [str(case["id"]) for case in expected_cases]
    actual_ids = [str(row.get("id") or "") for row in rows]
    duplicates = _duplicate_ids(actual_ids)
    if duplicates:
        raise ValueError(
            f"Paper protocol compiled artifact has duplicate IDs: {duplicates[:20]}"
        )
    if actual_ids != expected_ids:
        actual_set = set(actual_ids)
        expected_set = set(expected_ids)
        missing = [case_id for case_id in expected_ids if case_id not in actual_set]
        unexpected = [case_id for case_id in actual_ids if case_id not in expected_set]
        order_only = not missing and not unexpected
        raise ValueError(
            "Paper protocol compiled case universe mismatch: "
            f"expected={len(expected_ids)}, actual={len(actual_ids)}, "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"order_mismatch={order_only}"
        )
    case_binding_mismatches = [
        str(row.get("id") or "")
        for row, expected_case in zip(rows, expected_cases)
        if row.get("case") != expected_case
    ]
    if case_binding_mismatches:
        raise ValueError(
            "Paper protocol compiled case/manifest binding mismatch for "
            f"{case_binding_mismatches[:20]}"
        )
    invalid_failure_rows = []
    for row in rows:
        valid = bool(row.get("valid"))
        failure_class = row.get("failure_class")
        if (
            (valid and failure_class is not None)
            or (not valid and failure_class not in ALL_FAILURE_CLASSES)
        ):
            invalid_failure_rows.append({
                "id": row.get("id"),
                "valid": valid,
                "failure_class": failure_class,
            })
    if invalid_failure_rows:
        raise ValueError(
            "Paper protocol compiled failure classification mismatch: "
            f"{invalid_failure_rows[:20]}"
        )
    return contract


def _effective_attempts(task: str, refine_attempts: Optional[int]) -> int:
    """Max generation attempts for a task (1 = single-shot, no refine)."""
    if task not in REFINE_TASKS:
        return 1
    if refine_attempts is None:
        return DEFAULT_REFINE_ATTEMPTS
    return max(1, refine_attempts)


def infer(
    task: str,
    fmt: str,
    model: str,
    *,
    split: str = "demo",
    limit: Optional[int] = None,
    text_mode: str = "parametric",
    dry_run: bool = False,
    out: Path,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    refine_attempts: Optional[int] = None,
) -> Path:
    task_obj = resolve_task(task)
    fmt_obj = resolve_format(fmt)
    task_obj.check_format(fmt_obj)

    cases = load_cases(task, split, limit=limit)
    client = None if dry_run else get_client(model, config_dir)
    max_attempts = _effective_attempts(task, refine_attempts)

    rows = []
    for rc in cases:
        bundle = task_obj.build_prompt(
            fmt_obj, rc.case, [str(p) for p in rc.image_paths], text_mode=text_mode
        )
        row = {
            "id": rc.id,
            "task": task,
            "format": fmt,
            "model": model,
            "split": split,
            "text_mode": text_mode,
            "case": rc.case.to_dict(),
            "prompt": {"system": bundle.system, "user": bundle.user, "images": bundle.images},
            "raw_text": None,
            "code": None,
            "usage": {},
            "error": None,
            "failure_class": None,
        }
        if dry_run:
            rows.append(row)
            continue
        if max_attempts > 1:
            _infer_with_refine(client, fmt_obj, bundle, row, max_attempts=max_attempts)
        else:
            _infer_single_shot(client, fmt_obj, bundle, row)
        rows.append(row)

    write_jsonl(out, rows)
    logger.info("infer: wrote %d predictions -> %s", len(rows), out)
    return out


def _infer_single_shot(client, fmt_obj, bundle, row: dict) -> None:
    """One call, scored as-is (Text-to-3D, or refine disabled). API-level retry
    for transient transport failures still happens inside ``client.generate``."""
    try:
        resp = client.generate(bundle.user, images=bundle.images, system=bundle.system)
        row["raw_text"] = resp.text
        row["code"] = fmt_obj.extract_code(resp.text)
        row["usage"] = resp.usage
        row["failure_class"] = None
        if not row["code"].strip():
            row["error"] = "empty code extraction"
            row["failure_class"] = FAILURE_GENERATION_INVALID
    except Exception as exc:  # a failed call is just an error state
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["failure_class"] = FAILURE_INFERENCE_GAP
        logger.warning("infer failed for %s: %s", row["id"], exc)


def _infer_with_refine(client, fmt_obj, bundle, row: dict, *, max_attempts: int) -> None:
    """Compile-check-retry with error feedback (image-/assembly-3d).

    Port of cadbenchmark ``generate_with_retry``: generate -> extract -> compile;
    on an invalid compile, feed the error back and regenerate, up to
    ``max_attempts``. An LLM-side failure (call raised, or empty extraction) or an
    export-timeout-only failure stops the loop early — error feedback would be
    useless there. The intermediate compiles run in a temp dir purely to drive the
    loop; the authoritative artifacts are produced later by the ``compile`` stage.
    """
    user_prompt = bundle.user
    history: list[dict] = []
    total_usage: dict = {}
    code = ""
    last_error: Optional[str] = None
    valid = False
    failure_class: Optional[str] = None

    with tempfile.TemporaryDirectory(prefix="p3d_refine_") as td:
        td = Path(td)
        for attempt in range(1, max_attempts + 1):
            try:
                resp = client.generate(user_prompt, images=bundle.images, system=bundle.system)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                failure_class = FAILURE_INFERENCE_GAP
                logger.warning("infer %s attempt %d/%d failed: %s",
                               row["id"], attempt, max_attempts, last_error)
                history.append({"attempt": attempt, "valid": False, "errors": [last_error],
                                "llm_failed": True, "error_stage": "llm_generate"})
                break  # LLM-side failure: stop (feedback can't help)

            row["raw_text"] = resp.text
            _merge_usage(total_usage, resp.usage)
            code = fmt_obj.extract_code(resp.text)
            if not code.strip():
                last_error = "empty code extraction"
                failure_class = FAILURE_GENERATION_INVALID
                history.append({"attempt": attempt, "valid": False, "errors": [last_error],
                                "llm_failed": True, "error_stage": "llm_extract"})
                break  # no code to feed back

            try:
                cr = fmt_obj.compile(code, td / f"attempt_{attempt}")
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                failure_class = FAILURE_EVALUATOR_GAP
                history.append({
                    "attempt": attempt,
                    "valid": False,
                    "errors": [last_error],
                    "llm_failed": False,
                    "error_stage": "compile_evaluator",
                })
                break
            valid = bool(cr.valid)
            errors = list(cr.errors) if cr.errors else ([] if valid else ["invalid (no error message)"])
            primary = (cr.error_details or [{}])[0]
            sig = _error_signature(errors, primary)
            repeat = _count_repeats(history, sig)
            history.append({"attempt": attempt, "valid": valid, "errors": errors[:5],
                            "error_signature": sig, "repeat_count": repeat,
                            "error_stage": primary.get("stage"), "llm_failed": False})

            if valid:
                last_error = None
                failure_class = None
                logger.info("infer %s attempt %d/%d: valid", row["id"], attempt, max_attempts)
                break

            last_error = errors[0] if errors else "invalid"
            logger.info("infer %s attempt %d/%d: invalid — %s",
                        row["id"], attempt, max_attempts, errors[:2])
            failure_class = _compile_result_failure_class(cr.to_dict())
            if failure_class == FAILURE_EVALUATOR_GAP:
                failure_class = FAILURE_EVALUATOR_GAP
                break  # evaluator/runtime gaps cannot be fixed by regeneration
            if attempt < max_attempts:
                user_prompt = _build_refine_prompt(
                    bundle.user, code, errors, fmt_obj,
                    has_images=bool(bundle.images), error_detail=primary, repeat_count=repeat,
                )
                time.sleep(1)

    row["code"] = code
    row["usage"] = total_usage
    row["error"] = None if valid else last_error
    row["failure_class"] = None if valid else (
        failure_class or FAILURE_GENERATION_INVALID
    )
    row["attempts"] = len(history)
    row["attempt_history"] = history


# -- refine helpers (cadbenchmark port) ------------------------------------
_FENCE_LANG = {"minimal-json": "json", "openscad": "scad",
               "cadquery": "python", "threejs": "javascript"}


def _build_refine_prompt(original_user: str, code: str, errors: list, fmt_obj, *,
                         has_images: bool, error_detail: dict, repeat_count: int) -> str:
    """Rebuild the user prompt with the previous code + compile error fed back."""
    error_text = "\n".join(str(e) for e in errors) if errors else "Unknown error"
    lang = _FENCE_LANG.get(fmt_obj.slug, "")

    diag = []
    for key, label in (("stage", "Failure stage"), ("error_type", "Error type"),
                       ("line_number", "Failing line number"), ("line_text", "Failing line text")):
        val = (error_detail or {}).get(key)
        if val is not None and val != "":
            diag.append(f"- {label}: {val}")
    diag_block = ("Additional diagnostics:\n" + "\n".join(diag) + "\n") if diag else ""

    tb = (error_detail or {}).get("traceback")
    tb_block = f"Traceback:\n```\n{tb}\n```\n" if tb else ""

    escalation = ""
    if repeat_count >= 2:
        escalation = (
            f"\nIMPORTANT: This same failure has happened {repeat_count} times in a row. "
            "Do not make a superficial edit. Replace or rewrite the failing section more "
            "substantially, targeting the real root cause.\n"
        )

    image_hint = ""
    if has_images:
        image_hint = ("\nIMPORTANT: Look at the provided image(s) carefully. Your fixed code must "
                      "still accurately match the 3D geometry shown in the image(s).\n")

    return (
        f"{original_user}\n\n"
        "---\n"
        f"Your previous attempt produced the following {fmt_obj.display_name} code, but it "
        "failed to compile/export:\n\n"
        f"```{lang}\n{code}\n```\n\n"
        f"The error was:\n```\n{error_text}\n```\n"
        f"{diag_block}{tb_block}{escalation}{image_hint}\n"
        "Please fix the error and generate a corrected version of the complete code. "
        "Output ONLY the fixed code."
    )


def _error_signature(errors: list, primary: dict) -> str:
    msg = str(errors[0]).strip() if errors else "invalid_without_error"
    stage = (primary or {}).get("stage") or "unknown_stage"
    line = (primary or {}).get("line_number")
    return f"{stage}|{line}|{msg}"


def _count_repeats(history: list, sig: str) -> int:
    """How many consecutive recent attempts ended on this same error signature."""
    count = 1
    for prev in reversed(history):
        if prev.get("valid") or prev.get("error_signature") != sig:
            break
        count += 1
    return count


def _is_export_timeout_only(errors: list) -> bool:
    """True iff every error is a wall-clock export timeout (feedback won't help)."""
    if not errors:
        return False
    return all(isinstance(e, str) and "timed out" in e.lower() for e in errors)


def _compile_result_failure_class(result: dict) -> str:
    """Separate model-code invalidity from evaluator/runtime unavailability."""
    details = list(result.get("error_details") or [])
    error_types = {
        str(detail.get("error_type") or "").lower()
        for detail in details
    }
    stages = {
        str(detail.get("stage") or "").lower()
        for detail in details
    }
    text = "\n".join(
        [str(error) for error in (result.get("errors") or [])]
        + [str(detail.get("message") or "") for detail in details]
    ).lower()
    if (
        _is_export_timeout_only(result.get("errors") or [])
        or error_types & {
            "filenotfounderror",
            "missingdependencyerror",
            "timeoutexpired",
            "timeout",
        }
        or "compile_evaluator" in stages
        or any(marker in text for marker in (
            "openscad not found. install",
            "missing dependency",
            "required dependency",
            "binary unavailable",
            "executable unavailable",
        ))
    ):
        return FAILURE_EVALUATOR_GAP
    return FAILURE_GENERATION_INVALID


def _merge_usage(total: dict, addition: dict) -> None:
    """Accumulate token-usage counters across refine attempts."""
    if not isinstance(addition, dict):
        return
    for key, value in addition.items():
        if isinstance(value, bool):
            total.setdefault(key, value)
        elif isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        elif isinstance(value, dict):
            sub = total.setdefault(key, {})
            if isinstance(sub, dict):
                _merge_usage(sub, value)
        else:
            total.setdefault(key, value)


# --------------------------------------------------------------------------
# Stage 2: compile
# --------------------------------------------------------------------------
def compile_predictions(pred_path: Path, *, out: Path, work_dir: Path) -> Path:
    rows = []
    for row in read_jsonl(pred_path):
        case_dir = Path(work_dir) / row["id"].replace("/", "_")
        case_dir.mkdir(parents=True, exist_ok=True)
        result = {"valid": False, "stl": None, "step": None, "parts_meta": None,
                  "parts_dir": None, "errors": [], "error_details": []}
        code = row.get("code")
        failure_class = row.get("failure_class")
        if not code:
            result["errors"] = [row.get("error") or "no code to compile"]
        else:
            fmt_obj = resolve_format(row["format"])
            try:
                cr = fmt_obj.compile(code, case_dir)
                result = cr.to_dict()
            except Exception as exc:
                result["errors"] = [f"{type(exc).__name__}: {exc}"]
                result["error_details"] = [{
                    "stage": "compile_evaluator",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }]
                failure_class = FAILURE_EVALUATOR_GAP
        if result["valid"]:
            failure_class = None
        elif failure_class not in ALL_FAILURE_CLASSES:
            failure_class = _compile_result_failure_class(result)
        rows.append({
            **row,
            "compile": result,
            "valid": result["valid"],
            "failure_class": failure_class,
        })

    write_jsonl(out, rows)
    n_valid = sum(1 for r in rows if r["valid"])
    logger.info("compile: %d/%d valid -> %s", n_valid, len(rows), out)
    return out


# --------------------------------------------------------------------------
# Stage 3: score
# --------------------------------------------------------------------------
def score(
    compiled_path: Path,
    metric: str,
    *,
    out: Path,
    work_dir: Path,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    protocol_id: Optional[str] = None,
) -> Path:
    compiled_rows = list(read_jsonl(compiled_path))
    if not compiled_rows:
        write_jsonl(out, [])
        return out

    task = compiled_rows[0]["task"]
    buckets = resolve_metric_buckets(metric, task)
    case_universe = (
        _validate_formal_compiled_rows(compiled_rows, buckets)
        if protocol_id == PAPER_PROTOCOL_ID else None
    )

    judge_client = decompose_client = None
    if "judge" in buckets or "part" in buckets:
        jc = load_judge_config(config_dir)
        if "judge" in buckets:
            judge_client = _try_client(jc.judge_model, config_dir)
        if "part" in buckets:
            decompose_client = _try_client(jc.decompose_model, config_dir)

    rows = []
    for row in compiled_rows:
        rc = _resolve_case(row)
        case_dir = Path(work_dir) / row["id"].replace("/", "_")
        ctx = ScoreContext(
            task=row["task"],
            fmt=row["format"],
            case=rc,
            compiled=row.get("compile", {}),
            work_dir=case_dir,
            judge_client=judge_client,
            decompose_client=decompose_client,
            protocol_id=protocol_id,
            shared={"stage1_code": row.get("code"), "text_mode": row.get("text_mode", "parametric")},
        )
        raw_metrics: dict = {}
        failure_class = row.get("failure_class")
        if failure_class in GAP_FAILURE_CLASSES:
            failure_message = (
                row.get("error")
                or next(iter((row.get("compile") or {}).get("errors") or []), None)
                or "upstream gap"
            )
            raw_metrics["_pipeline_error"] = (
                f"{failure_class}: {failure_message}"
            )
        else:
            for bucket_name in buckets:
                try:
                    bucket = get_metric_bucket(bucket_name)
                    raw_metrics.update(bucket.score(ctx) or {})
                except Exception as exc:
                    logger.warning("score bucket %s failed for %s: %s", bucket_name, row["id"], exc)
                    raw_metrics[f"_{bucket_name}_error"] = f"{type(exc).__name__}: {exc}"
        sidecar_path = None
        calls = ctx.shared.get("evaluation_calls") or []
        if calls:
            sidecar_relative = (
                Path("evaluation_meta")
                / f"{row['id'].replace('/', '_')}.json"
            )
            sidecar = out.parent / sidecar_relative
            write_json(sidecar, {
                "protocol_id": protocol_id,
                "case_id": row["id"],
                "task": row["task"],
                "format": row["format"],
                "calls": calls,
            })
            sidecar_path = sidecar_relative.as_posix()
        rows.append(
            {
                "id": row["id"],
                "task": row["task"],
                "format": row["format"],
                "model": row["model"],
                "split": row["split"],
                "text_mode": row.get("text_mode", "parametric"),
                "valid": bool(row.get("valid")),
                "failure_class": failure_class,
                "buckets": buckets,
                "raw_metrics": raw_metrics,
                "protocol_id": protocol_id,
                "case_universe": case_universe,
                "evaluation_meta_path": sidecar_path,
            }
        )

    write_jsonl(out, rows)
    logger.info("score: scored %d cases (%s) -> %s", len(rows), ",".join(buckets), out)
    return out


# --------------------------------------------------------------------------
# Stage 4: summarize
# --------------------------------------------------------------------------
def summarize(metrics_path: Path, *, out: Path) -> Path:
    rows = list(read_jsonl(metrics_path))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["format"], row["model"])].append(row)

    protocol_ids = {row.get("protocol_id") for row in rows}
    summary = {
        "protocol_id": (
            next(iter(protocol_ids)) if len(protocol_ids) == 1 else "mixed"
        ),
        "groups": [],
        "task_profiles": [],
    }
    formal_tasks = {
        str(row.get("task"))
        for row in rows
        if row.get("protocol_id") == PAPER_PROTOCOL_ID
    }
    mixed_formal_task_profile = len(formal_tasks) > 1
    profile_inputs: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for (task, fmt, model), grp in groups.items():
        n = len(grp)
        n_valid = sum(1 for r in grp if r["valid"])
        bucket_sums: dict[str, list[float]] = defaultdict(list)
        bucket_gaps: dict[str, list[dict]] = defaultdict(list)
        requested_buckets = set().union(*(set(r.get("buckets") or []) for r in grp))
        protocol_values = {r.get("protocol_id") for r in grp}
        formal = PAPER_PROTOCOL_ID in protocol_values
        promotion_blockers: list[str] = []
        failure_gaps = [
            {
                "id": row.get("id"),
                "failure_class": row.get("failure_class"),
                "error": (row.get("raw_metrics") or {}).get(
                    "_pipeline_error"
                ),
            }
            for row in grp
            if row.get("failure_class") in GAP_FAILURE_CLASSES
        ]
        if formal and protocol_values != {PAPER_PROTOCOL_ID}:
            promotion_blockers.append("mixed_protocol")
        if formal and mixed_formal_task_profile:
            promotion_blockers.append("mixed_task_profile")
        if formal and failure_gaps:
            promotion_blockers.append("inference_or_evaluator_gap")

        text_modes = {str(r.get("text_mode", "parametric")) for r in grp}
        formats = {str(r.get("format")) for r in grp}
        splits = {str(r.get("split")) for r in grp}
        panels = {tuple(r.get("buckets") or []) for r in grp}
        if formal and len(text_modes) != 1:
            promotion_blockers.append("mixed_text_mode")
        if formal and formats != {fmt}:
            promotion_blockers.append("mixed_format")
        if formal and len(splits) != 1:
            promotion_blockers.append("mixed_split")
        if formal and len(panels) != 1:
            promotion_blockers.append("mixed_metric_panel")

        expected_panel = set(resolve_metric_buckets("all", task))
        if formal and requested_buckets != expected_panel:
            promotion_blockers.append("partial_metric_panel")

        for r in grp:
            scoring_valid = (
                True
                if r.get("failure_class") in GAP_FAILURE_CLASSES
                else r["valid"]
            )
            per_bucket = bucket_score_for_case(
                task,
                r["raw_metrics"],
                scoring_valid,
                r.get("text_mode", "parametric"),
                required_buckets=(requested_buckets if formal else None),
            )
            for b, v in per_bucket.items():
                if v is not None:
                    bucket_sums[b].append(v)
            if formal:
                for bucket, missing in missing_required_metrics(
                    task,
                    r["raw_metrics"],
                    scoring_valid,
                    r.get("text_mode", "parametric"),
                    required_buckets=requested_buckets,
                ).items():
                    bucket_gaps[bucket].append({
                        "id": r["id"],
                        "missing": missing,
                        "error": next(
                            (value for key, value in r["raw_metrics"].items()
                             if key.startswith("_") and key.endswith("_error")),
                            None,
                        ),
                    })

        bucket_means = (
            {
                bucket: (sum(values) / len(values))
                for bucket, values in bucket_sums.items()
                if values and not bucket_gaps.get(bucket) and len(values) == n
            }
            if formal else
            {
                bucket: (sum(values) / len(values))
                for bucket, values in bucket_sums.items()
                if values
            }
        )
        representative_mode = (
            next(iter(text_modes)) if len(text_modes) == 1 else "parametric"
        )
        applicable_score_buckets = set(bucket_membership(
            task, representative_mode
        )) & set(SCORE_BUCKETS)
        complete_full_panel = (
            applicable_score_buckets.issubset(requested_buckets)
            and all(bucket in bucket_means for bucket in applicable_score_buckets)
        )
        score_buckets = [bucket_means[b] for b in SCORE_BUCKETS
                         if b in applicable_score_buckets and b in bucket_means]
        headline = (
            sum(score_buckets) / len(score_buckets) * 100.0
            if score_buckets
            and (complete_full_panel or not formal)
            and not promotion_blockers else None
        )
        if formal and bucket_gaps:
            promotion_blockers.append("required_metric_gap")

        case_universe = None
        if formal:
            actual_ids = [str(r.get("id") or "") for r in grp]
            duplicates = _duplicate_ids(actual_ids)
            expected_ids: list[str] = []
            universe_error = None
            if len(splits) == 1:
                try:
                    expected_ids, expected_contract = _formal_expected_case_ids(
                        task, next(iter(splits))
                    )
                except ValueError as exc:
                    expected_contract = {}
                    universe_error = str(exc)
            else:
                expected_contract = {}
                universe_error = "mixed split values"

            expected_set = set(expected_ids)
            actual_set = set(actual_ids)
            missing = [
                case_id for case_id in expected_ids if case_id not in actual_set
            ]
            unexpected = [
                case_id for case_id in actual_ids if case_id not in expected_set
            ] if expected_ids else list(actual_ids)
            order_match = bool(expected_ids) and actual_ids == expected_ids
            contract_fields = (
                "task_profile",
                "expected_count",
                "ordered_ids_sha256",
                "ordered_source_ids_sha256",
                "uids_sha256",
                "dataset_revision",
                "dataset_manifest_sha256",
                "qa_source_sha256",
                "qa_dataset_content_sha256",
                "qa_bank_version",
            )
            embedded_contracts = {
                tuple(
                    (r.get("case_universe") or {}).get(field)
                    for field in contract_fields
                )
                for r in grp
            }
            contract_match = (
                len(embedded_contracts) == 1
                and expected_contract
                and next(iter(embedded_contracts)) == tuple(
                    expected_contract.get(field)
                    for field in contract_fields
                )
            )
            universe_complete = bool(
                not universe_error
                and not duplicates
                and not missing
                and not unexpected
                and order_match
                and contract_match
            )
            case_universe = {
                **expected_contract,
                "actual_count": len(actual_ids),
                "actual_unique_count": len(actual_set),
                "duplicate_ids": duplicates[:20],
                "missing_ids": missing[:20],
                "unexpected_ids": unexpected[:20],
                "order_match": order_match,
                "embedded_contract_match": contract_match,
                "error": universe_error,
                "complete": universe_complete,
            }
            if not universe_complete:
                promotion_blockers.append("case_universe_mismatch")

        metric_coverage = {}
        if formal and "geometry" in requested_buckets:
            iou_eligible = iou_measured = iou_inapplicable = 0
            iou_applicability_gaps: list[dict] = []
            generation_invalid = 0
            for row in grp:
                if row.get("failure_class") in GAP_FAILURE_CLASSES:
                    iou_applicability_gaps.append({
                        "id": row.get("id"),
                        "reason": row.get("failure_class"),
                    })
                    continue
                if not row.get("valid"):
                    generation_invalid += 1
                    continue
                raw = row.get("raw_metrics") or {}
                applicable = iou_applicability(raw, True)
                if applicable is False:
                    iou_inapplicable += 1
                elif applicable is None:
                    iou_applicability_gaps.append({
                        "id": row.get("id"),
                        "reason": "missing_or_nonfinite_open_edge_ratio",
                    })
                else:
                    iou_eligible += 1
                    if normalize_value("iou", raw.get("iou")) is not None:
                        iou_measured += 1
                    else:
                        iou_applicability_gaps.append({
                            "id": row.get("id"),
                            "reason": "eligible_iou_missing",
                        })
            metric_coverage["iou"] = {
                "eligible_cases": iou_eligible,
                "measured_cases": iou_measured,
                "inapplicable_cases": iou_inapplicable,
                "generation_invalid_cases": generation_invalid,
                "gap_count": len(iou_applicability_gaps),
                "gaps": iou_applicability_gaps[:20],
            }

        promotion_blockers = list(dict.fromkeys(promotion_blockers))
        group_summary = {
            "task": task,
            "format": fmt,
            "model": model,
            "n_cases": n,
            "valid_rate": (
                None
                if formal and failure_gaps
                else (n_valid / n if n else 0.0)
            ),
            "buckets": {b: round(v, 4) for b, v in bucket_means.items()},
            "score": round(headline, 2) if headline is not None else None,
        }
        if formal:
            group_summary.update({
                "evaluation_gaps": dict(bucket_gaps),
                "evaluation_gap_count": sum(len(items) for items in bucket_gaps.values()),
                "promotion_ready": not promotion_blockers,
                "promotion_blockers": promotion_blockers,
                "case_universe": case_universe,
                "metric_coverage": metric_coverage,
                "failure_gaps": failure_gaps[:20],
                "failure_gap_count": len(failure_gaps),
            })
        summary["groups"].append(group_summary)
        profile_inputs[(task, model, representative_mode)].append({
            "format": fmt,
            "bucket_means": bucket_means,
            "formal": formal,
            "promotion_ready": (
                not promotion_blockers if formal else complete_full_panel
            ),
        })

    # Task-level cells are equal means over the task's supported output formats.
    # Per-format groups above remain unchanged and retain their own case counts.
    for (task, model, text_mode), inputs in profile_inputs.items():
        supported_formats = list(resolve_task(task).supported_formats)
        by_format = {item["format"]: item for item in inputs}
        present_formats = [
            fmt for fmt in supported_formats if fmt in by_format
        ]
        complete_formats = present_formats == supported_formats
        applicable = set(bucket_membership(task, text_mode)) & set(
            SCORE_BUCKETS
        )
        profile_buckets: dict[str, float] = {}
        for bucket in SCORE_BUCKETS:
            if bucket not in applicable or not complete_formats:
                continue
            values = [
                by_format[fmt]["bucket_means"].get(bucket)
                for fmt in supported_formats
            ]
            if all(value is not None for value in values):
                profile_buckets[bucket] = sum(values) / len(values)
        promotion_ready = bool(
            complete_formats
            and all(by_format[fmt]["promotion_ready"] for fmt in supported_formats)
            and applicable.issubset(profile_buckets)
        )
        profile_score = (
            sum(profile_buckets[bucket] for bucket in SCORE_BUCKETS
                if bucket in applicable)
            / len(applicable)
            * 100.0
            if promotion_ready and applicable else None
        )
        summary["task_profiles"].append({
            "task": task,
            "model": model,
            "text_mode": text_mode,
            "format_aggregation": "equal_mean",
            "supported_formats": supported_formats,
            "present_formats": present_formats,
            "complete_formats": complete_formats,
            "buckets": {
                bucket: round(value, 4)
                for bucket, value in profile_buckets.items()
            },
            "score": round(profile_score, 2) if profile_score is not None else None,
            "promotion_ready": promotion_ready,
            "promotion_blockers": (
                [] if promotion_ready else
                (["missing_supported_format"] if not complete_formats else
                 ["incomplete_format_group"])
            ),
        })

    write_json(out, summary)
    logger.info("summarize: %d group(s) -> %s", len(summary["groups"]), out)
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _resolve_case(row: dict) -> ResolvedCase:
    return ResolvedCase(Case.from_dict(row["case"]), data_root(row["split"]))


def _try_client(model_name: str, config_dir: Path):
    """Build a judge/decompose client, or None for a clean skip (e.g. no API key)."""
    try:
        client = get_client(model_name, config_dir)
        _ = client.cfg.api_key  # validate the key is present up front
        return client
    except Exception as exc:
        logger.warning("judge/part client '%s' unavailable (%s); that bucket will be skipped",
                       model_name, exc)
        return None
