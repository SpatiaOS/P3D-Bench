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
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .config import DEFAULT_CONFIG_DIR, load_judge_config
from .data.loader import ResolvedCase, data_root, load_cases
from .data.schema import Case
from .metrics.base import (
    SCORE_BUCKETS,
    ScoreContext,
    bucket_score_for_case,
)
from .models import get_client
from .registry import (
    get_metric_bucket,
    resolve_format,
    resolve_metric_buckets,
    resolve_task,
)
from .utils import read_jsonl, write_json, write_jsonl

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Stage 1: infer
# --------------------------------------------------------------------------
# Tasks that get the compile-check-retry error-feedback loop (cadbenchmark port).
# Text-to-3D stays single-shot by design.
REFINE_TASKS = ("image-to-3d", "assembly-3d")
DEFAULT_REFINE_ATTEMPTS = 3


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
            # True only when the API never returned a usable response: the case
            # counts as untested (dropped from every metric and from the Valid
            # denominator), not as a model failure.
            "llm_failed": False,
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
        if not row["code"].strip():
            # The model answered, just not with usable code: that is a model
            # failure (valid=False, worst-filled), not an untested case.
            row["error"] = "empty code extraction"
    except Exception as exc:  # the call itself failed -> untested
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["llm_failed"] = True
        logger.warning("infer failed for %s: %s", row["id"], exc)


def _infer_with_refine(client, fmt_obj, bundle, row: dict, *, max_attempts: int) -> None:
    """Compile-check-retry with error feedback (image-/assembly-3d).

    Port of cadbenchmark ``generate_with_retry``: generate -> extract -> compile;
    on an invalid compile, feed the error back and regenerate, up to
    ``max_attempts``. An LLM-side failure (call raised, or empty extraction) or an
    export-timeout-only failure stops the loop early — error feedback would be
    useless there. ``row["llm_failed"]`` is set only when no attempt ever returned
    a response, so a model that answered with uncompilable code is still scored.
    The intermediate compiles run in a temp dir purely to drive the loop; the
    authoritative artifacts are produced later by the ``compile`` stage.
    """
    user_prompt = bundle.user
    history: list[dict] = []
    total_usage: dict = {}
    code = ""
    last_error: Optional[str] = None
    valid = False
    responded = False

    with tempfile.TemporaryDirectory(prefix="p3d_refine_") as td:
        td = Path(td)
        for attempt in range(1, max_attempts + 1):
            try:
                resp = client.generate(user_prompt, images=bundle.images, system=bundle.system)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("infer %s attempt %d/%d failed: %s",
                               row["id"], attempt, max_attempts, last_error)
                history.append({"attempt": attempt, "valid": False, "errors": [last_error],
                                "llm_failed": True, "error_stage": "llm_generate"})
                break  # LLM-side failure: stop (feedback can't help)

            responded = True
            row["raw_text"] = resp.text
            _merge_usage(total_usage, resp.usage)
            code = fmt_obj.extract_code(resp.text)
            if not code.strip():
                # Answered but unparseable: a model failure, not an API failure.
                last_error = "empty code extraction"
                history.append({"attempt": attempt, "valid": False, "errors": [last_error],
                                "llm_failed": False, "api_parse_failed": True,
                                "error_stage": "llm_extract"})
                break  # no code to feed back

            cr = fmt_obj.compile(code, td / f"attempt_{attempt}")
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
                logger.info("infer %s attempt %d/%d: valid", row["id"], attempt, max_attempts)
                break

            last_error = errors[0] if errors else "invalid"
            logger.info("infer %s attempt %d/%d: invalid — %s",
                        row["id"], attempt, max_attempts, errors[:2])
            if _is_export_timeout_only(errors):
                break  # timeouts: feedback is useless, code preserved as-is
            if attempt < max_attempts:
                user_prompt = _build_refine_prompt(
                    bundle.user, code, errors, fmt_obj,
                    has_images=bool(bundle.images), error_detail=primary, repeat_count=repeat,
                )
                time.sleep(1)

    row["code"] = code
    row["usage"] = total_usage
    row["error"] = None if valid else last_error
    row["llm_failed"] = not valid and not responded
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
        if not code:
            result["errors"] = [row.get("error") or "no code to compile"]
        else:
            fmt_obj = resolve_format(row["format"])
            try:
                cr = fmt_obj.compile(code, case_dir)
                result = cr.to_dict()
            except Exception as exc:
                # One case must never abort the whole batch: record it as
                # invalid (worst-filled downstream, like any compile failure).
                logger.warning("compile crashed for %s: %s", row["id"], exc)
                result["errors"] = [f"{type(exc).__name__}: {exc}"]
                result["error_details"] = [{
                    "stage": "compile",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }]
        rows.append({**row, "compile": result, "valid": result["valid"]})

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
) -> Path:
    compiled_rows = list(read_jsonl(compiled_path))
    if not compiled_rows:
        write_jsonl(out, [])
        return out

    task = compiled_rows[0]["task"]
    buckets = resolve_metric_buckets(metric, task)

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
            shared={"stage1_code": row.get("code"), "text_mode": row.get("text_mode", "parametric")},
        )
        raw_metrics: dict = {}
        llm_failed = bool(row.get("llm_failed"))
        if llm_failed:
            # Untested: there is no model output to measure. summarize() drops
            # the case rather than scoring it.
            logger.info("score: skipping %s (llm_failed)", row["id"])
        else:
            for bucket_name in buckets:
                try:
                    bucket = get_metric_bucket(bucket_name)
                    raw_metrics.update(bucket.score(ctx) or {})
                except Exception as exc:
                    logger.warning("score bucket %s failed for %s: %s", bucket_name, row["id"], exc)
                    raw_metrics[f"_{bucket_name}_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "id": row["id"],
                "task": row["task"],
                "format": row["format"],
                "model": row["model"],
                "split": row["split"],
                "text_mode": row.get("text_mode", "parametric"),
                "valid": bool(row.get("valid")),
                "llm_failed": llm_failed,
                "buckets": buckets,
                "raw_metrics": raw_metrics,
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

    summary = {"groups": []}
    for (task, fmt, model), grp in groups.items():
        n = len(grp)
        # An LLM API failure means the case was never tested: it is dropped from
        # every metric AND from the Valid denominator, and reported on its own as
        # llm_fail_rate. A model that *answered* with unusable code is a
        # different thing — that stays valid=False and gets worst-filled.
        tested = [r for r in grp if not r.get("llm_failed")]
        n_llm_fail = n - len(tested)
        n_tested = len(tested)
        n_valid = sum(1 for r in tested if r["valid"])
        bucket_sums: dict[str, list[float]] = defaultdict(list)
        for r in tested:
            per_bucket = bucket_score_for_case(
                task, r["raw_metrics"], r["valid"], r.get("text_mode", "parametric")
            )
            for b, v in per_bucket.items():
                if v is not None:
                    bucket_sums[b].append(v)
        bucket_means = {b: (sum(v) / len(v)) for b, v in bucket_sums.items() if v}
        score_buckets = [bucket_means[b] for b in SCORE_BUCKETS if b in bucket_means]
        headline = (sum(score_buckets) / len(score_buckets) * 100.0) if score_buckets else None
        summary["groups"].append(
            {
                "task": task,
                "format": fmt,
                "model": model,
                "n_cases": n,
                "n_tested": n_tested,
                "llm_fail_cases": n_llm_fail,
                "llm_fail_rate": round(n_llm_fail / n, 4) if n else 0.0,
                "valid_rate": (n_valid / n_tested) if n_tested else None,
                "buckets": {b: round(v, 4) for b, v in bucket_means.items()},
                "score": round(headline, 2) if headline is not None else None,
            }
        )

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
