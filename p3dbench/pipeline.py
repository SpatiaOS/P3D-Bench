"""The four staged functions: infer -> compile -> score -> summarize.

Each stage reads and writes a plain JSONL/JSON artifact and carries no hidden
state, so the same predictions can be re-scored under a different metric without
re-running inference. Records are self-contained (they embed the case) so a
downstream stage never has to re-read the manifest.
"""

from __future__ import annotations

import logging
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
) -> Path:
    task_obj = resolve_task(task)
    fmt_obj = resolve_format(fmt)
    task_obj.check_format(fmt_obj)

    cases = load_cases(task, split, limit=limit)
    client = None if dry_run else get_client(model, config_dir)

    rows = []
    for rc in cases:
        bundle = task_obj.build_prompt(fmt_obj, rc.case, [str(p) for p in rc.image_paths])
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
        }
        if dry_run:
            rows.append(row)
            continue
        try:
            resp = client.generate(bundle.user, images=bundle.images, system=bundle.system)
            row["raw_text"] = resp.text
            row["code"] = fmt_obj.extract_code(resp.text)
            row["usage"] = resp.usage
            if not row["code"].strip():
                row["error"] = "empty code extraction"
        except Exception as exc:  # single-shot: a failed call is just an error state
            row["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("infer failed for %s: %s", rc.id, exc)
        rows.append(row)

    write_jsonl(out, rows)
    logger.info("infer: wrote %d predictions -> %s", len(rows), out)
    return out


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
            cr = fmt_obj.compile(code, case_dir)
            result = cr.to_dict()
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
        n_valid = sum(1 for r in grp if r["valid"])
        bucket_sums: dict[str, list[float]] = defaultdict(list)
        for r in grp:
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
                "valid_rate": n_valid / n if n else 0.0,
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
