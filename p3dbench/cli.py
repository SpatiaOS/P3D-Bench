"""p3dbench command-line entry point.

Subcommands: infer | compile | score | summarize | run | download | validate.
Each of the first four reads/writes a plain JSONL/JSON artifact; ``run`` chains
all four. An evaluation is one (task x format x metric) triple, selected
independently from the CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import FORMAT_SLUGS, METRIC_SLUGS, TASK_SLUGS, __version__
from .config import DEFAULT_CONFIG_DIR, load_dotenv
from .utils import setup_logging

RESULTS_DIR = Path("results")


def _run_id(args) -> str:
    parts = [getattr(args, k, None) for k in ("task", "format", "model", "metric")]
    return "_".join(str(p).replace("-", "") for p in parts if p)


def _stage_paths(run_dir: Path) -> dict:
    return {
        "predictions": run_dir / "predictions.jsonl",
        "compiled": run_dir / "compiled.jsonl",
        "metrics": run_dir / "metrics.jsonl",
        "summary": run_dir / "summary.json",
        "work": run_dir / "work",
    }


def cmd_infer(args) -> int:
    from .pipeline import infer

    out = Path(args.out) if args.out else RESULTS_DIR / _run_id(args) / "predictions.jsonl"
    infer(
        args.task, args.format, args.model,
        split=args.split, limit=args.limit, text_mode=args.text_mode,
        dry_run=args.dry_run, out=out, config_dir=Path(args.config_dir),
    )
    print(f"predictions -> {out}")
    return 0


def cmd_compile(args) -> int:
    from .pipeline import compile_predictions

    pred = Path(args.pred)
    out = Path(args.out) if args.out else pred.with_name("compiled.jsonl")
    work = Path(args.work_dir) if args.work_dir else pred.parent / "work"
    compile_predictions(pred, out=out, work_dir=work)
    print(f"compiled -> {out}")
    return 0


def cmd_score(args) -> int:
    from .pipeline import score

    compiled = Path(args.compiled)
    out = Path(args.out) if args.out else compiled.with_name("metrics.jsonl")
    work = Path(args.work_dir) if args.work_dir else compiled.parent / "work"
    score(compiled, args.metric, out=out, work_dir=work, config_dir=Path(args.config_dir))
    print(f"metrics -> {out}")
    return 0


def cmd_summarize(args) -> int:
    from .pipeline import summarize

    metrics = Path(args.metrics)
    out = Path(args.out) if args.out else metrics.with_name("summary.json")
    summarize(metrics, out=out)
    _print_summary(out)
    return 0


def cmd_run(args) -> int:
    from .pipeline import compile_predictions, infer, score, summarize

    run_dir = RESULTS_DIR / _run_id(args)
    paths = _stage_paths(run_dir)
    infer(
        args.task, args.format, args.model,
        split=args.split, limit=args.limit, text_mode=args.text_mode,
        dry_run=args.dry_run, out=paths["predictions"], config_dir=Path(args.config_dir),
    )
    if args.dry_run:
        print(f"dry-run predictions -> {paths['predictions']}")
        return 0
    compile_predictions(paths["predictions"], out=paths["compiled"], work_dir=paths["work"])
    score(paths["compiled"], args.metric, out=paths["metrics"],
          work_dir=paths["work"], config_dir=Path(args.config_dir))
    summarize(paths["metrics"], out=paths["summary"])
    _print_summary(paths["summary"])
    return 0


def cmd_download(args) -> int:
    from .scripts.download_data import download

    download(args.split)
    return 0


def cmd_validate(args) -> int:
    from .data.validate import validate_split

    report = validate_split(args.split)
    print(f"split={report['split']}  ok={report['ok']}")
    for task, tr in report["tasks"].items():
        print(f"  {task:12s} {tr.get('status','?'):10s} cases={tr.get('cases',0)} "
              f"missing_files={tr.get('missing_files',0)}")
    for problem in report["problems"][:20]:
        print(f"  ! {problem}")
    return 0 if report["ok"] else 1


def _print_summary(summary_path: Path) -> None:
    import json

    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    print(f"\nsummary -> {summary_path}")
    for g in data.get("groups", []):
        buckets = "  ".join(f"{b}={v:.3f}" for b, v in g["buckets"].items())
        score = "-" if g["score"] is None else f"{g['score']:.2f}"
        print(f"  [{g['task']}/{g['format']}] {g['model']}  "
              f"valid={g['valid_rate']:.2f}  {buckets}  Score={score}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p3dbench", description="P3D-Bench evaluation harness")
    p.add_argument("--version", action="version", version=f"p3dbench {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common_infer(sp):
        sp.add_argument("--task", required=True, choices=TASK_SLUGS)
        sp.add_argument("--format", required=True, choices=FORMAT_SLUGS)
        sp.add_argument("--model", required=True, help="model name from configs/models.yaml")
        sp.add_argument("--split", default="demo", choices=["demo", "full"])
        sp.add_argument("--limit", type=int, default=None, help="first N cases")
        sp.add_argument("--text-mode", default="parametric",
                        choices=["parametric", "descriptive"], help="Text-to-3D only")
        sp.add_argument("--dry-run", action="store_true",
                        help="build prompts / validate config without calling a model")
        sp.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))

    sp = sub.add_parser("infer", help="run a model -> predictions.jsonl")
    add_common_infer(sp)
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_infer)

    sp = sub.add_parser("compile", help="predictions -> compiled.jsonl (STEP/STL + valid)")
    sp.add_argument("--pred", required=True)
    sp.add_argument("--out")
    sp.add_argument("--work-dir")
    sp.set_defaults(func=cmd_compile)

    sp = sub.add_parser("score", help="compiled -> metrics.jsonl for one metric bucket")
    sp.add_argument("--compiled", required=True)
    sp.add_argument("--metric", required=True, choices=[*METRIC_SLUGS, "all"])
    sp.add_argument("--out")
    sp.add_argument("--work-dir")
    sp.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("summarize", help="metrics -> summary.json")
    sp.add_argument("--metrics", required=True)
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_summarize)

    sp = sub.add_parser("run", help="chain infer -> compile -> score -> summarize")
    add_common_infer(sp)
    sp.add_argument("--metric", required=True, choices=[*METRIC_SLUGS, "all"])
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("download", help="fetch a data split (full = HuggingFace)")
    sp.add_argument("--split", default="demo", choices=["demo", "full"])
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("validate", help="manifest integrity + referenced-file check")
    sp.add_argument("--split", default="demo", choices=["demo", "full"])
    sp.set_defaults(func=cmd_validate)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    load_dotenv()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
