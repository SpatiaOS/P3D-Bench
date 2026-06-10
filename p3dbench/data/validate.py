"""Manifest integrity + referenced-file existence checks."""

from __future__ import annotations

from pathlib import Path

from ..utils import read_jsonl
from .loader import data_root, manifest_path
from .schema import Case


def validate_split(split: str = "demo", manifest_dir: Path = Path("data/manifests")) -> dict:
    """Return a report dict; raises nothing — surfaces problems as a list."""
    root = data_root(split)
    report = {"split": split, "tasks": {}, "ok": True, "problems": []}

    for task in ("text-to-3d", "image-to-3d", "assembly-3d"):
        path = manifest_path(task, split, manifest_dir)
        task_report = {"manifest": str(path), "cases": 0, "missing_files": 0}
        if not path.exists():
            task_report["status"] = "missing"
            report["ok"] = False
            report["problems"].append(f"{task}: missing manifest {path}")
            report["tasks"][task] = task_report
            continue
        seen_ids: set[str] = set()
        for row in read_jsonl(path):
            case = Case.from_dict(row)
            task_report["cases"] += 1
            if case.id in seen_ids:
                report["problems"].append(f"{task}: duplicate id {case.id}")
                report["ok"] = False
            seen_ids.add(case.id)
            for rel in _referenced_paths(case):
                if rel and not (root / rel).exists():
                    task_report["missing_files"] += 1
                    report["problems"].append(f"{case.id}: missing {rel}")
                    report["ok"] = False
        task_report["status"] = "ok" if task_report["missing_files"] == 0 else "incomplete"
        report["tasks"][task] = task_report

    return report


def _referenced_paths(case: Case) -> list[str]:
    t = case.target
    paths = list(case.input.image_paths)
    paths += [t.code_path, t.step_path, t.mesh_path, t.qa_bank_path]
    paths += list(t.render_paths) + list(t.part_paths)
    return [p for p in paths if p]
