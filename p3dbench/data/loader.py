"""Load cases from manifests and resolve their data paths.

- ``demo`` split: manifests + data ship in-repo (``data/manifests/``, ``data/demo/``).
- ``full`` split: materialized locally by ``p3dbench download --split full`` into
  ``data/full/`` + ``data/manifests/*_full.jsonl`` (see ``data.full_builder``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from ..utils import read_jsonl
from .schema import Case

MANIFEST_DIR = Path("data/manifests")
DATA_ROOTS = {"demo": Path("data/demo"), "full": Path("data/full")}

# CLI slug -> manifest task token (manifests use the same slug).
_TASK_TOKEN = {"text-to-3d": "text_to_3d", "image-to-3d": "image_to_3d", "assembly-3d": "assembly_3d"}


@dataclass
class ResolvedCase:
    """A Case with all paths resolved to absolute filesystem paths."""

    case: Case
    data_root: Path

    @property
    def id(self) -> str:
        return self.case.id

    def _abs(self, rel: Optional[str]) -> Optional[Path]:
        if not rel:
            return None
        return (self.data_root / rel).resolve()

    @property
    def image_paths(self) -> list[Path]:
        return [self._abs(p) for p in self.case.input.image_paths if p]

    @property
    def gt_step(self) -> Optional[Path]:
        return self._abs(self.case.target.step_path)

    @property
    def gt_mesh(self) -> Optional[Path]:
        return self._abs(self.case.target.mesh_path)

    @property
    def gt_code(self) -> Optional[Path]:
        return self._abs(self.case.target.code_path)

    @property
    def gt_renders(self) -> list[Path]:
        return [self._abs(p) for p in self.case.target.render_paths if p]

    @property
    def gt_parts(self) -> list[Path]:
        return [self._abs(p) for p in self.case.target.part_paths if p]

    @property
    def qa_bank(self) -> Optional[Path]:
        return self._abs(self.case.target.qa_bank_path)


def manifest_path(task: str, split: str, manifest_dir: Path = MANIFEST_DIR) -> Path:
    token = _TASK_TOKEN.get(task, task.replace("-", "_"))
    return Path(manifest_dir) / f"{token}_{split}.jsonl"


def data_root(split: str) -> Path:
    if split not in DATA_ROOTS:
        raise ValueError(f"Unknown split '{split}'. Choices: {', '.join(DATA_ROOTS)}")
    return DATA_ROOTS[split]


def load_cases(
    task: str,
    split: str = "demo",
    *,
    limit: Optional[int] = None,
    manifest_dir: Path = MANIFEST_DIR,
) -> list[ResolvedCase]:
    path = manifest_path(task, split, manifest_dir)
    if not path.exists():
        if split == "full":
            raise FileNotFoundError(
                f"Full-split manifest {path} not found. Materialize it first with "
                "`p3dbench download --split full --source-root <path>` (see docs/DATA.md), "
                "or use `--split demo` for the in-repo demo cases."
            )
        raise FileNotFoundError(f"Manifest not found: {path}")
    root = data_root(split)
    cases: list[ResolvedCase] = []
    for row in read_jsonl(path):
        cases.append(ResolvedCase(Case.from_dict(row), root))
        if limit is not None and len(cases) >= limit:
            break
    return cases


def iter_all_manifests(split: str = "demo", manifest_dir: Path = MANIFEST_DIR) -> Iterator[Path]:
    for token in _TASK_TOKEN.values():
        path = Path(manifest_dir) / f"{token}_{split}.jsonl"
        if path.exists():
            yield path
