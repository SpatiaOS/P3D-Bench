"""Manifest record schema.

One JSONL row per case under ``data/manifests/<task>_<split>.jsonl``. Paths in
``input``/``target`` are relative to the data root (demo: in-repo ``data/demo/``;
full split: the HuggingFace download cache).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CaseInput:
    text: str = ""
    image_paths: list[str] = field(default_factory=list)
    # Assembly-3D only: structured per-part inventory (role / semantic / gt part path).
    part_annotations: list[dict] = field(default_factory=list)


@dataclass
class CaseTarget:
    format: str = "minimal-json"          # GT program format (or "step" when GT is direct STEP)
    code_path: Optional[str] = None       # GT program (minimal-json); None for direct-STEP GT
    step_path: Optional[str] = None       # GT STEP
    mesh_path: Optional[str] = None       # GT STL/OBJ (tessellated)
    render_paths: list[str] = field(default_factory=list)   # GT multiview renders (judge)
    part_paths: list[str] = field(default_factory=list)     # GT per-part meshes (Assembly-3D)
    qa_bank_path: Optional[str] = None    # prebuilt QA bank (Text-to-3D judge bucket)


@dataclass
class Case:
    id: str
    task: str
    split: str
    input: CaseInput
    target: CaseTarget
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict) -> "Case":
        return cls(
            id=row["id"],
            task=row["task"],
            split=row.get("split", "demo"),
            input=CaseInput(**(row.get("input") or {})),
            target=CaseTarget(**(row.get("target") or {})),
            metadata=row.get("metadata") or {},
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "split": self.split,
            "input": {
                "text": self.input.text,
                "image_paths": self.input.image_paths,
                "part_annotations": self.input.part_annotations,
            },
            "target": {
                "format": self.target.format,
                "code_path": self.target.code_path,
                "step_path": self.target.step_path,
                "mesh_path": self.target.mesh_path,
                "render_paths": self.target.render_paths,
                "part_paths": self.target.part_paths,
                "qa_bank_path": self.target.qa_bank_path,
            },
            "metadata": self.metadata,
        }
