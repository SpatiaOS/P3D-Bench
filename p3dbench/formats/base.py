"""Format ABC: the prompt side (system guidelines + code extraction).

The compile side (program -> STEP/STL) lives in :mod:`p3dbench.compile`, which
each Format delegates to via :meth:`compile`.
"""

from __future__ import annotations

import re
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Format(ABC):
    """One instance per executable CAD format."""

    slug: str                       # CLI slug: minimal-json | openscad | cadquery | threejs
    display_name: str               # interpolated into task prompts: CadQuery / OpenSCAD / ...
    extension: str                  # .json | .scad | .py | .js
    system_guidelines: str          # format-specific system prompt
    fence_langs: tuple = field(default_factory=tuple)   # extra fenced-code language tags to accept

    def extract_code(self, raw_text: str) -> str:
        """Pull the program out of a model response (first fenced block, else whole text)."""
        langs = "|".join(re.escape(x) for x in ((self.slug, *self.fence_langs)) if x)
        pattern = rf"```(?:{langs})?\s*\n(.*?)```"
        matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        return matches[0].strip() if matches else raw_text.strip()

    def compile(self, code: str, output_dir: Path) -> "CompileResult":
        """Compile ``code`` to STEP/STL. Delegates to p3dbench.compile.exporter."""
        from ..compile.exporter import compile_code

        return compile_code(code, self.slug, Path(output_dir))


@dataclass
class CompileResult:
    """Outcome of compiling one program. ``valid`` == an STL was produced."""

    valid: bool
    stl: Optional[str] = None
    step: Optional[str] = None
    parts_meta: Optional[str] = None      # path to parts_meta.json (assembly decomposition)
    parts_dir: Optional[str] = None       # dir holding per-part STLs
    errors: list[str] = field(default_factory=list)
    error_details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "stl": self.stl,
            "step": self.step,
            "parts_meta": self.parts_meta,
            "parts_dir": self.parts_dir,
            "errors": self.errors,
            "error_details": self.error_details,
        }
