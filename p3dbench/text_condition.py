"""Shared Text-to-3D condition resolution for generation and evaluation."""

from __future__ import annotations

from .data.schema import Case


def resolve_text_condition(case: Case, text_mode: str) -> str:
    """Return the exact condition shown to the generator for ``text_mode``."""
    if text_mode not in {"parametric", "descriptive"}:
        raise ValueError(f"Unsupported Text-to-3D mode: {text_mode!r}")
    parametric = (case.input.text or "").strip()
    if text_mode == "descriptive":
        descriptive = str(case.metadata.get("text_desc") or "").strip()
        return descriptive or parametric
    return parametric
