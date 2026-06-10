"""Format registry: slug -> Format instance."""

from __future__ import annotations

from .base import CompileResult, Format
from .cadquery import CADQUERY_FORMAT
from .minimal_json import MINIMAL_JSON_FORMAT
from .openscad import OPENSCAD_FORMAT
from .threejs import THREEJS_FORMAT

FORMATS: dict[str, Format] = {
    "minimal-json": MINIMAL_JSON_FORMAT,
    "openscad": OPENSCAD_FORMAT,
    "cadquery": CADQUERY_FORMAT,
    "threejs": THREEJS_FORMAT,
}


def get_format(slug: str) -> Format:
    slug = slug.lower()
    if slug not in FORMATS:
        raise KeyError(f"Unknown format '{slug}'. Choices: {', '.join(FORMATS)}")
    return FORMATS[slug]


__all__ = ["FORMATS", "Format", "CompileResult", "get_format"]
