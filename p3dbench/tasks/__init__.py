"""Task registry: slug (+ legacy aliases) -> Task instance."""

from __future__ import annotations

from .assembly_3d import TASK as ASSEMBLY_3D
from .base import PromptBundle, Task
from .image_to_3d import TASK as IMAGE_TO_3D
from .text_to_3d import TASK as TEXT_TO_3D

TASKS: dict[str, Task] = {
    "text-to-3d": TEXT_TO_3D,
    "image-to-3d": IMAGE_TO_3D,
    "assembly-3d": ASSEMBLY_3D,
}

# Old research-code names -> release slugs.
TASK_ALIASES = {
    "text2cad": "text-to-3d",
    "image2cad": "image-to-3d",
    "text_image2cad": "assembly-3d",
    "textimage2cad": "assembly-3d",
}


def get_task(slug: str) -> Task:
    slug = TASK_ALIASES.get(slug, slug).lower()
    if slug not in TASKS:
        raise KeyError(f"Unknown task '{slug}'. Choices: {', '.join(TASKS)}")
    return TASKS[slug]


__all__ = ["TASKS", "TASK_ALIASES", "Task", "PromptBundle", "get_task"]
