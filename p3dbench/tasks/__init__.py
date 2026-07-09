"""Task registry: slug -> Task instance."""

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

def get_task(slug: str) -> Task:
    slug = slug.lower()
    if slug not in TASKS:
        raise KeyError(f"Unknown task '{slug}'. Choices: {', '.join(TASKS)}")
    return TASKS[slug]


__all__ = ["TASKS", "Task", "PromptBundle", "get_task"]
