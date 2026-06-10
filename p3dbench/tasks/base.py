"""Task ABC: builds the task- and format-aware prompt for one case."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..data.schema import Case
from ..formats.base import Format


@dataclass
class PromptBundle:
    """Everything one inference call needs."""

    system: str            # format.system_guidelines (optionally task-augmented)
    user: str              # task.build_prompt text
    images: list[str]      # local image paths to attach (may be empty)


class Task(ABC):
    slug: str
    supported_formats: tuple[str, ...]
    condition_inputs: str   # "text" | "image" | "image+text"

    @abstractmethod
    def build_prompt(self, fmt: Format, case: Case, image_paths: list[str]) -> PromptBundle:
        ...

    def check_format(self, fmt: Format) -> None:
        if fmt.slug not in self.supported_formats:
            raise ValueError(
                f"Task '{self.slug}' does not support format '{fmt.slug}'. "
                f"Supported: {', '.join(self.supported_formats)}"
            )
