"""Image-to-3D task: one rendered image -> CAD program."""

from __future__ import annotations

from ..data.schema import Case
from ..formats.base import Format
from .base import PromptBundle, Task

PROMPT_TEMPLATE = """\
Analyze the image and generate {display_name} code to recreate this 3D model.

Requirements:
- Carefully observe the geometry, dimensions, and features in the image
- Estimate reasonable dimensions based on the proportions shown
- Use parametric design with clear variable definitions
- Include comments explaining your interpretation
- Make the code clean and well-structured\
"""


class ImageTo3DTask(Task):
    slug = "image-to-3d"
    supported_formats = ("openscad", "cadquery", "threejs")
    condition_inputs = "image"

    def build_prompt(
        self, fmt: Format, case: Case, image_paths: list[str], *, text_mode: str = "parametric"
    ) -> PromptBundle:
        self.check_format(fmt)
        if not image_paths:
            raise ValueError(f"Image-to-3D case {case.id} has no input image")
        user = PROMPT_TEMPLATE.format(display_name=fmt.display_name)
        return PromptBundle(system=fmt.system_guidelines, user=user, images=image_paths[:1])


TASK = ImageTo3DTask()
