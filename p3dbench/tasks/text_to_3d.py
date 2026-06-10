"""Text-to-3D task: parametric/descriptive text -> CAD program."""

from __future__ import annotations

from ..data.schema import Case
from ..formats.base import Format
from .base import PromptBundle, Task

PROMPT_TEMPLATE = """\
Generate {display_name} code for the following CAD model:

{text}

Requirements:
- Use parametric design with clear variable definitions
- Include comments explaining each step
- Make the code clean and well-structured
- Ensure all dimensions are clearly defined\
"""


class TextTo3DTask(Task):
    slug = "text-to-3d"
    supported_formats = ("minimal-json", "openscad")
    condition_inputs = "text"

    def build_prompt(self, fmt: Format, case: Case, image_paths: list[str]) -> PromptBundle:
        self.check_format(fmt)
        user = PROMPT_TEMPLATE.format(display_name=fmt.display_name, text=case.input.text)
        return PromptBundle(system=fmt.system_guidelines, user=user, images=[])


TASK = TextTo3DTask()
