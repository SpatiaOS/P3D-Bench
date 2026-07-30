"""Text-to-3D task: parametric/descriptive text -> CAD program."""

from __future__ import annotations

from ..data.schema import Case
from ..formats.base import Format
from ..text_condition import resolve_text_condition
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

    def build_prompt(
        self, fmt: Format, case: Case, image_paths: list[str], *, text_mode: str = "parametric"
    ) -> PromptBundle:
        self.check_format(fmt)
        # Shared with the descriptive J-Sem judge so both see the same condition.
        text = resolve_text_condition(case, text_mode)
        user = PROMPT_TEMPLATE.format(display_name=fmt.display_name, text=text)
        return PromptBundle(system=fmt.system_guidelines, user=user, images=[])


TASK = TextTo3DTask()
