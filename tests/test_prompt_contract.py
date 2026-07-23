from __future__ import annotations

import json
import re

from p3dbench.formats.minimal_json import SYSTEM_GUIDELINES as JSON_GUIDELINES
from p3dbench.formats.openscad import SYSTEM_GUIDELINES as OPENSCAD_GUIDELINES


def test_openscad_prompt_preserves_stated_millimeter_values() -> None:
    assert (
        "Treat all stated linear dimensions as millimeters, without unit conversion "
        "or global rescaling."
    ) in OPENSCAD_GUIDELINES
    assert "Keep dimensions in millimeters for consistency" not in OPENSCAD_GUIDELINES


def test_text2cad_json_example_is_valid_json() -> None:
    match = re.search(
        r"```json\n(?P<example>.*?)```",
        JSON_GUIDELINES,
        flags=re.DOTALL,
    )
    assert match is not None

    example = json.loads(match.group("example"))
    feature = example["parts"]["part_1"]
    assert set(feature) == {"coordinate_system", "sketch", "extrusion"}
