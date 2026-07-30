"""The minimal-JSON example in the system guidelines must stay parseable.

The example is what the model copies; if it drifts from the compiler contract
the drift is silent, so parse it in CI.
"""

from __future__ import annotations

import json
import re

from p3dbench.formats.minimal_json import SYSTEM_GUIDELINES as JSON_GUIDELINES


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
