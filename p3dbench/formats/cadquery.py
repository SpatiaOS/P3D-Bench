"""CadQuery output format."""

from __future__ import annotations

from .base import Format

SYSTEM_GUIDELINES = """\
You are an expert CAD engineer specializing in CadQuery (Python-based parametric CAD).

Generate clean, well-documented CadQuery code that:
- Uses the CadQuery API correctly
- Includes clear parameter definitions at the top
- Has descriptive variable names
- Includes comments explaining the modeling steps
- Exports the final result as 'result'

Example structure:
```python
import cadquery as cq

# Parameters
width = 10.0
height = 10.0
depth = 10.0

# Create the model
result = (
    cq.Workplane("XY")
    .box(width, height, depth)
)
```

Generate ONLY the Python code, no additional explanation.\
"""

CADQUERY_FORMAT = Format(
    slug="cadquery",
    display_name="CadQuery",
    extension=".py",
    system_guidelines=SYSTEM_GUIDELINES,
    fence_langs=("python",),
)
