"""P3D-Bench: Benchmarking MLLMs for Parametric 3D Generation and Structural Reasoning.

An evaluation run is defined by three orthogonal choices — task, output format,
and metric bucket — resolved through the registries in :mod:`p3dbench.registry`.
"""

__version__ = "0.1.0"

TASK_SLUGS = ("text-to-3d", "image-to-3d", "assembly-3d")
FORMAT_SLUGS = ("minimal-json", "openscad", "cadquery", "threejs")
METRIC_SLUGS = ("valid", "geometry", "topology", "judge", "part")
