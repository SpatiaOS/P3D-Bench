"""Modular dispatch: three orthogonal registries (task / format / metric).

Resolving a ``--task T --format F --metric M`` triple:
1. look up T; assert F in T.supported_formats;
2. infer builds the prompt from T + F.system_guidelines, calls the model -> predictions;
3. compile runs F.compile per prediction -> compiled;
4. score loads the metric buckets for M and runs each bucket's ``score`` -> metrics;
5. summarize normalizes + aggregates -> summary.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .formats import FORMATS, get_format
from .metrics.base import ALL_BUCKETS
from .tasks import TASKS, get_task

if TYPE_CHECKING:
    from .formats.base import Format
    from .metrics.base import MetricBucket
    from .tasks.base import Task

# Bucket name -> module path under p3dbench.metrics (each exposes BUCKET).
_METRIC_MODULES = {
    "valid": "p3dbench.metrics.valid",
    "geometry": "p3dbench.metrics.geometry",
    "topology": "p3dbench.metrics.topology",
    "judge": "p3dbench.metrics.judge",
    "part": "p3dbench.metrics.part",
}


def resolve_task(slug: str) -> "Task":
    return get_task(slug)


def resolve_format(slug: str) -> "Format":
    return get_format(slug)


def resolve_metric_buckets(metric: str, task: str) -> list[str]:
    """Expand a ``--metric`` choice into a list of bucket names valid for ``task``.

    ``all`` expands to every bucket the task actually reports (e.g. Part only for
    Assembly-3D); a single bucket name is returned as-is.
    """
    task_buckets = _buckets_for_task(task)
    if metric == "all":
        return [b for b in ALL_BUCKETS if b in task_buckets]
    if metric not in ALL_BUCKETS:
        raise ValueError(f"Unknown metric '{metric}'. Choices: {', '.join(ALL_BUCKETS)}, all")
    if metric not in task_buckets:
        raise ValueError(
            f"Metric bucket '{metric}' is not reported for task '{task}'. "
            f"This task reports: {', '.join(task_buckets)}"
        )
    return [metric]


def _buckets_for_task(task: str) -> list[str]:
    # Valid + the score buckets the task reports. Part only on Assembly-3D.
    if task == "assembly-3d":
        return ["valid", "geometry", "topology", "judge", "part"]
    if task == "text-to-3d":
        return ["valid", "geometry", "topology", "judge"]
    if task == "image-to-3d":
        return ["valid", "geometry", "topology", "judge"]
    raise ValueError(f"Unknown task '{task}'")


def get_metric_bucket(name: str) -> "MetricBucket":
    if name not in _METRIC_MODULES:
        raise KeyError(f"Unknown metric bucket '{name}'. Choices: {', '.join(_METRIC_MODULES)}")
    module = importlib.import_module(_METRIC_MODULES[name])
    return module.BUCKET


def validate_triple(task: str, fmt: str) -> None:
    t = resolve_task(task)
    f = resolve_format(fmt)
    t.check_format(f)


__all__ = [
    "TASKS",
    "FORMATS",
    "resolve_task",
    "resolve_format",
    "resolve_metric_buckets",
    "get_metric_bucket",
    "validate_triple",
]
