"""Metric bucket interface + normalization / aggregation.

Five buckets — ``valid``, ``geometry``, ``topology``, ``judge``, ``part`` — each a
plug-in resolved by :mod:`p3dbench.registry`. A bucket's ``score`` returns RAW
sub-metric values (or ``None`` when a metric does not apply to the task/format);
normalization to [0,1] and bucket/headline aggregation happen here at summarize
time, following the paper (App. bucket details).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# --------------------------------------------------------------------------
# Scoring context handed to each bucket
# --------------------------------------------------------------------------
@dataclass
class ScoreContext:
    task: str
    fmt: str
    case: Any                 # data.loader.ResolvedCase
    compiled: dict            # one compiled.jsonl record (valid, stl, step, parts_meta, ...)
    work_dir: Path
    judge_client: Any = None          # models.ModelClient or None
    decompose_client: Any = None      # models.ModelClient or None
    # Cross-bucket cache (e.g. geometry stores align_transform_4x4 for part metric).
    shared: dict = field(default_factory=dict)


class MetricBucket(ABC):
    """One module per bucket. ``requires`` declares heavy capabilities so the CLI
    can report a clear message when an optional extra / external runtime is absent."""

    bucket: str
    requires: set[str] = set()   # subset of {"mesh", "render", "judge_model", "parts"}

    @abstractmethod
    def score(self, ctx: ScoreContext) -> dict[str, Any]:
        """Return raw sub-metric values keyed by the canonical metric keys below."""


# --------------------------------------------------------------------------
# Sub-metric normalization specs (paper App. bucket details)
#   normalized value is always in [0,1], 1 = best.
#   A prediction that fails the Valid gate is worst-filled -> normalized 0.0.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricSpec:
    key: str
    bucket: str
    label: str                       # paper-facing abbreviation
    normalize: Callable[[float], float]
    higher_is_better: bool = True


def _identity01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _judge_1to10(v: float) -> float:
    return max(0.0, min(1.0, (float(v) - 1.0) / 9.0))


def _one_minus(v: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(v)))


def _cd_capped(v: float, cap: float = 0.01) -> float:
    return max(0.0, min(1.0, 1.0 - float(v) / cap))


# Unbounded-lower-better CD worst cap (paper WORST_FILL["chamfer_distance"]).
CD_WORST_CAP = 0.01

METRIC_SPECS: dict[str, MetricSpec] = {
    # geometry
    "chamfer_distance": MetricSpec("chamfer_distance", "geometry", "CD", _cd_capped, False),
    "f_score_005": MetricSpec("f_score_005", "geometry", "F@.05", _identity01),
    "f_score_001": MetricSpec("f_score_001", "geometry", "F@.01", _identity01),
    "normal_consistency": MetricSpec("normal_consistency", "geometry", "NC", _identity01),
    "iou": MetricSpec("iou", "geometry", "IoU", _identity01),
    # topology
    "no_open_edge": MetricSpec("no_open_edge", "topology", "NoOE", _identity01),
    "inverted_normal_ratio": MetricSpec("inverted_normal_ratio", "topology", "InvN", _one_minus, False),
    "non_manifold_edge_ratio": MetricSpec("non_manifold_edge_ratio", "topology", "NM", _one_minus, False),
    # judge
    "qa_semantic": MetricSpec("qa_semantic", "judge", "QA-S", _identity01),
    "qa_param": MetricSpec("qa_param", "judge", "QA-P", _identity01),
    "judge_semantic": MetricSpec("judge_semantic", "judge", "J-Sem", _judge_1to10),
    "judge_geometry": MetricSpec("judge_geometry", "judge", "J-Geo", _judge_1to10),
    "judge_aesthetics": MetricSpec("judge_aesthetics", "judge", "J-Aes", _judge_1to10),
    # part
    "part_match_f1": MetricSpec("part_match_f1", "part", "PartMatchF1", _identity01),
    "part_fs": MetricSpec("part_fs", "part", "PartFS", _identity01),
}

ALL_BUCKETS = ("valid", "geometry", "topology", "judge", "part")
# Buckets that contribute to the headline Score (Valid reported alongside, excluded).
SCORE_BUCKETS = ("geometry", "topology", "judge", "part")


# --------------------------------------------------------------------------
# Task/format-conditioned bucket membership (which sub-metrics apply)
# --------------------------------------------------------------------------
def bucket_membership(task: str, text_mode: str = "parametric") -> dict[str, list[str]]:
    """Which sub-metric keys are members of each bucket for this task.

    ``text_mode`` ('parametric' | 'descriptive') only matters for Text-to-3D.
    """
    geo = ["chamfer_distance", "f_score_005", "f_score_001", "normal_consistency", "iou"]
    topo = ["no_open_edge", "inverted_normal_ratio", "non_manifold_edge_ratio"]

    if task == "text-to-3d":
        if text_mode == "descriptive":
            return {"judge": ["qa_semantic", "judge_semantic"]}
        # parametric: full geometry/topology + QA judge
        return {"geometry": geo, "topology": topo, "judge": ["qa_semantic", "qa_param"]}

    judge_visual = ["judge_semantic", "judge_geometry", "judge_aesthetics"]
    if task == "image-to-3d":
        return {"geometry": geo, "topology": topo, "judge": judge_visual}
    if task == "assembly-3d":
        return {
            "geometry": geo,
            "topology": topo,
            "judge": judge_visual,
            "part": ["part_match_f1", "part_fs"],
        }
    raise ValueError(f"Unknown task '{task}'")


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def normalize_value(key: str, value: Optional[float]) -> Optional[float]:
    spec = METRIC_SPECS.get(key)
    if spec is None or value is None:
        return None
    try:
        return spec.normalize(value)
    except (TypeError, ValueError):
        return None


def bucket_score_for_case(
    task: str,
    raw_metrics: dict[str, Any],
    valid: bool,
    text_mode: str = "parametric",
) -> dict[str, Optional[float]]:
    """Normalized per-bucket score for ONE case.

    Worst-fill: a case failing the Valid gate contributes 0.0 for every member
    sub-metric (worst value -> normalized 0). A member sub-metric that is simply
    not measured for a valid case (e.g. IoU skipped because the mesh is open) is
    dropped from that bucket's mean rather than zero-filled.
    """
    membership = bucket_membership(task, text_mode)
    out: dict[str, Optional[float]] = {}
    for bucket, keys in membership.items():
        vals: list[float] = []
        for key in keys:
            if not valid:
                vals.append(0.0)
                continue
            nv = normalize_value(key, raw_metrics.get(key))
            if nv is not None:
                vals.append(nv)
        out[bucket] = (sum(vals) / len(vals)) if vals else None
    return out
