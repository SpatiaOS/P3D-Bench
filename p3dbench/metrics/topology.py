"""Topology bucket — per-case mesh closedness and surface consistency.

Three raw sub-metrics derived from the predicted mesh's edge structure (paper
NoOE / InvN / NM, App. bucket details):

- ``no_open_edge`` (NoOE, higher better): per-case closedness gate — ``1.0`` if the
  mesh has zero open (border) edges, else ``0.0``. This is the binarized form of
  the underlying ``open_edge_ratio``; the paper reports the fraction of watertight
  cases, so each case contributes a hard 0/1.
- ``inverted_normal_ratio`` (InvN, lower better): fraction of manifold edges with
  inconsistent winding (passed through).
- ``non_manifold_edge_ratio`` (NM, lower better): fraction of unique edges shared by
  >= 3 faces (passed through).

The edge math itself lives in :func:`p3dbench.metrics.geometry.compute_topology_metrics`
(shared with the geometry pipeline); this module only loads the predicted mesh and
maps the result onto the canonical raw keys.
"""

from __future__ import annotations

from .base import MetricBucket, ScoreContext
from .geometry import compute_topology_metrics


class _TopologyBucket(MetricBucket):
    bucket = "topology"
    requires: set[str] = {"mesh"}

    def score(self, ctx: ScoreContext) -> dict:
        from ..compile.step_mesh import load_mesh

        stl_path = ctx.compiled.get("stl")
        mesh = load_mesh(stl_path) if stl_path else None
        if mesh is None:
            return {
                "no_open_edge": None,
                "inverted_normal_ratio": None,
                "non_manifold_edge_ratio": None,
            }

        topo = compute_topology_metrics(mesh)
        return {
            # Per-case closedness: hard 0/1 (paper NoOE).
            "no_open_edge": 1.0 if topo["open_edge_ratio"] == 0 else 0.0,
            "inverted_normal_ratio": topo["inverted_normal_ratio"],
            "non_manifold_edge_ratio": topo["non_manifold_edge_ratio"],
        }


BUCKET = _TopologyBucket()
