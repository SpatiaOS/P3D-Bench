"""Eval-side part dedup consistency with the reference (cadbenchmark) pipeline.

The part metric trusts upstream dedup (skips fingerprint dedup -> source
``loader_instance_count``) iff every GT part it receives carries an
``instance_count``. These checks pin that precondition on the in-repo demo split,
so the assembly part metric stays aligned with the reference behavior. They need
no heavy deps / upstream data (the demo ships in-repo).
"""

from __future__ import annotations

import json
from pathlib import Path

from p3dbench.data.loader import load_cases

REPO = Path(__file__).resolve().parents[1]


def test_gt_parts_meta_carries_instance_count():
    cases = load_cases("assembly-3d", "demo")
    assert cases, "expected in-repo demo assembly cases"
    for rc in cases:
        gpm = rc.gt_parts_meta
        assert gpm, f"{rc.id}: no gt parts"
        # Every part must carry instance_count -> flips the metric to the
        # "loader_already_deduped" branch (source=loader_instance_count), matching
        # the reference pipeline. Missing it would silently re-fingerprint the GT.
        assert all("instance_count" in p for p in gpm), \
            f"{rc.id}: a gt part lacks instance_count -> dedup would diverge"
        # stl_path resolves under the demo data root and exists.
        for p in gpm:
            assert Path(p["stl_path"]).exists(), f"{rc.id}: missing {p['stl_path']}"


def test_gt_parts_meta_matches_manifest_annotations():
    """gt_parts_meta must mirror the manifest's part_annotations (joined by mesh_path)."""
    rows = [json.loads(l) for l in
            (REPO / "data/manifests/assembly_3d_demo.jsonl").read_text().splitlines() if l.strip()]
    by_id = {r["id"]: r for r in rows}
    for rc in load_cases("assembly-3d", "demo"):
        gpm = rc.gt_parts_meta
        annos = by_id[rc.id]["input"]["part_annotations"]
        assert len(gpm) == len(annos)
        for entry, anno in zip(gpm, annos):
            assert entry["instance_count"] == anno["instance_count"]
            assert entry.get("part_id") == anno["part_id"]
            assert Path(entry["stl_path"]).name == Path(anno["mesh_path"]).name
