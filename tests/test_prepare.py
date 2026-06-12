"""PREPARE stage: reproduce the demo _shared_cache artifacts from raw upstream.

These are heavy, environment-gated checks: they need a Blender binary on
``$P3DBENCH_BLENDER``, Xvfb + OCP, and a local raw upstream tree. Point them at the
raw tree with ``$P3DBENCH_TEST_SOURCE_ROOT`` (defaults to the package's
``DEFAULT_SOURCE_ROOT``). They are skipped automatically when those aren't present,
so the light suite still runs in CI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from p3dbench.data.full_builder import DEFAULT_SOURCE_ROOT
from p3dbench.render import blender

# Demo UID picks (mirror scripts/build_demo_data.py) so we can compare to the
# shipped demo + the research _shared_cache golden.
_ASSEMBLY_UID = "117698_aca36590"
_TEXT_UID = "0002/00022687"

_SOURCE_ROOT = Path(os.environ.get("P3DBENCH_TEST_SOURCE_ROOT", str(DEFAULT_SOURCE_ROOT)))

_skip_no_raw = pytest.mark.skipif(
    not _SOURCE_ROOT.exists(), reason=f"raw source-root not present: {_SOURCE_ROOT}")
_skip_no_blender = pytest.mark.skipif(
    not blender.is_blender_available(),
    reason="Blender not available (set $P3DBENCH_BLENDER)")


def _hf_anno(task_dir: str, uid: str) -> dict:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("SpatiaOS/P3D-Bench", f"data/{task_dir}/annotations.jsonl",
                        repo_type="dataset")
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("uid") == uid:
            return json.loads(line)
    return {}


@_skip_no_raw
@_skip_no_blender
def test_prepare_assembly_reproduces_cache(tmp_path):
    from p3dbench.data.prepare import SharedDataCache, render_modes_for_task
    from p3dbench.data.raw_loaders import build_assembly_case_data

    cd = build_assembly_case_data(_ASSEMBLY_UID, _SOURCE_ROOT, _hf_anno("assembly_3d", _ASSEMBLY_UID))
    assert cd is not None and cd["part_match"] == "part_id"

    cache = SharedDataCache(tmp_path / "fusion")
    manifest = cache.prepare_case(_ASSEMBLY_UID, cd, render_modes_for_task("assembly-3d"),
                                  overwrite=True)
    assert manifest is not None and manifest["status"] == "complete"

    cdir = cache.case_dir(_ASSEMBLY_UID)
    assert (cdir / "gt_model.stl").stat().st_size > 0
    assert (cdir / "renders/occ/single_view/gt_render.png").exists()
    assert len(list((cdir / "renders/blender_clay/multiview").glob("view_*.png"))) == 4
    assert len(list((cdir / "gt_parts").glob("part_*.stl"))) == len(cd["gt_parts"])
    # manifest gt_parts carry HF role/semantic + instance_count and a part STL each.
    for gp in manifest["gt_parts"]:
        assert gp.get("part_id") and gp.get("role_name")
        assert "instance_count" in gp
        assert Path(gp["stl_path"]).exists()

    # Golden: part count matches the research _shared_cache when present.
    research = _SOURCE_ROOT / "fusion360/assembly/_shared_cache" / _ASSEMBLY_UID / "manifest.json"
    if research.exists():
        rm = json.loads(research.read_text(encoding="utf-8"))
        assert len(manifest["gt_parts"]) == len(rm.get("gt_parts", []))


@_skip_no_raw
def test_prepare_text_generates_step_stl(tmp_path):
    from p3dbench.data.prepare import SharedDataCache, render_modes_for_task
    from p3dbench.data.raw_loaders import build_text_case_data

    cd = build_text_case_data(_TEXT_UID, _SOURCE_ROOT, _hf_anno("text_to_3d", _TEXT_UID))
    if cd is None:
        pytest.skip(f"text uid {_TEXT_UID} not in local source-root")

    cache = SharedDataCache(tmp_path / "text2cad")
    manifest = cache.prepare_case(_TEXT_UID, cd, render_modes_for_task("text-to-3d"),
                                  overwrite=True)
    assert manifest is not None and manifest["status"] == "complete"
    cdir = cache.case_dir(_TEXT_UID)
    assert (cdir / "gt_model.step").exists()
    assert (cdir / "gt_model.stl").stat().st_size > 0
    assert (cdir / "renders/occ/single_view/gt_render.png").exists()


@_skip_no_raw
@_skip_no_blender
def test_prepare_then_materialize_roundtrip(tmp_path, monkeypatch):
    """End-to-end: prepare a fresh cache, materialize via build_full, validate the row."""
    import p3dbench.data.full_builder as fb
    from p3dbench.data.prepare import prepare_task

    # Build a fresh cache for one assembly UID under a scratch fusion root that
    # symlinks the raw assembly dir but starts with an empty _shared_cache.
    scratch = tmp_path / "src"
    (scratch / "fusion360/assembly").mkdir(parents=True)
    real_assembly = _SOURCE_ROOT / "fusion360/assembly/assembly"
    (scratch / "fusion360/assembly/assembly").symlink_to(real_assembly)

    anns = {_ASSEMBLY_UID: _hf_anno("assembly_3d", _ASSEMBLY_UID)}
    rep = prepare_task("assembly-3d", [_ASSEMBLY_UID], scratch, anns, overwrite=True)
    assert rep["built"] == 1, rep

    # Materialize into temp roots so the repo's data/ is untouched.
    monkeypatch.setattr(fb, "FULL_ROOT", tmp_path / "full")
    monkeypatch.setattr(fb, "MANIFEST_DIR", tmp_path / "manifests")
    report = fb.build_full(source_root=scratch, tasks=("assembly-3d",), limit=1, overwrite=True)
    tr = report["tasks"]["assembly-3d"]
    assert tr["built"] == 1, tr

    manifest_path = Path(tr["manifest"])
    row = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["task"] == "assembly-3d"
    assert row["target"]["part_paths"], "expected materialized gt part paths"
    # From-raw path: no research decision.json -> difficulty degrades to "unknown".
    assert row["metadata"]["difficulty"] == "unknown"
    # Per-part annotations carry instance_count (eval-side dedup consistency).
    assert all("instance_count" in a for a in row["input"]["part_annotations"])
