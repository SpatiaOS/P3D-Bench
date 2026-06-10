#!/usr/bin/env python3
"""One-off data-prep helper: assemble the in-repo demo split from local sources.

NOT part of the shipped package or the eval pipeline. It copies 3 small cases
per task out of the research datasets into ``data/demo/`` and writes the matching
manifests under ``data/manifests/``. The full P3D-Dataset lives on HuggingFace
(coming soon); this only produces the tiny smoke-test split.

Run from the repo root:  python scripts/build_demo_data.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "data" / "demo"
MANIFESTS = REPO / "data" / "manifests"

FUSION = Path("/mnt/CFS/yangyikang/cad_dataset/fusion360/assembly")
TEXT2CAD = Path("/mnt/CFS/yangyikang/cad_dataset/text2cad")

RENDER_MAX_EDGE = 512  # downscale GT judge renders to keep the demo light

# Demo case picks (small, low-complexity).
IMAGE_CASES = ["132535_a374e751", "127453_75e818dd", "132687_d1a27238"]
ASSEMBLY_CASES = ["117698_aca36590", "126655_183b0675", "21133_f9a1614e"]
TEXT_CASES = ["0002/00022687", "0035/00351503", "0085/00858822"]


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_render(src: Path, dst: Path) -> None:
    """Copy a render, downscaling to RENDER_MAX_EDGE to keep the repo light."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)
    if max(img.size) > RENDER_MAX_EDGE:
        img.thumbnail((RENDER_MAX_EDGE, RENDER_MAX_EDGE), Image.Resampling.LANCZOS)
    img.save(dst)


def build_image() -> list[dict]:
    rows = []
    for i, uid in enumerate(IMAGE_CASES):
        cid = f"p3d_image-to-3d_{i:06d}"
        cache = FUSION / "_shared_cache" / uid
        decision = json.loads((cache / "_filter" / "decision.json").read_text())

        _copy_render(cache / "renders/occ/single_view/gt_render.png",
                     DEMO / "inputs" / cid / "view_000.png")
        _copy(FUSION / "assembly" / uid / "assembly.step", DEMO / "targets/step" / f"{cid}.step")
        _copy(cache / "gt_model.stl", DEMO / "targets/mesh" / f"{cid}.stl")
        renders = []
        for v in range(4):
            rel = f"targets/renders/{cid}/view_{v:03d}.png"
            _copy_render(cache / f"renders/blender_clay/multiview/view_{v:03d}.png", DEMO / rel)
            renders.append(rel)

        rows.append({
            "id": cid, "task": "image-to-3d", "split": "demo",
            "input": {"text": "", "image_paths": [f"inputs/{cid}/view_000.png"], "part_annotations": []},
            "target": {"format": "step", "code_path": None,
                       "step_path": f"targets/step/{cid}.step",
                       "mesh_path": f"targets/mesh/{cid}.stl",
                       "render_paths": renders, "part_paths": [], "qa_bank_path": None},
            "metadata": {"source": "fusion360-gallery", "source_id": uid,
                         "license_group": "fusion360-gallery",
                         "semantic_category": decision.get("semantic_category"),
                         "difficulty_raw": decision.get("complexity"),
                         "difficulty": _fusion_difficulty(decision.get("complexity"))},
        })
    return rows


def build_assembly() -> list[dict]:
    rows = []
    for i, uid in enumerate(ASSEMBLY_CASES):
        cid = f"p3d_assembly-3d_{i:06d}"
        cache = FUSION / "_shared_cache" / uid
        manifest = json.loads((cache / "manifest.json").read_text())
        decision = json.loads((cache / "_filter" / "decision.json").read_text())
        condition = (cache / "condition.txt").read_text(encoding="utf-8")

        _copy_render(cache / "renders/occ/single_view/gt_render.png",
                     DEMO / "inputs" / cid / "view_000.png")
        _copy(FUSION / "assembly" / uid / "assembly.step", DEMO / "targets/step" / f"{cid}.step")
        _copy(cache / "gt_model.stl", DEMO / "targets/mesh" / f"{cid}.stl")

        part_paths, annotations = [], []
        for gp in manifest["gt_parts"]:
            stl_name = Path(gp["stl_path"]).name
            rel = f"targets/parts/{cid}/{stl_name}"
            _copy(cache / "gt_parts" / stl_name, DEMO / rel)
            part_paths.append(rel)
            annotations.append({
                "part_id": gp.get("part_id"),
                "role_name": gp.get("role_name", ""),
                "instance_count": gp.get("instance_count", 1),
                "semantic": (gp.get("semantic") or "")[:240],
                "mesh_path": rel,
            })

        renders = []
        for v in range(4):
            rel = f"targets/renders/{cid}/view_{v:03d}.png"
            _copy_render(cache / f"renders/blender_clay/multiview/view_{v:03d}.png", DEMO / rel)
            renders.append(rel)

        rows.append({
            "id": cid, "task": "assembly-3d", "split": "demo",
            "input": {"text": condition.strip(),
                      "image_paths": [f"inputs/{cid}/view_000.png"],
                      "part_annotations": annotations},
            "target": {"format": "step", "code_path": None,
                       "step_path": f"targets/step/{cid}.step",
                       "mesh_path": f"targets/mesh/{cid}.stl",
                       "render_paths": renders, "part_paths": part_paths, "qa_bank_path": None},
            "metadata": {"source": "fusion360-gallery", "source_id": uid,
                         "license_group": "fusion360-gallery",
                         "assembly_class": decision.get("semantic_category"),
                         "n_parts": len(part_paths),
                         "difficulty_raw": decision.get("complexity"),
                         "difficulty": _fusion_difficulty(decision.get("complexity"))},
        })
    return rows


def build_text() -> list[dict]:
    rows = []
    for i, case in enumerate(TEXT_CASES):
        bucket, fid = case.split("/")
        cid = f"p3d_text-to-3d_{i:06d}"
        cache = TEXT2CAD / "_shared_cache" / f"{bucket}__{fid}"
        condition = (cache / "condition.txt").read_text(encoding="utf-8")

        _copy(TEXT2CAD / "minimal_json" / bucket / fid / "minimal_json" / f"{fid}.json",
              DEMO / "targets/minimal-json" / f"{cid}.json")
        _copy(cache / "gt_model.step", DEMO / "targets/step" / f"{cid}.step")
        _copy(cache / "gt_model.stl", DEMO / "targets/mesh" / f"{cid}.stl")
        renders = []
        occ = cache / "renders/occ/single_view/gt_render.png"
        if occ.exists():
            _copy_render(occ, DEMO / "inputs" / cid / "view_000.png")  # QA answerer render
            renders.append(f"targets/renders/{cid}/view_000.png")
            _copy_render(occ, DEMO / f"targets/renders/{cid}/view_000.png")
        qa_rel = None
        qa_src = TEXT2CAD / "qa_bank" / bucket / fid / "qa_bank.json"
        if qa_src.exists():
            qa_rel = f"targets/qa/{cid}.json"
            _copy(qa_src, DEMO / qa_rel)

        rows.append({
            "id": cid, "task": "text-to-3d", "split": "demo",
            "input": {"text": condition.strip(), "image_paths": [], "part_annotations": []},
            "target": {"format": "minimal-json",
                       "code_path": f"targets/minimal-json/{cid}.json",
                       "step_path": f"targets/step/{cid}.step",
                       "mesh_path": f"targets/mesh/{cid}.stl",
                       "render_paths": renders, "part_paths": [], "qa_bank_path": qa_rel},
            "metadata": {"source": "text2cad-v1.1", "source_id": case,
                         "license_group": "cc-by-nc-sa-4.0"},
        })
    return rows


def _fusion_difficulty(complexity) -> str:
    if complexity is None:
        return "unknown"
    if complexity <= 3:
        return "easy"
    if complexity <= 5:
        return "medium"
    return "hard"


def _write_manifest(name: str, rows: list[dict]) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS / name
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path}  ({len(rows)} cases)")


def main() -> None:
    if DEMO.exists():
        shutil.rmtree(DEMO)
    print("Building demo split ...")
    _write_manifest("text_to_3d_demo.jsonl", build_text())
    _write_manifest("image_to_3d_demo.jsonl", build_image())
    _write_manifest("assembly_3d_demo.jsonl", build_assembly())
    total = sum(1 for _ in DEMO.rglob("*") if _.is_file())
    print(f"Done. {total} files under {DEMO.relative_to(REPO)}")


if __name__ == "__main__":
    main()
