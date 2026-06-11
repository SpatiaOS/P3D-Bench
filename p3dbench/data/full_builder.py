"""Materialize the full P3D-Bench split from HuggingFace + a local source root.

HuggingFace (``SpatiaOS/P3D-Bench``) publishes only the *redistributable*
metadata: the final benchmark UID lists and the P3D-derived text/assembly
annotations. It deliberately does **not** redistribute upstream raw geometry
(Fusion 360 Gallery STEP/renders, Text2CAD minimal-JSON). This builder pulls
the UID lists/annotations from the Hub and reads the heavy geometry assets from
a local ``--source-root`` (the Fusion 360 + Text2CAD working trees), then writes
an evaluator-ready ``data/full/`` tree plus ``data/manifests/*_full.jsonl`` whose
layout matches the in-repo demo split, so ``--split full`` "just works".

Per task:
  * image-to-3d / assembly-3d : copy GT STEP/STL/renders/parts straight out of
    ``fusion360/assembly/_shared_cache/<uid>/`` (mirrors ``build_demo_data.py``).
  * text-to-3d : copy the GT minimal-JSON program, take the input text from the
    Hub annotation (``text_param`` / ``text_desc``), and *generate* the GT STEP +
    STL from the minimal-JSON via the same interpreter used to compile
    predictions (cached STEP/STL are not shipped for most Text2CAD cases).

The build is idempotent: a case whose target files already exist is skipped
unless ``overwrite=True``; a UID whose upstream assets are missing is skipped
with a recorded reason rather than aborting the whole run.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HF_REPO_ID = "SpatiaOS/P3D-Bench"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"

# Default local working trees the demo was built from (see build_demo_data.py).
DEFAULT_SOURCE_ROOT = Path("/mnt/CFS/yangyikang/cad_dataset")

REPO = Path(__file__).resolve().parents[2]
FULL_ROOT = REPO / "data" / "full"
MANIFEST_DIR = REPO / "data" / "manifests"

ALL_TASKS = ("text-to-3d", "image-to-3d", "assembly-3d")
_MANIFEST_TOKEN = {
    "text-to-3d": "text_to_3d",
    "image-to-3d": "image_to_3d",
    "assembly-3d": "assembly_3d",
}
_HF_DIR = {"text-to-3d": "text_to_3d", "image-to-3d": "image_to_3d", "assembly-3d": "assembly_3d"}


# --------------------------------------------------------------------------
# HuggingFace metadata
# --------------------------------------------------------------------------
def _hf_download(rel_path: str, token: Optional[str]):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO_ID, rel_path, repo_type="dataset", token=token)


def load_hf_uids(task: str, token: Optional[str] = None) -> list[str]:
    path = _hf_download(f"data/{_HF_DIR[task]}/uids.jsonl", token)
    return [json.loads(l)["uid"] for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_hf_annotations(task: str, token: Optional[str] = None) -> dict[str, dict]:
    """Return ``{uid: annotation_row}``; ``{}`` when the task has no annotations file."""
    rel = f"data/{_HF_DIR[task]}/annotations.jsonl"
    try:
        path = _hf_download(rel, token)
    except Exception as exc:  # image-to-3d ships only uids.jsonl
        logger.info("no annotations for %s (%s)", task, exc)
        return {}
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["uid"]: r for r in rows if "uid" in r}


# --------------------------------------------------------------------------
# asset copy helpers
# --------------------------------------------------------------------------
def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_image(src: Path, dst: Path, max_edge: int) -> None:
    """Copy a PNG, optionally downscaling its longest edge to ``max_edge`` (0 = keep)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not max_edge:
        shutil.copy2(src, dst)
        return
    from PIL import Image

    img = Image.open(src)
    if max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    img.save(dst)


def _fusion_difficulty(complexity) -> str:
    if complexity is None:
        return "unknown"
    if complexity <= 3:
        return "easy"
    if complexity <= 5:
        return "medium"
    return "hard"


def _done(paths: list[Path], overwrite: bool) -> bool:
    return (not overwrite) and all(p.exists() for p in paths)


# --------------------------------------------------------------------------
# per-task builders
# --------------------------------------------------------------------------
def build_image(uids, source_root, annotations, *, max_edge, overwrite, limit):
    fusion = source_root / "fusion360" / "assembly"
    rows, skipped = [], []
    for i, uid in enumerate(uids):
        if limit and len(rows) >= limit:
            break
        cid = f"p3d_image-to-3d_{i:06d}"
        cache = fusion / "_shared_cache" / uid
        step_src = fusion / "assembly" / uid / "assembly.step"
        gt_render = cache / "renders/occ/single_view/gt_render.png"
        mv = [cache / f"renders/blender_clay/multiview/view_{v:03d}.png" for v in range(4)]
        if not (step_src.exists() and gt_render.exists() and (cache / "gt_model.stl").exists()
                and all(p.exists() for p in mv)):
            skipped.append((uid, "missing upstream assets"))
            continue

        renders = [f"targets/renders/{cid}/view_{v:03d}.png" for v in range(4)]
        targets = [FULL_ROOT / "inputs" / cid / "view_000.png",
                   FULL_ROOT / "targets/step" / f"{cid}.step",
                   FULL_ROOT / "targets/mesh" / f"{cid}.stl",
                   *[FULL_ROOT / r for r in renders]]
        if not _done(targets, overwrite):
            _copy_image(gt_render, FULL_ROOT / "inputs" / cid / "view_000.png", max_edge)
            _copy(step_src, FULL_ROOT / "targets/step" / f"{cid}.step")
            _copy(cache / "gt_model.stl", FULL_ROOT / "targets/mesh" / f"{cid}.stl")
            for v, rel in enumerate(renders):
                _copy_image(mv[v], FULL_ROOT / rel, max_edge)

        decision = _read_json(cache / "_filter" / "decision.json") or {}
        meta = {"source": "fusion360-gallery", "source_id": uid,
                "license_group": "fusion360-gallery",
                "semantic_category": decision.get("semantic_category"),
                "difficulty_raw": decision.get("complexity"),
                "difficulty": _fusion_difficulty(decision.get("complexity"))}
        rows.append({
            "id": cid, "task": "image-to-3d", "split": "full",
            "input": {"text": "", "image_paths": [f"inputs/{cid}/view_000.png"], "part_annotations": []},
            "target": {"format": "step", "code_path": None,
                       "step_path": f"targets/step/{cid}.step", "mesh_path": f"targets/mesh/{cid}.stl",
                       "render_paths": renders, "part_paths": [], "qa_bank_path": None},
            "metadata": meta,
        })
    return rows, skipped


def build_assembly(uids, source_root, annotations, *, max_edge, overwrite, limit):
    fusion = source_root / "fusion360" / "assembly"
    rows, skipped = [], []
    for i, uid in enumerate(uids):
        if limit and len(rows) >= limit:
            break
        cid = f"p3d_assembly-3d_{i:06d}"
        cache = fusion / "_shared_cache" / uid
        manifest = _read_json(cache / "manifest.json")
        step_src = fusion / "assembly" / uid / "assembly.step"
        cond = cache / "condition.txt"
        mv = [cache / f"renders/blender_clay/multiview/view_{v:03d}.png" for v in range(4)]
        if not (manifest and step_src.exists() and cond.exists() and (cache / "gt_model.stl").exists()
                and all(p.exists() for p in mv)):
            skipped.append((uid, "missing upstream assets"))
            continue
        gt_parts = manifest.get("gt_parts", [])
        if not all((cache / "gt_parts" / Path(gp["stl_path"]).name).exists() for gp in gt_parts):
            skipped.append((uid, "missing gt parts"))
            continue

        part_paths, anns = [], []
        for gp in gt_parts:
            stl_name = Path(gp["stl_path"]).name
            rel = f"targets/parts/{cid}/{stl_name}"
            _copy(cache / "gt_parts" / stl_name, FULL_ROOT / rel)
            part_paths.append(rel)
            anns.append({"part_id": gp.get("part_id"), "role_name": gp.get("role_name", ""),
                         "instance_count": gp.get("instance_count", 1),
                         "semantic": (gp.get("semantic") or "")[:240], "mesh_path": rel})

        renders = [f"targets/renders/{cid}/view_{v:03d}.png" for v in range(4)]
        _copy_image(cache / "renders/occ/single_view/gt_render.png",
                    FULL_ROOT / "inputs" / cid / "view_000.png", max_edge)
        _copy(step_src, FULL_ROOT / "targets/step" / f"{cid}.step")
        _copy(cache / "gt_model.stl", FULL_ROOT / "targets/mesh" / f"{cid}.stl")
        for v, rel in enumerate(renders):
            _copy_image(mv[v], FULL_ROOT / rel, max_edge)

        decision = _read_json(cache / "_filter" / "decision.json") or {}
        a = annotations.get(uid, {})
        meta = {"source": "fusion360-gallery", "source_id": uid, "license_group": "fusion360-gallery",
                "assembly_class": a.get("assembly_class") or decision.get("semantic_category"),
                "n_parts": len(part_paths), "instance_count": a.get("instance_count"),
                "unique_part_count": a.get("unique_part_count"),
                "difficulty_raw": decision.get("complexity"),
                "difficulty": _fusion_difficulty(decision.get("complexity"))}
        rows.append({
            "id": cid, "task": "assembly-3d", "split": "full",
            "input": {"text": cond.read_text(encoding="utf-8").strip(),
                      "image_paths": [f"inputs/{cid}/view_000.png"], "part_annotations": anns},
            "target": {"format": "step", "code_path": None,
                       "step_path": f"targets/step/{cid}.step", "mesh_path": f"targets/mesh/{cid}.stl",
                       "render_paths": renders, "part_paths": part_paths, "qa_bank_path": None},
            "metadata": meta,
        })
    return rows, skipped


def build_text(uids, source_root, annotations, *, max_edge, overwrite, limit):
    t2c = source_root / "text2cad"
    rows, skipped = [], []
    for i, uid in enumerate(uids):
        if limit and len(rows) >= limit:
            break
        bucket, fid = uid.split("/")
        cid = f"p3d_text-to-3d_{i:06d}"
        mj_src = t2c / "minimal_json" / bucket / fid / "minimal_json" / f"{fid}.json"
        if not mj_src.exists():
            skipped.append((uid, "missing minimal_json"))
            continue
        ann = annotations.get(uid, {})
        text_param = (ann.get("text_param") or "").strip()
        if not text_param:
            # fall back to the local parametric condition if the Hub row is absent
            cond = t2c / "_shared_cache" / f"{bucket}__{fid}" / "condition.txt"
            text_param = cond.read_text(encoding="utf-8").strip() if cond.exists() else ""
        if not text_param:
            skipped.append((uid, "no parametric text"))
            continue

        code_rel = f"targets/minimal-json/{cid}.json"
        step_rel = f"targets/step/{cid}.step"
        mesh_rel = f"targets/mesh/{cid}.stl"
        _copy(mj_src, FULL_ROOT / code_rel)

        # Generate GT STEP + STL from the minimal-JSON (cached copies are not shipped).
        step_dst, mesh_dst = FULL_ROOT / step_rel, FULL_ROOT / mesh_rel
        if not _done([step_dst, mesh_dst], overwrite):
            ok = _gen_step_stl_from_minimal_json(mj_src.read_text(encoding="utf-8"), step_dst, mesh_dst)
            if not ok:
                skipped.append((uid, "GT minimal-json failed to compile"))
                continue

        # Optional GT assets if cached locally (renders / QA bank).
        renders, qa_rel = [], None
        cache = t2c / "_shared_cache" / f"{bucket}__{fid}"
        occ = cache / "renders/occ/single_view/gt_render.png"
        if occ.exists():
            rel = f"targets/renders/{cid}/view_000.png"
            _copy_image(occ, FULL_ROOT / rel, max_edge)
            _copy_image(occ, FULL_ROOT / "inputs" / cid / "view_000.png", max_edge)
            renders.append(rel)
        qa_src = t2c / "qa_bank" / bucket / fid / "qa_bank.json"
        if qa_src.exists():
            qa_rel = f"targets/qa/{cid}.json"
            _copy(qa_src, FULL_ROOT / qa_rel)

        meta = {"source": "text2cad-v1.1", "source_id": uid, "license_group": "cc-by-nc-sa-4.0",
                "summary": ann.get("summary"), "text_desc": (ann.get("text_desc") or "").strip() or None}
        rows.append({
            "id": cid, "task": "text-to-3d", "split": "full",
            "input": {"text": text_param, "image_paths": [], "part_annotations": []},
            "target": {"format": "minimal-json", "code_path": code_rel,
                       "step_path": step_rel, "mesh_path": mesh_rel,
                       "render_paths": renders, "part_paths": [], "qa_bank_path": qa_rel},
            "metadata": meta,
        })
    return rows, skipped


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _gen_step_stl_from_minimal_json(code: str, step_dst: Path, mesh_dst: Path) -> bool:
    """Compile a GT minimal-JSON program to STEP + STL via the shared interpreter."""
    from ..compile.exporter import compile_code

    with tempfile.TemporaryDirectory() as td:
        cr = compile_code(code, "minimal-json", Path(td))
        if not cr.valid or not cr.step or not cr.stl:
            return False
        _copy(Path(cr.step), step_dst)
        _copy(Path(cr.stl), mesh_dst)
    return True


def _write_manifest(task: str, rows: list[dict], *, merge: bool = False) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{_MANIFEST_TOKEN[task]}_full.jsonl"
    if merge and path.exists():
        # Partial (``--limit``) build: keep rows for cases this run did not
        # (re)build so a quick subset run never shrinks an existing full
        # manifest. New rows replace same-id rows; the rest are preserved.
        by_id: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                by_id[r["id"]] = r
        for r in rows:
            by_id[r["id"]] = r
        rows = [by_id[k] for k in sorted(by_id)]
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
_BUILDERS = {"image-to-3d": build_image, "assembly-3d": build_assembly, "text-to-3d": build_text}


def build_full(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    tasks: tuple[str, ...] = ALL_TASKS,
    limit: Optional[int] = None,
    max_edge: int = 0,
    overwrite: bool = False,
    token: Optional[str] = None,
) -> dict:
    """Download Hub metadata, materialize ``data/full/`` + manifests, return a report."""
    source_root = Path(source_root)
    report: dict = {"tasks": {}, "source_root": str(source_root)}
    FULL_ROOT.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        uids = load_hf_uids(task, token)
        anns = load_hf_annotations(task, token)
        rows, skipped = _BUILDERS[task](
            uids, source_root, anns, max_edge=max_edge, overwrite=overwrite, limit=limit
        )
        # A limited run builds only a prefix of the UID list; merge so it never
        # truncates an already-materialized full manifest. A full run (no limit)
        # rebuilds every UID, so it overwrites and prunes stale rows.
        path = _write_manifest(task, rows, merge=limit is not None)
        report["tasks"][task] = {"requested": len(uids), "built": len(rows),
                                 "skipped": len(skipped), "skipped_detail": skipped[:10],
                                 "manifest": str(path)}
        logger.info("full/%s: built %d/%d (skipped %d)%s -> %s",
                    task, len(rows), len(uids), len(skipped),
                    " [merged into existing manifest]" if limit is not None else "", path)
    return report
