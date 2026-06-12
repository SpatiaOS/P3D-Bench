"""PREPARE stage: build the per-case ``_shared_cache`` from raw upstream.

Ports the data-processing pipeline from the research ``render/cache.py``
(``SharedDataCache``) so the cache this produces is identical in layout/contents
to the one :mod:`p3dbench.data.full_builder` already materializes from. The only
deltas vs the source pipeline:

  * **renderers**: input image = OCC single-view (:mod:`p3dbench.render.occ_single`),
    judge image = Blender clay multiview (:mod:`p3dbench.render.blender`). There is
    **no pyrender path** — both are hard requirements; a missing backend is an
    error, not a silent degrade.
  * **geometry primitives are reused**, not re-ported: STEP→STL via
    :func:`p3dbench.compile.step_mesh.export_step_to_stl` (the same per-solid
    tessellation the geometry metric uses), minimal-JSON→STEP via
    :func:`p3dbench.compile.exporter.compile_code`, normalization via
    :func:`p3dbench.metrics.geometry.normalize_mesh`.
  * per-part assembly role/semantic come from HF annotations (joined in
    :mod:`p3dbench.data.raw_loaders`), not the unreleased ``_capv2`` pipeline.

The cache root is task-specific so it lands exactly where ``full_builder`` reads:
  image-/assembly-3d : ``<source_root>/fusion360/assembly/_shared_cache/<uid>/``
  text-to-3d         : ``<source_root>/text2cad/_shared_cache/<bucket>__<id>/``

Each ``prepare_case`` writes ``manifest.json`` plus, per task:
  gt_model.stl, gt_model_normalized.stl, condition.txt (assembly/text),
  gt.json (text), gt_parts/part_NN.stl (assembly),
  renders/occ/single_view/gt_render.png, renders/blender_clay/multiview/view_00N.png.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import tempfile
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Map P3D task -> the render/asset modes its _shared_cache must carry.
#   single_view            : OCC input image (always)
#   multiview_blender_clay : 4-view judge image (image/assembly)
#   gt_parts_stl           : per-body tessellation (assembly only)
_TASK_MODES = {
    "text-to-3d": {"single_view"},
    "image-to-3d": {"single_view", "multiview_blender_clay"},
    "assembly-3d": {"single_view", "multiview_blender_clay", "gt_parts_stl"},
}


def render_modes_for_task(task: str) -> Set[str]:
    """Single source of truth for what the cache must produce for *task*."""
    return set(_TASK_MODES.get(task, {"single_view"}))


def _safe_case_id(case_id: str) -> str:
    return case_id.replace("/", "__")


class _KeyedLock:
    """Minimal per-key lock (single-process; supports an optional --jobs pool)."""

    def __init__(self):
        self._global = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def get(self, key: str) -> threading.Lock:
        with self._global:
            return self._locks.setdefault(key, threading.Lock())


class SharedDataCache:
    """Manifest-driven cache for shared GT data (port of the research pipeline)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks = _KeyedLock()

    # ---- public API -----------------------------------------------------
    def case_dir(self, case_id: str) -> Path:
        return self.root / _safe_case_id(case_id)

    def is_prepared(self, case_id: str, render_modes: Set[str] = frozenset()) -> bool:
        manifest = self._load_manifest(case_id)
        if not manifest or manifest.get("status") != "complete":
            return False
        return self._validate_manifest_files(manifest, render_modes)

    def prepare_case(self, case_id: str, case_data: dict,
                     render_modes: Set[str] = frozenset(),
                     *, overwrite: bool = False) -> Optional[dict]:
        """Thread-safe: prepare all GT data for *case_id*; return its manifest or None."""
        with self._locks.get(case_id):
            if not overwrite and self.is_prepared(case_id, render_modes):
                return self._load_manifest(case_id)
            return self._do_prepare(case_id, case_data, render_modes)

    # ---- manifest io ----------------------------------------------------
    def _manifest_path(self, case_id: str) -> Path:
        return self.case_dir(case_id) / "manifest.json"

    def _load_manifest(self, case_id: str) -> Optional[dict]:
        path = self._manifest_path(case_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_manifest(self, case_id: str, data: dict) -> None:
        path = self._manifest_path(case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _validate_manifest_files(self, manifest: dict,
                                 render_modes: Set[str] = frozenset()) -> bool:
        gt_step = manifest.get("gt_step_path")
        if not gt_step or not Path(gt_step).exists():
            return False
        gt_stl = manifest.get("gt_stl_path")
        if gt_stl and not Path(gt_stl).exists():
            return False
        renders = manifest.get("renders", {})
        if "single_view" in render_modes or not render_modes:
            occ = renders.get("occ", {}).get("single_view")
            if not (occ and Path(occ).exists()):
                return False
        for mode in render_modes:
            if not mode.startswith("multiview_"):
                continue
            backend = mode[len("multiview_"):]
            mv = renders.get(backend, {}).get("multiview", [])
            if not mv or not all(Path(p).exists() for p in mv):
                return False
        if "gt_parts_stl" in render_modes:
            if not manifest.get("gt_parts_prepared"):
                return False
            parts = manifest.get("gt_parts") or []
            if not parts:
                return False
            for entry in parts:
                stl = entry.get("stl_path")
                if stl and not Path(stl).exists():
                    return False
        return True

    # ---- the pipeline ---------------------------------------------------
    def _do_prepare(self, case_id: str, case_data: dict,
                    render_modes: Set[str]) -> Optional[dict]:
        cdir = self.case_dir(case_id)
        cdir.mkdir(parents=True, exist_ok=True)
        manifest: dict = {"status": "incomplete", "case_id": case_id, "renders": {}}

        # 1. GT STEP (direct, or generated from minimal-JSON).
        gt_step, gt_step_source = _resolve_step(case_data, cdir)
        if not gt_step:
            logger.warning("No GT STEP for %s — skipping", case_id)
            return None
        manifest["gt_step_path"] = gt_step
        manifest["gt_step_source"] = gt_step_source

        # 2. STEP -> STL (shared per-solid tessellation).
        gt_stl = _convert_to_stl(gt_step, cdir / "gt_model.stl")
        if not gt_stl:
            logger.warning("STEP->STL failed for %s — skipping", case_id)
            return None
        manifest["gt_stl_path"] = gt_stl

        # 3. Normalized STL (debug artifact).
        manifest["gt_normalized_stl_path"] = _normalize_stl(
            gt_stl, cdir / "gt_model_normalized.stl")

        # 4. Renders. OCC single-view (input) + Blender clay multiview (judge).
        renders = manifest.get("renders", {})
        if "single_view" in render_modes or not render_modes:
            renders = _render_single_view_occ(gt_step, cdir, renders)
        mv_backends = {m[len("multiview_"):] for m in render_modes
                       if m.startswith("multiview_")}
        for backend in sorted(mv_backends):
            renders = _render_multiview(backend, gt_stl, gt_step, cdir, renders)
        manifest["renders"] = renders

        # 5. Text data (condition.txt / gt.json).
        _save_text_data(case_data, cdir, manifest)

        # 6. Per-part GT STL (assembly).
        if "gt_parts_stl" in render_modes and case_data.get("gt_parts"):
            manifest["gt_parts"] = _prepare_gt_parts_stl(case_data["gt_parts"], cdir)
            manifest["gt_parts_prepared"] = True
        if case_data.get("part_match"):
            manifest["part_match"] = case_data["part_match"]

        # 7. Validate required artifacts before marking complete (hard, no degrade).
        if not self._validate_manifest_files(manifest, render_modes):
            missing = _missing_artifacts(manifest, render_modes)
            logger.warning("%s: incomplete after prepare (%s) — not marking complete",
                           case_id, missing)
            self._save_manifest(case_id, manifest)  # keep partial for debugging
            return None

        manifest["status"] = "complete"
        self._save_manifest(case_id, manifest)
        return manifest


# ======================================================================
# pipeline helpers (reuse P3D primitives; ported orchestration)
# ======================================================================
def _resolve_step(case_data: dict, cdir: Path) -> tuple[Optional[str], Optional[str]]:
    """Direct GT STEP, or generate one from minimal-JSON via the shared interpreter."""
    direct = case_data.get("gt_step")
    if direct and Path(direct).exists():
        return direct, "direct_step"

    mj = case_data.get("gt_minimal_json")
    if not mj or not Path(mj).exists():
        return None, None
    out_step = cdir / "gt_model.step"
    if out_step.exists():
        return str(out_step), "minimal_json"
    try:
        from ..compile.exporter import compile_code

        code = Path(mj).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as td:
            cr = compile_code(code, "minimal-json", Path(td))
            if not cr.valid or not cr.step:
                logger.warning("minimal-json compile produced no STEP for %s", mj)
                return None, None
            out_step.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cr.step, out_step)
        return str(out_step), "minimal_json"
    except Exception as exc:
        logger.warning("STEP export failed for %s: %s", mj, exc)
        return None, None


def _convert_to_stl(step_path: str, output_path: Path) -> Optional[str]:
    """STEP -> STL via the shared per-solid pipeline (same as the geometry metric)."""
    if output_path.exists() and output_path.stat().st_size > 0:
        return str(output_path)
    try:
        from ..compile.step_mesh import export_step_to_stl

        err, _stats = export_step_to_stl(Path(step_path), output_path)
        if err or not (output_path.exists() and output_path.stat().st_size > 0):
            logger.warning("STEP->STL conversion failed for %s: %s", step_path, err)
            return None
        return str(output_path)
    except Exception as exc:
        logger.warning("STEP->STL conversion error for %s: %s", step_path, exc)
        return None


def _normalize_stl(stl_path: str, output_path: Path) -> Optional[str]:
    if output_path.exists():
        return str(output_path)
    try:
        from ..compile.step_mesh import load_mesh
        from ..metrics.geometry import normalize_mesh

        mesh = load_mesh(stl_path)
        if mesh is None:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalize_mesh(mesh).export(str(output_path))
        return str(output_path)
    except Exception as exc:
        logger.warning("Normalize failed for %s: %s", stl_path, exc)
        return None


def _render_single_view_occ(gt_step: str, cdir: Path, renders: dict) -> dict:
    """OCC single-view = the canonical input image. Hard requirement."""
    renders = dict(renders)
    occ_out = cdir / "renders" / "occ" / "single_view" / "gt_render.png"
    if not occ_out.exists():
        try:
            from ..render.occ_single import render_occ_single

            ok = render_occ_single(gt_step, str(occ_out), elev_deg=30, azim_deg=45,
                                   timeout_sec=180)
            if not ok:
                logger.warning("OCC single_view returned false for %s", gt_step)
        except Exception as exc:
            logger.warning("OCC single_view failed for %s: %s", gt_step, exc)
    if occ_out.exists():
        renders.setdefault("occ", {})["single_view"] = str(occ_out)
    return renders


def _render_multiview(backend: str, gt_stl: Optional[str], gt_step: str,
                      cdir: Path, renders: dict) -> dict:
    """Blender clay multiview = the judge image set. Hard requirement."""
    renders = dict(renders)
    if backend != "blender_clay":
        logger.warning("unsupported multiview backend %r (only blender_clay)", backend)
        return renders
    mv_dir = cdir / "renders" / backend / "multiview"
    existing = renders.get(backend, {}).get("multiview", [])
    if existing and all(Path(p).exists() for p in existing):
        return renders
    mv_dir.mkdir(parents=True, exist_ok=True)
    src = gt_stl or gt_step
    try:
        from ..render.blender import render_multiview

        paths = render_multiview(src, str(mv_dir), n_views=4)
        if paths:
            renders.setdefault(backend, {})["multiview"] = [str(p) for p in paths]
    except Exception as exc:
        logger.warning("Blender clay multiview failed for %s: %s", src, exc)
    return renders


def _save_text_data(case_data: dict, cdir: Path, manifest: dict) -> None:
    text = case_data.get("text")
    if text:
        cond = cdir / "condition.txt"
        if not cond.exists():
            cond.write_text(text, encoding="utf-8")
        manifest["condition_text"] = text
        manifest["condition_text_path"] = str(cond)
    mj = case_data.get("gt_minimal_json")
    if mj and Path(mj).exists():
        dst = cdir / "gt.json"
        if not dst.exists():
            shutil.copy2(mj, dst)
        manifest["gt_json_path"] = str(dst)


def _prepare_gt_parts_stl(source_parts: list, cdir: Path) -> list:
    """Tessellate each per-body STEP into ``gt_parts/part_NN.stl`` (stable order)."""
    parts_dir = cdir / "gt_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for idx, part in enumerate(source_parts):
        entry = dict(part)
        src_step = part.get("step_path")
        stl_path = parts_dir / f"part_{idx:02d}.stl"
        if src_step and Path(src_step).exists():
            if not (stl_path.exists() and stl_path.stat().st_size > 0):
                if _convert_to_stl(src_step, stl_path) is None:
                    logger.warning("Failed to tessellate GT part %d (%s)",
                                   idx, part.get("part_id") or src_step)
            if stl_path.exists() and stl_path.stat().st_size > 0:
                entry["stl_path"] = str(stl_path)
        out.append(entry)
    return out


def _missing_artifacts(manifest: dict, render_modes: Set[str]) -> str:
    bits = []
    if not manifest.get("gt_stl_path"):
        bits.append("gt_stl")
    renders = manifest.get("renders", {})
    if ("single_view" in render_modes) and not renders.get("occ", {}).get("single_view"):
        bits.append("occ_single_view")
    for mode in render_modes:
        if mode.startswith("multiview_") and not renders.get(
                mode[len("multiview_"):], {}).get("multiview"):
            bits.append(mode)
    if "gt_parts_stl" in render_modes and not manifest.get("gt_parts_prepared"):
        bits.append("gt_parts")
    return ", ".join(bits) or "unknown"


# ======================================================================
# orchestration over a UID list (called by the CLI prepare stage)
# ======================================================================
def cache_root_for_task(task: str, source_root: Path) -> Path:
    """Where the _shared_cache lives, matching full_builder's read paths."""
    source_root = Path(source_root)
    if task == "text-to-3d":
        return source_root / "text2cad" / "_shared_cache"
    return source_root / "fusion360" / "assembly" / "_shared_cache"


def build_case_data(task: str, uid: str, source_root: Path,
                    hf_anno: Optional[dict]) -> Optional[dict]:
    from .raw_loaders import (build_assembly_case_data, build_image_case_data,
                              build_text_case_data)
    if task == "image-to-3d":
        return build_image_case_data(uid, source_root)
    if task == "assembly-3d":
        return build_assembly_case_data(uid, source_root, hf_anno)
    if task == "text-to-3d":
        return build_text_case_data(uid, source_root, hf_anno)
    raise ValueError(f"unknown task {task!r}")


def prepare_task(task: str, uids: list, source_root: Path,
                 annotations: dict, *, overwrite: bool = False,
                 limit: Optional[int] = None) -> dict:
    """Build _shared_cache for one task over its UID list. Returns a report."""
    modes = render_modes_for_task(task)
    cache = SharedDataCache(cache_root_for_task(task, source_root))
    built, skipped = [], []
    for uid in uids:
        if limit and len(built) >= limit:
            break
        case_data = build_case_data(task, uid, source_root, annotations.get(uid))
        if case_data is None:
            skipped.append((uid, "missing raw assets"))
            continue
        manifest = cache.prepare_case(case_data["case_id"], case_data, modes,
                                      overwrite=overwrite)
        if manifest is None:
            skipped.append((uid, "prepare incomplete (render/tessellation failed)"))
        else:
            built.append(uid)
    return {"task": task, "built": len(built), "skipped": len(skipped),
            "skipped_detail": skipped[:10], "cache_root": str(cache.root)}
