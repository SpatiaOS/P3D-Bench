"""Materialize the full P3D-Bench split from HuggingFace + a local source root.

HuggingFace (``SpatiaOS/P3D-Bench``) publishes the *redistributable* assets: the
final benchmark UID lists, the P3D-derived text/assembly annotations, and — for
Text-to-3D — the GT CAD **programs** (Text2CAD-derived minimal-JSON, shipped
because Text2CAD is CC BY-NC-SA 4.0). It deliberately does **not** redistribute
the Fusion 360 Gallery raw geometry (STEP/renders/meshes), whose license forbids
it. This builder pulls the Hub assets and reads any local heavy geometry from a
``--source-root`` (the Fusion 360 + Text2CAD working trees), then writes an
evaluator-ready ``data/full/`` tree plus ``data/manifests/*_full.jsonl`` whose
layout matches the in-repo demo split, so ``--split full`` "just works".

Per task:
  * image-to-3d / assembly-3d : copy GT STEP/STL/renders/parts straight out of
    ``fusion360/assembly/_shared_cache/<uid>/`` (mirrors ``build_demo_data.py``);
    requires the Fusion 360 geometry at ``--source-root``.
  * text-to-3d : take the GT minimal-JSON program (from the local source-root if
    present, else the Hub-shipped program — so no local Text2CAD tree is needed),
    the input text from the Hub annotation (``text_param`` / ``text_desc``), and
    *generate* the GT STEP + STL from the minimal-JSON via the same interpreter
    used to compile predictions (cached STEP/STL are not shipped).

The build is idempotent: a case whose target files already exist is skipped
unless ``overwrite=True``; a UID whose upstream assets are missing is skipped
with a recorded reason rather than aborting the whole run.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from ..protocol import (
    PAPER_CANONICAL_VIEW_COUNT,
    PAPER_QA_BANK_VERSION,
    PAPER_RENDER_RESOLUTION,
    PAPER_RENDER_SAMPLES,
    PAPER_RENDER_SEED,
    file_fingerprint,
    normalize_frozen_qa_questions,
    paper_dataset_contract,
    paper_manifest_provenance,
    paper_qa_source_contract,
    paper_task_contract,
    qa_dataset_content_sha256,
    qa_questions_content_sha256,
    validate_paper_source_ids,
)

logger = logging.getLogger(__name__)

HF_REPO_ID = "SpatiaOS/P3D-Bench"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"

# Where the upstream raw working trees (Fusion 360 + Text2CAD) live locally. The
# Hub never ships these (licensing), so the user provides them via ``--source-root``
# or the ``P3DBENCH_SOURCE_ROOT`` env var; this default is just a neutral relative
# placeholder so no machine-specific path is baked into the repo.
DEFAULT_SOURCE_ROOT = Path(os.environ.get("P3DBENCH_SOURCE_ROOT", "cad_dataset"))

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

    return hf_hub_download(
        HF_REPO_ID,
        rel_path,
        repo_type="dataset",
        revision=paper_dataset_contract()["dataset_revision"],
        token=token,
    )


def load_hf_uids(task: str, token: Optional[str] = None) -> list[str]:
    path = _hf_download(f"data/{_HF_DIR[task]}/uids.jsonl", token)
    fingerprint = file_fingerprint(str(path))
    task_contract = paper_task_contract(task)
    if fingerprint["sha256"] != task_contract["uids_sha256"]:
        raise ValueError(
            f"Pinned {task} UID source SHA mismatch: "
            f"expected {task_contract['uids_sha256']}, "
            f"got {fingerprint['sha256']}"
        )
    uids = [
        json.loads(line)["uid"]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_paper_source_ids(task, uids)
    return uids


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


def load_hf_qa(token: Optional[str] = None) -> dict[str, list]:
    """Return ``{uid: [question, ...]}`` from the Hub Text-to-3D QA banks.

    ``data/text_to_3d/qa.jsonl`` packs every ``text_mode``×``format`` variant of
    a case into one ``questions`` list (each carries ``text_mode``/``format``/
    ``qid``/``question``/``options``/``answer``); the eval-time judge selects the
    subset for the active run. ``{}`` when the file is absent (older Hub revision).
    """
    try:
        path = _hf_download("data/text_to_3d/qa.jsonl", token)
    except Exception as exc:
        logger.info("no Hub QA banks (%s); falling back to source-root qa_bank/", exc)
        return {}
    fingerprint = file_fingerprint(str(path))
    qa_contract = paper_dataset_contract()["qa"]
    if fingerprint["sha256"] != qa_contract["source_sha256"]:
        raise ValueError(
            "Pinned QA source SHA mismatch: "
            f"expected {qa_contract['source_sha256']}, "
            f"got {fingerprint['sha256']}"
        )
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
    validate_paper_source_ids(
        "text-to-3d",
        [str(row.get("uid") or "") for row in rows],
    )
    question_count = sum(len(row.get("questions") or []) for row in rows)
    if question_count != int(qa_contract["question_count"]):
        raise ValueError(
            "Pinned QA question count mismatch: "
            f"expected {qa_contract['question_count']}, got {question_count}"
        )
    content_sha = qa_dataset_content_sha256(rows)
    if content_sha != qa_contract["normalized_content_sha256"]:
        raise ValueError(
            "Pinned normalized QA dataset digest mismatch: "
            f"expected {qa_contract['normalized_content_sha256']}, "
            f"got {content_sha}"
        )
    out: dict[str, list] = {}
    for row in rows:
        if row.get("uid") and row.get("questions"):
            out[str(row["uid"])] = list(row["questions"])
    return out


def load_hf_minimal_json(token: Optional[str] = None) -> dict[str, str]:
    """Return ``{uid: minimal_json_program}`` from the Hub Text-to-3D GT programs.

    ``data/text_to_3d/minimal_json.jsonl`` ships the redistributable Text2CAD-derived
    GT CAD program for every Text-to-3D case (Text2CAD is CC BY-NC-SA 4.0), one row
    ``{"uid", "minimal_json"}`` where ``minimal_json`` is the program serialized as a
    JSON string. With it, the Text-to-3D split materializes with **no local Text2CAD
    tree**. ``{}`` when the file is absent (older Hub revision), in which case
    ``build_text`` falls back to the local ``--source-root`` minimal_json.
    """
    try:
        path = _hf_download("data/text_to_3d/minimal_json.jsonl", token)
    except Exception as exc:
        logger.info("no Hub minimal_json (%s); falling back to source-root minimal_json/", exc)
        return {}
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("uid") and row.get("minimal_json"):
            out[row["uid"]] = row["minimal_json"]
    return out


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


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def _materialize_text_gt_renders(
    mesh_path: Path,
    case_id: str,
    *,
    overwrite: bool,
) -> tuple[list[str], Optional[dict], Optional[str]]:
    """Materialize the frozen four-view Blender-clay GT evidence."""
    relative_paths = [
        f"targets/renders/{case_id}/view_{index:03d}.png"
        for index in range(PAPER_CANONICAL_VIEW_COUNT)
    ]
    output_paths = [FULL_ROOT / relative for relative in relative_paths]
    contract_path = (
        FULL_ROOT / "targets" / "renders" / case_id / "render_contract.json"
    )
    mesh_sha = file_fingerprint(str(mesh_path))["sha256"]
    existing = _read_json(contract_path)
    fixed_profile = {
        "renderer": "blender_clay",
        "view_count": PAPER_CANONICAL_VIEW_COUNT,
        "resolution": PAPER_RENDER_RESOLUTION,
        "samples": PAPER_RENDER_SAMPLES,
        "seed": PAPER_RENDER_SEED,
        "source_mesh_sha256": mesh_sha,
    }
    if not overwrite and existing:
        expected_files = existing.get("views") or []
        if (
            all(existing.get(key) == value for key, value in fixed_profile.items())
            and len(expected_files) == PAPER_CANONICAL_VIEW_COUNT
            and all(path.is_file() for path in output_paths)
            and all(
                _image_size(path) == (
                    PAPER_RENDER_RESOLUTION,
                    PAPER_RENDER_RESOLUTION,
                )
                for path in output_paths
            )
            and all(
                expected_files[index].get("sha256")
                == file_fingerprint(str(path))["sha256"]
                for index, path in enumerate(output_paths)
            )
        ):
            return relative_paths, existing, None

    from ..render import blender

    with tempfile.TemporaryDirectory(prefix="p3d_text_gt_render_") as tmp:
        rendered = blender.render_multiview(
            str(mesh_path),
            tmp,
            n_views=PAPER_CANONICAL_VIEW_COUNT,
            resolution=PAPER_RENDER_RESOLUTION,
            samples=PAPER_RENDER_SAMPLES,
            seed=PAPER_RENDER_SEED,
        )
        rendered_paths = [Path(path) for path in rendered or []]
        if (
            len(rendered_paths) != PAPER_CANONICAL_VIEW_COUNT
            or not all(path.is_file() for path in rendered_paths)
        ):
            return [], None, (
                "GT Blender-clay render gap: expected exactly "
                f"{PAPER_CANONICAL_VIEW_COUNT} views"
            )
        if any(
            _image_size(path) != (
                PAPER_RENDER_RESOLUTION,
                PAPER_RENDER_RESOLUTION,
            )
            for path in rendered_paths
        ):
            return [], None, "GT Blender-clay render gap: wrong resolution"
        for source, destination in zip(rendered_paths, output_paths):
            _copy(source, destination)

    render_contract = {
        **fixed_profile,
        "views": [
            {
                "path": relative,
                "sha256": file_fingerprint(str(path))["sha256"],
            }
            for relative, path in zip(relative_paths, output_paths)
        ],
    }
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(render_contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return relative_paths, render_contract, None


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
                "difficulty": _fusion_difficulty(decision.get("complexity")),
                "paper_dataset_contract": paper_manifest_provenance(
                    "image-to-3d"
                )}
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

        a = annotations.get(uid, {})
        # Per-part role/semantic: prefer the cache manifest (research caches carry
        # them; PREPARE-built caches embed the HF text), fall back to the HF
        # part_level_annotations joined by part_id. HF ships `description_short`,
        # not `semantic`. This never overrides a present manifest value, so the
        # prebuilt-research-cache path is unchanged.
        hf_parts = {p.get("part_id"): p for p in (a.get("part_level_annotations") or [])}
        part_paths, anns = [], []
        for gp in gt_parts:
            stl_name = Path(gp["stl_path"]).name
            rel = f"targets/parts/{cid}/{stl_name}"
            _copy(cache / "gt_parts" / stl_name, FULL_ROOT / rel)
            part_paths.append(rel)
            hf = hf_parts.get(gp.get("part_id"), {})
            anns.append({
                "part_id": gp.get("part_id"),
                "role_name": gp.get("role_name") or hf.get("role_name", ""),
                "instance_count": gp.get("instance_count") or hf.get("instance_count", 1),
                "semantic": (gp.get("semantic") or hf.get("description_short") or "")[:240],
                "mesh_path": rel,
            })

        renders = [f"targets/renders/{cid}/view_{v:03d}.png" for v in range(4)]
        _copy_image(cache / "renders/occ/single_view/gt_render.png",
                    FULL_ROOT / "inputs" / cid / "view_000.png", max_edge)
        _copy(step_src, FULL_ROOT / "targets/step" / f"{cid}.step")
        _copy(cache / "gt_model.stl", FULL_ROOT / "targets/mesh" / f"{cid}.stl")
        for v, rel in enumerate(renders):
            _copy_image(mv[v], FULL_ROOT / rel, max_edge)

        # `_filter/decision.json` is research-only (review MLLM) and absent on the
        # from-raw PREPARE path; difficulty then degrades to "unknown" and
        # assembly_class falls back to the HF annotation.
        decision = _read_json(cache / "_filter" / "decision.json") or {}
        meta = {"source": "fusion360-gallery", "source_id": uid, "license_group": "fusion360-gallery",
                "assembly_class": a.get("assembly_class") or decision.get("semantic_category"),
                "n_parts": len(part_paths), "instance_count": a.get("instance_count"),
                "unique_part_count": a.get("unique_part_count"),
                "difficulty_raw": decision.get("complexity"),
                "difficulty": _fusion_difficulty(decision.get("complexity")),
                "paper_dataset_contract": paper_manifest_provenance(
                    "assembly-3d"
                )}
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


def build_text(uids, source_root, annotations, *, max_edge, overwrite, limit, qa_map=None, mj_map=None):
    t2c = source_root / "text2cad"
    qa_map = qa_map or {}
    mj_map = mj_map or {}
    rows, skipped = [], []
    for i, uid in enumerate(uids):
        if limit and len(rows) >= limit:
            break
        bucket, fid = uid.split("/")
        cid = f"p3d_text-to-3d_{i:06d}"
        mj_src = t2c / "minimal_json" / bucket / fid / "minimal_json" / f"{fid}.json"
        # GT program: prefer the local source-root file; fall back to the
        # redistributable Hub-shipped program (Text2CAD is CC BY-NC-SA 4.0) so the
        # Text-to-3D split builds with no local Text2CAD tree.
        if mj_src.exists():
            code_text = mj_src.read_text(encoding="utf-8")
        elif uid in mj_map:
            code_text = mj_map[uid]
        else:
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
        code_dst = FULL_ROOT / code_rel
        code_dst.parent.mkdir(parents=True, exist_ok=True)
        code_dst.write_text(code_text, encoding="utf-8")

        # Generate GT STEP + STL from the minimal-JSON (cached copies are not shipped).
        step_dst, mesh_dst = FULL_ROOT / step_rel, FULL_ROOT / mesh_rel
        if not _done([step_dst, mesh_dst], overwrite):
            ok = _gen_step_stl_from_minimal_json(code_text, step_dst, mesh_dst)
            if not ok:
                skipped.append((uid, "GT minimal-json failed to compile"))
                continue

        # TextDesc J-Sem requires four same-view GT Blender-clay renders.
        try:
            renders, gt_render_contract, render_error = (
                _materialize_text_gt_renders(
                    mesh_dst,
                    cid,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:
            renders, gt_render_contract = [], None
            render_error = (
                "GT Blender-clay render gap: "
                f"{type(exc).__name__}: {exc}"
            )
        if render_error:
            skipped.append((uid, render_error))

        # QA bank.
        qa_rel = None
        # QA bank (Text-to-3D Judge): prefer the Hub qa.jsonl (all text_mode×
        # format variants), fall back to a local prebuilt bank under the source
        # root. The from-raw PREPARE path has no local qa_bank/, so the Hub is
        # the only QA source there.
        hf_q = qa_map.get(uid)
        if hf_q:
            qa_rel = f"targets/qa/{cid}.json"
            _write_qa_bank(uid, hf_q, FULL_ROOT / qa_rel, overwrite)
        else:
            qa_src = t2c / "qa_bank" / bucket / fid / "qa_bank.json"
            if qa_src.exists():
                qa_rel = f"targets/qa/{cid}.json"
                _copy(qa_src, FULL_ROOT / qa_rel)

        meta = {"source": "text2cad-v1.1", "source_id": uid, "license_group": "cc-by-nc-sa-4.0",
                "summary": ann.get("summary"), "text_desc": (ann.get("text_desc") or "").strip() or None,
                "paper_gt_render_contract": gt_render_contract,
                "paper_gt_render_error": render_error,
                "paper_dataset_contract": paper_manifest_provenance(
                    "text-to-3d"
                )}
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


def _write_qa_bank(uid: str, questions: list, dst: Path, overwrite: bool) -> None:
    """Write a frozen Hub QA bank with explicit split/source provenance.

    The source questions carry ``text_mode``/``format`` but omit ``split`` and
    ``source_text_level``. Those are deterministic packaging fields:
    semantic -> ``detailed`` and param -> ``parametric_detail``.
    """
    norm = normalize_frozen_qa_questions(questions)
    source_contract = paper_qa_source_contract()
    payload = {
        "uid": uid,
        "qa_bank_version": PAPER_QA_BANK_VERSION,
        "source_contract": source_contract,
        "dataset_content_sha256": source_contract[
            "normalized_content_sha256"
        ],
        "content_sha256": qa_questions_content_sha256(norm),
        "questions": norm,
    }
    if not overwrite and dst.exists() and _read_json(dst) == payload:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # The Hub artifact is the frozen v9 bank used by the paper.  This builder
    # only normalizes packaging fields; it never generates or rewrites questions.
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")


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
        extra = ({"qa_map": load_hf_qa(token), "mj_map": load_hf_minimal_json(token)}
                 if task == "text-to-3d" else {})
        rows, skipped = _BUILDERS[task](
            uids, source_root, anns, max_edge=max_edge, overwrite=overwrite, limit=limit, **extra
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
