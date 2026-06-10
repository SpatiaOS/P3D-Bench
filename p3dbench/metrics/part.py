"""Part bucket — per-part evaluation for the Assembly-3D task.

Per-part evaluation for Assembly-3D (PartFS, PartMatchF1). Given a list of
GT part meshes and a list of predicted (decomposed) part meshes for one case, it:

1. (Optionally) drops the case under a *fidelity gate* when the decomposition step
   redesigned the geometry (CD > 5e-4 AND IoU-V < 0.95) — all numerics become None
   so the case leaves aggregate means instead of penalising the model.
2. Dedups both sides by a rotation/translation-invariant geometric fingerprint
   (collapses N repeated instances of one body into 1 representative).
3. Maps every part of a side through ONE shared transform (GT: its own union-bbox
   unit-cube normalize; pred: ``align_transform_4x4`` from the geometry bucket) so
   relative inter-part sizes survive.
4. For every (gt_part, pred_part) tries 24 proper cube rotations + centroid
   translation, keeps the rotation minimising bidirectional Chamfer Distance, and
   reports (cd, f_score, precision, recall) at that one alignment.
5. Hungarian one-to-one matching on cost ``1 - f_score`` (scipy).
6. **PartMatchF1** (count-level, acceptance F-Score@tau >= 0.7) and **PartFS**
   (quality-level, unfiltered Hungarian mean F-Score).

The bucket's ``score`` ALSO runs the single-call stage-2 decomposition itself (no
retry): it builds the decompose prompt, calls ``ctx.decompose_client``, compiles
the parts-structured program, computes stage-2 fidelity, and feeds the result to
:func:`evaluate_assembly_parts`.

Heavy dependencies (numpy, scipy, trimesh) are imported lazily via
:func:`p3dbench.utils.require` so importing this module never requires the
``geometry`` extra to be installed; a bucket whose dependency is absent lets the
``MissingDependencyError`` propagate.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils import require
from .base import MetricBucket, ScoreContext

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Lazy heavy-dependency accessors (so module import never needs the extra)
# --------------------------------------------------------------------------
def _np():
    return require("numpy", "geometry", "Part metrics")


def _trimesh():
    return require("trimesh", "geometry", "Part metrics")


def _ckdtree():
    return require("scipy", "geometry", "Part metrics").spatial.cKDTree


def _linear_sum_assignment():
    return require("scipy", "geometry", "Part metrics").optimize.linear_sum_assignment


def _load_mesh(path):
    """Shared mesh loader (STEP via OCP tessellation; STL/OBJ via trimesh)."""
    from ..compile.step_mesh import load_mesh

    return load_mesh(path)


def _sample_points(mesh, n_pts: int):
    """Deterministic surface sampling (seed=42), shared with the geometry bucket."""
    from .geometry import _sample_points as _gs

    return _gs(mesh, n_pts)


# --------------------------------------------------------------------------
# Constants (thresholds — verbatim from the source)
# --------------------------------------------------------------------------
# F-Score acceptance: a pair counts as matched iff its bidirectional coverage F1
# at tau = F_SCORE_TAU_FRAC * diag_gt is at least F_SCORE_MIN. tau scales with
# each GT part's own bbox diagonal in the shared GT-normalised frame, so the
# criterion is dimensionless and unbiased across part sizes (Tatarchenko et al.
# CVPR'19 convention). F1 = 0.7 is the boundary "shape correct AND size within
# ~10-12%". The historical fixed CD cap (CD_HARD_MAX = 0.01) was replaced by this
# F-Score@tau because CD scales with extent^2 (too lenient on small parts, too
# strict on large ones) — do not reintroduce it.
F_SCORE_TAU_FRAC = 0.05
F_SCORE_MIN = 0.7

# Significant-figures quantization for the geometric fingerprint used in
# pred/GT dedup. Asymmetric on purpose: GT STLs share a single per-body STEP
# tessellation (byte-identical repeated instances → tight sig=5), while pred
# STLs are independent tessellations of separate module invocations and carry
# ~3e-4 relative noise on V/A/inertia across rotated copies → loose sig=3.
GT_DEDUP_SIG = 5
PRED_DEDUP_SIG = 3

# Surface samples per part for the pair CD/F-Score. 1024 is enough for
# rank-stable matching at part-level scale.
N_SAMPLE_POINTS = 1024

# Fidelity gate defaults (the gate only fires on a well-formed dict).
FIDELITY_CD_MAX = 5e-4
FIDELITY_IOU_MIN = 0.95


# --------------------------------------------------------------------------
# 24 proper rotations of the cube (det = +1, no mirrors).
# --------------------------------------------------------------------------
_PROPER_ROTATIONS_24 = None  # lazy init (numpy not always available at import)


def _build_24_proper_rotations():
    """6 axis permutations x 8 sign patterns = 48 signed permutation matrices;
    half have det(R) = +1 (proper rotations), half det(R) = -1 (reflections).
    Keep only the +1 group -> 24 matrices. Chirality is intentionally penalised."""
    np = _np()
    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            R = np.zeros((3, 3), dtype=np.float64)
            for row, col in enumerate(perm):
                R[row, col] = signs[row]
            if np.linalg.det(R) > 0.5:
                rots.append(R)
    return np.stack(rots)


def _get_24_rotations():
    global _PROPER_ROTATIONS_24
    if _PROPER_ROTATIONS_24 is None:
        _PROPER_ROTATIONS_24 = _build_24_proper_rotations()
    return _PROPER_ROTATIONS_24


# --------------------------------------------------------------------------
# Mesh helpers
# --------------------------------------------------------------------------
def load_pred_parts_from_dir(case_dir: Path) -> List[Dict[str, Any]]:
    """Read ``parts_meta.json`` produced by the multipart exporter."""
    import json

    meta_path = Path(case_dir) / "parts_meta.json"
    if not meta_path.exists():
        return []
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not parse {meta_path}: {exc}")
        return []
    if not isinstance(data, list):
        return []
    return data


def _build_normalize_4x4(union_mesh_path: str):
    """Build a 4x4 transform mapping ORIGINAL coordinates of ``union_mesh`` into
    its unit-cube-normalised frame: ``v' = (v - center) / scale`` where ``center``
    is the union bbox center and ``scale`` the union bbox max extent. The SAME
    (center, scale) is then applied to every part of that assembly, so relative
    sizes between parts are preserved (normalization uses the *union* bbox, never
    per-part bboxes)."""
    try:
        mesh = _load_mesh(union_mesh_path)
    except Exception as exc:
        logger.warning(f"_build_normalize_4x4 load failed for {union_mesh_path}: {exc}")
        return None
    if mesh is None:
        return None
    np = _np()
    bbox_min, bbox_max = mesh.bounds
    center = (bbox_min + bbox_max) / 2.0
    extent = float(np.max(bbox_max - bbox_min))
    if extent <= 0:
        extent = 1.0
    norm_mat = np.eye(4)
    norm_mat[:3, :3] = np.eye(3) / extent
    norm_mat[:3, 3] = -center / extent
    return norm_mat


def _sample_part_with_transform(
    mesh_path: str,
    n_pts: int,
    transform_4x4,
    cache: Optional[Dict[str, Any]] = None,
):
    """Load mesh in ORIGINAL coordinates, apply a single 4x4 transform that puts
    it in the desired target frame, sample N surface points. Caller composes the
    transform (normalize, normalize + global align, ...); this helper does not
    implicitly normalize.

    NOTE: the cache key is the path only (not the transform). Safe here because
    each path is only ever sampled under one transform per call (GT parts under
    T_gt, pred parts under T_pred); flag if the cache is ever shared across
    transforms."""
    cache_key = mesh_path
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    try:
        mesh = _load_mesh(mesh_path)
    except Exception as exc:
        logger.warning(f"_sample_part_with_transform load failed for {mesh_path}: {exc}")
        if cache is not None:
            cache[cache_key] = None
        return None
    if mesh is None:
        if cache is not None:
            cache[cache_key] = None
        return None

    try:
        mesh = mesh.copy()
        if transform_4x4 is not None:
            mesh.apply_transform(transform_4x4)
        pts = _sample_points(mesh, n_pts)
    except Exception as exc:
        logger.warning(f"_sample_part_with_transform sample failed for {mesh_path}: {exc}")
        if cache is not None:
            cache[cache_key] = None
        return None

    if cache is not None:
        cache[cache_key] = pts
    return pts


# --------------------------------------------------------------------------
# Symmetric GT/pred dedup by rotation/translation-invariant geometric fingerprint
# --------------------------------------------------------------------------
#
# A single body can appear N times in an assembly at different occurrence
# transforms. The 24-rot + centroid-translate pair metric already discards
# placement, so those N copies all score identically against the same
# counterpart — pre-collapsing avoids paying N x pair-metric cost and, more
# importantly, keeps the per-part mean from being silently weighted by instance
# multiplicity.
#
# Fingerprint = (V, A, principal_inertia_components_sorted), each rounded to
# ``sig`` significant figures. All three are rotation+translation invariant at
# the mesh level (AABB/OBB extents were tried and rejected — rotation-dependent /
# tessellation-unstable). Volume falls back to convex_hull.volume when the mesh
# is not watertight. Known limitation: mirror-twin parts collapse (V/A/eigvals
# are reflection-invariant).

_PART_PATH_KEYS = ("stl", "step", "stl_path", "step_path")
_PART_ALIAS_KEYS = ("semantic", "role_name", "geometry_class")


def _resolve_part_path(part: Dict[str, Any]) -> Optional[str]:
    """Pick the first non-empty path key. Pred entries use ``stl``/``step``, GT
    entries use ``stl_path``/``step_path`` — handle both."""
    for k in _PART_PATH_KEYS:
        v = part.get(k)
        if v:
            return str(v)
    return None


def _collect_aliases(part: Dict[str, Any]) -> List[str]:
    out = []
    for k in _PART_ALIAS_KEYS:
        v = part.get(k)
        if v:
            out.append(str(v))
    return out


def _round_sig(x: float, sig: int = 3) -> float:
    """Round ``x`` to ``sig`` significant figures (passthrough for 0/non-finite)."""
    import math

    if x == 0.0 or not math.isfinite(x):
        return float(x)
    d = sig - int(math.floor(math.log10(abs(x)))) - 1
    return round(x, d)


def _geom_fingerprint(stl_path: str, sig: int = 3):
    """Compute a rotation+translation-invariant fingerprint of a mesh.

    Fingerprint = ``(round_sig(V), round_sig(A), tuple(round_sig(sorted eig)))``.
    ``V`` = mesh.volume if watertight else convex_hull.volume (with a second
    convex-hull fallback); ``A`` = mesh.area; eigenvalues from trimesh
    ``principal_inertia_components`` (sorted). Returns ``None`` on any failure
    (caller keeps the part as a singleton — conservative, never falsely merges)."""
    try:
        _trimesh()
    except Exception as exc:
        logger.warning(f"_geom_fingerprint: trimesh import failed: {exc}")
        return None

    try:
        mesh = _load_mesh(stl_path)
    except Exception as exc:
        logger.warning(f"_geom_fingerprint load failed for {stl_path}: {exc}")
        return None
    if mesh is None or mesh.is_empty:
        return None

    try:
        if mesh.is_watertight:
            V = float(mesh.volume)
        else:
            V = float(mesh.convex_hull.volume)
    except Exception:
        try:
            V = float(mesh.convex_hull.volume)
        except Exception as exc:
            logger.warning(f"_geom_fingerprint volume failed for {stl_path}: {exc}")
            return None

    try:
        A = float(mesh.area)
        eig = sorted(float(e) for e in mesh.principal_inertia_components)
    except Exception as exc:
        logger.warning(f"_geom_fingerprint area/inertia failed for {stl_path}: {exc}")
        return None

    return (
        _round_sig(V, sig),
        _round_sig(A, sig),
        tuple(_round_sig(e, sig) for e in eig),
    )


def _dedupe_parts_by_fingerprint(
    parts: List[Dict[str, Any]],
    sig: int = 3,
) -> Dict[str, Any]:
    """Group parts by geometric fingerprint; one representative per group.

    Generic over GT and pred (uses ``_resolve_part_path`` and ``_collect_aliases``).
    ``sig`` is exposed because GT and pred carry different noise profiles (see the
    module constants). Returns ``{"unique", "groups", "input_count",
    "unique_count", "unfingerprinted"}``. Each representative is a shallow copy of
    the first member with ``instance_count`` and sorted-unique ``aliases`` stamped
    on. Parts that fail fingerprinting are kept as singletons (conservative)."""
    groups: Dict[Any, List[int]] = {}
    unfp_count = 0
    for i, part in enumerate(parts):
        path = _resolve_part_path(part)
        fp = _geom_fingerprint(path, sig=sig) if path else None
        if fp is None:
            fp = ("__unfingerprinted__", i)  # unique key -> singleton
            unfp_count += 1
        groups.setdefault(fp, []).append(i)

    unique: List[Dict[str, Any]] = []
    group_summaries: List[Dict[str, Any]] = []
    for fp, member_indices in groups.items():
        rep = dict(parts[member_indices[0]])
        rep["instance_count"] = len(member_indices)
        alias_set = set()
        for k in member_indices:
            for a in _collect_aliases(parts[k]):
                alias_set.add(a)
        rep["aliases"] = sorted(alias_set)
        unique.append(rep)
        group_summaries.append({
            "fingerprint": (None if isinstance(fp, tuple)
                            and fp and fp[0] == "__unfingerprinted__"
                            else list(fp) if isinstance(fp, tuple) else fp),
            "indices": member_indices,
            "instance_count": len(member_indices),
            "aliases": rep["aliases"],
        })

    return {
        "unique": unique,
        "groups": group_summaries,
        "input_count": len(parts),
        "unique_count": len(unique),
        "unfingerprinted": unfp_count,
    }


# --------------------------------------------------------------------------
# Per-pair metrics (CD + F-Score) over 24 rotations + center translate
# --------------------------------------------------------------------------
def _pair_metrics_min_over_24rot(pts_pred, pts_gt, tau, rotations_24=None):
    """Per-pair metrics: rotate pred by all 24 proper rotations of the cube,
    translate each so its centroid coincides with GT's. At the rotation that
    MINIMIZES bidirectional Chamfer Distance, also compute coverage F-Score at
    radius ``tau``. All four returned values describe the same alignment.

    Returns ``{"cd", "f_score", "precision", "recall"}`` or ``None`` on failure.

    No per-pair scale and no PCA refinement — size is part of what's measured;
    orientation is only refinable to cube symmetry (90 deg steps), position fully
    refinable (centroid). Distances compared to ``tau`` are raw (not squared);
    only CD uses squares."""
    np = _np()
    cKDTree = _ckdtree()
    if pts_pred is None or pts_gt is None:
        return None
    if pts_pred.shape[0] == 0 or pts_gt.shape[0] == 0:
        return None
    if not (isinstance(tau, (int, float)) and tau > 0):
        return None
    if rotations_24 is None:
        rotations_24 = _get_24_rotations()

    n_rot = rotations_24.shape[0]
    n_pred = pts_pred.shape[0]
    gt_centroid = pts_gt.mean(axis=0)

    # Rotate pred by all 24 rotations: (n_rot, n_pred, 3). One matmul.
    rotated = pts_pred @ rotations_24.transpose(0, 2, 1)
    # Center-translate: each rotated cloud's centroid -> gt_centroid.
    rot_centroids = rotated.mean(axis=1, keepdims=True)
    rotated = rotated - rot_centroids + gt_centroid[None, None, :]

    # Batched pred -> gt distances (raw, not squared) for all rotations.
    tree_gt = cKDTree(pts_gt)
    flat_rotated = rotated.reshape(-1, 3)
    d_pg_flat, _ = tree_gt.query(flat_rotated)
    d_pg = d_pg_flat.reshape(n_rot, n_pred)
    d_pg2_mean = (d_pg ** 2).mean(axis=1)

    # Per-rotation gt -> pred; track the min-CD rotation as we go.
    best_k = -1
    best_cd = float("inf")
    best_d_gp = None
    for k in range(n_rot):
        tree_pred = cKDTree(rotated[k])
        d_gp, _ = tree_pred.query(pts_gt)
        cd_k = (d_pg2_mean[k] + float(np.mean(d_gp ** 2))) / 2.0
        if cd_k < best_cd:
            best_cd = cd_k
            best_k = k
            best_d_gp = d_gp

    if best_k < 0 or best_d_gp is None:
        return None

    prec = float((d_pg[best_k] <= tau).mean())
    rec = float((best_d_gp <= tau).mean())
    fscore = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "cd": float(best_cd),
        "f_score": float(fscore),
        "precision": prec,
        "recall": rec,
    }


# --------------------------------------------------------------------------
# Hungarian matching
# --------------------------------------------------------------------------
def align_assembly_parts_by_geometry(
    gt_parts: List[Dict[str, Any]],
    pred_parts: List[Dict[str, Any]],
    gt_union_path: str,
    pred_union_path: str,
    align_transform_4x4,
    n_points: int = N_SAMPLE_POINTS,
    f_score_tau_frac: float = F_SCORE_TAU_FRAC,
    f_score_min: float = F_SCORE_MIN,
    pointcloud_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One-to-one match GT bodies <-> pred (decomposed) parts.

    GT side transform: ``_build_normalize_4x4(gt_union_path)`` (GT parts into GT's
    unit-cube frame). Pred side transform: ``align_transform_4x4`` as given
    (already = ``best_transform @ pred_norm``). The same one transform per side is
    applied to every part, so relative inter-part sizes are preserved.

    Cost is F-Score (``1 - f_score``), not CD — CD as cost biased the Hungarian
    assignment toward small-extent pred parts (CD scales with extent^2). ``tau_ij``
    is per-GT-part: ``f_score_tau_frac * diag_gt_i`` from the GT cloud's AABB
    diagonal in the shared frame. Returns ALL Hungarian-assigned pairs in
    ``aligned`` with an ``accepted = (f_score >= f_score_min)`` flag;
    ``unmatched_gt``/``unmatched_pred`` capture only dimension excess."""
    n, m = len(gt_parts), len(pred_parts)
    if n == 0:
        return {"aligned": [], "unmatched_gt": [],
                "unmatched_pred": list(range(m)), "error": "No GT parts"}
    if m == 0:
        return {"aligned": [], "unmatched_gt": list(range(n)),
                "unmatched_pred": [], "error": "No pred parts"}

    if align_transform_4x4 is None:
        return {"aligned": [], "unmatched_gt": list(range(n)),
                "unmatched_pred": list(range(m)),
                "error": "no align_transform_4x4 (global align must run before part eval)"}

    try:
        np = _np()
        linear_sum_assignment = _linear_sum_assignment()
    except Exception as exc:
        return {"aligned": [], "unmatched_gt": list(range(n)),
                "unmatched_pred": list(range(m)),
                "error": f"scipy/numpy import failed: {exc}"}

    T_pred = np.asarray(align_transform_4x4, dtype=np.float64)
    if T_pred.shape != (4, 4):
        return {"aligned": [], "unmatched_gt": list(range(n)),
                "unmatched_pred": list(range(m)),
                "error": f"align_transform_4x4 has shape {T_pred.shape}, expected (4, 4)"}

    T_gt = _build_normalize_4x4(gt_union_path)
    if T_gt is None:
        return {"aligned": [], "unmatched_gt": list(range(n)),
                "unmatched_pred": list(range(m)),
                "error": f"could not build GT normalize from {gt_union_path}"}
    # pred_union_path is advisory: align_transform_4x4 already encodes pred's
    # normalize step. We still verify it loads as a canary that the decomposition
    # export landed.
    if not Path(pred_union_path).exists():
        return {"aligned": [], "unmatched_gt": list(range(n)),
                "unmatched_pred": list(range(m)),
                "error": f"pred union mesh missing at {pred_union_path}"}

    if pointcloud_cache is None:
        pointcloud_cache = {}

    gt_clouds = []
    for entry in gt_parts:
        path = entry.get("stl_path") or entry.get("step_path")
        if not path or not Path(path).exists():
            gt_clouds.append(None)
            continue
        gt_clouds.append(_sample_part_with_transform(
            path, n_points, T_gt, cache=pointcloud_cache,
        ))

    pred_clouds = []
    for entry in pred_parts:
        path = entry.get("stl") or entry.get("step")
        if not path or not Path(path).exists():
            pred_clouds.append(None)
            continue
        pred_clouds.append(_sample_part_with_transform(
            path, n_points, T_pred, cache=pointcloud_cache,
        ))

    # Per-GT-part bbox diagonal in the shared GT-normalised frame, from sampled
    # points. Used to scale the F-Score acceptance radius tau.
    gt_diags: List[Optional[float]] = []
    for cloud in gt_clouds:
        if cloud is None or cloud.shape[0] == 0:
            gt_diags.append(None)
            continue
        diag = float(np.linalg.norm(cloud.max(axis=0) - cloud.min(axis=0)))
        gt_diags.append(diag if diag > 0 else None)

    # Cost = 1 - f_score; pairs with no usable metrics get INF_COST so Hungarian
    # skips them (only relevant when a part failed to sample).
    INF_COST = 1e9
    cost = np.full((n, m), INF_COST, dtype=float)
    metrics_grid: List[List[Optional[Dict[str, float]]]] = [
        [None] * m for _ in range(n)
    ]
    rotations = _get_24_rotations()
    for i, ga in enumerate(gt_clouds):
        if ga is None or gt_diags[i] is None:
            continue
        tau = f_score_tau_frac * gt_diags[i]
        for j, pa in enumerate(pred_clouds):
            if pa is None:
                continue
            pair = _pair_metrics_min_over_24rot(
                pa, ga, tau=tau, rotations_24=rotations,
            )
            if pair is None:
                continue
            metrics_grid[i][j] = pair
            fs = pair["f_score"]
            if fs is not None:
                cost[i, j] = 1.0 - float(fs)

    row_ind, col_ind = linear_sum_assignment(cost)
    aligned: List[Dict[str, Any]] = []
    seen_gt, seen_pred = set(), set()
    for r, c in zip(row_ind, col_ind):
        if float(cost[r, c]) >= INF_COST:
            # No metrics for this pair (sampling failed) — leave both sides as
            # "unmatched" (dimension-excess bucket).
            continue
        pair = metrics_grid[r][c]
        if pair is None:
            continue
        accepted = bool(pair["f_score"] >= f_score_min)
        aligned.append({
            "gt_idx": int(r),
            "pred_idx": int(c),
            "cd_match": pair["cd"],
            "f_score": pair["f_score"],
            "f_precision": pair["precision"],
            "f_recall": pair["recall"],
            "tau": float(f_score_tau_frac * gt_diags[r]),
            "accepted": accepted,
        })
        seen_gt.add(int(r))
        seen_pred.add(int(c))

    return {
        "aligned": aligned,
        "unmatched_gt": sorted(set(range(n)) - seen_gt),
        "unmatched_pred": sorted(set(range(m)) - seen_pred),
        "error": None,
    }


# --------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------
def evaluate_assembly_parts(
    gt_parts: List[Dict[str, Any]],
    pred_parts: List[Dict[str, Any]],
    gt_union_path: Optional[str] = None,
    pred_union_path: Optional[str] = None,
    align_transform_4x4=None,
    stage2_fidelity: Optional[Dict[str, Any]] = None,
    fidelity_cd_max: float = FIDELITY_CD_MAX,
    fidelity_iou_min: float = FIDELITY_IOU_MIN,
    n_points: int = N_SAMPLE_POINTS,
    f_score_tau_frac: float = F_SCORE_TAU_FRAC,
    f_score_min: float = F_SCORE_MIN,
    gt_dedup_sig: int = GT_DEDUP_SIG,
    pred_dedup_sig: int = PRED_DEDUP_SIG,
) -> Dict[str, Any]:
    """Run dedup -> global align -> 24-rotation per-pair CD/F-Score -> Hungarian
    for one case.

    Fidelity gate: when ``stage2_fidelity`` shows the decomposition redesigned the
    union (``cd > fidelity_cd_max`` AND ``iou_v < fidelity_iou_min``), every
    numeric field is None so the case drops from aggregate means. The gate only
    fires on a well-formed numeric dict; pass ``None`` to disable it.

    Count-level (PartMatchF1): a Hungarian pair is accepted iff
    ``f_score >= f_score_min``; recall/precision/F1 over accepted post-dedup unique
    counts. Quality-level (PartFS): mean per-pair F-Score over ALL Hungarian pairs
    (no acceptance filter, to avoid selection bias) plus a size-aware normalized
    variant ``sum(f_score) / max(N, M)``."""
    fidelity_excluded = False
    if isinstance(stage2_fidelity, dict):
        cd = stage2_fidelity.get("cd")
        iou = stage2_fidelity.get("iou_v")
        if (isinstance(cd, (int, float)) and isinstance(iou, (int, float))
                and cd > fidelity_cd_max and iou < fidelity_iou_min):
            fidelity_excluded = True

    # Symmetric GT/pred dedup by geometric fingerprint. Same body geometry placed
    # N times produces N byte-different STLs that all score identically under the
    # 24-rot + centroid-translate pair metric. Collapsing both sides keeps the
    # metric self-contained and turns N x M Hungarian into K_gt x K_pred. Skipped
    # under fidelity exclusion since the case is dropping anyway.
    gt_dedupe_info: Optional[Dict[str, Any]] = None
    pred_dedupe_info: Optional[Dict[str, Any]] = None
    gt_parts_unique: List[Dict[str, Any]] = gt_parts
    pred_parts_unique: List[Dict[str, Any]] = pred_parts
    # If every gt_parts entry already carries ``instance_count``, an upstream
    # loader has deduped GT (using B-Rep-derived part classes — stronger than an
    # STL-derived fingerprint, since topology counts come from the source B-Rep,
    # not tessellated surfaces). Trust it and record a no-op so downstream readers
    # see input_count == unique_count.
    if not fidelity_excluded and gt_parts:
        loader_already_deduped = all(
            "instance_count" in p for p in gt_parts
        )
        if loader_already_deduped:
            gt_parts_unique = list(gt_parts)
            gt_dedupe_info = {
                "unique": list(gt_parts),
                "groups": [],
                "input_count": len(gt_parts),
                "unique_count": len(gt_parts),
                "unfingerprinted": 0,
                "source": "loader_instance_count",
            }
        else:
            gt_dedupe_info = _dedupe_parts_by_fingerprint(gt_parts, sig=gt_dedup_sig)
            gt_dedupe_info["source"] = "metric_stl_fingerprint"
            gt_parts_unique = gt_dedupe_info["unique"]
            if gt_dedupe_info["input_count"] != gt_dedupe_info["unique_count"]:
                logger.debug(
                    "part eval gt dedup: %d -> %d (sig=%d, unfingerprinted=%d)",
                    gt_dedupe_info["input_count"],
                    gt_dedupe_info["unique_count"],
                    gt_dedup_sig,
                    gt_dedupe_info["unfingerprinted"],
                )
    if not fidelity_excluded and pred_parts:
        pred_dedupe_info = _dedupe_parts_by_fingerprint(pred_parts, sig=pred_dedup_sig)
        pred_parts_unique = pred_dedupe_info["unique"]
        if pred_dedupe_info["input_count"] != pred_dedupe_info["unique_count"]:
            logger.debug(
                "part eval pred dedup: %d -> %d (sig=%d, unfingerprinted=%d)",
                pred_dedupe_info["input_count"],
                pred_dedupe_info["unique_count"],
                pred_dedup_sig,
                pred_dedupe_info["unfingerprinted"],
            )

    gt_count = len(gt_parts_unique)
    pred_count = len(pred_parts_unique)

    fidelity_gate = {
        "cd_max": fidelity_cd_max,
        "iou_min": fidelity_iou_min,
        "stage2_cd": (stage2_fidelity or {}).get("cd"),
        "stage2_iou_v": (stage2_fidelity or {}).get("iou_v"),
    }

    if fidelity_excluded:
        return {
            "alignment": {
                "gt_count": gt_count,
                "pred_count": pred_count,
                "gt_input_count": gt_count,
                "pred_input_count": pred_count,
                "matched_count": None,
                "unmatched_gt": None,
                "unmatched_pred": None,
                "match_recall": None,
                "match_precision": None,
                "match_f1": None,
                "f_score_tau_frac": f_score_tau_frac,
                "f_score_min": f_score_min,
                "error": "fidelity_excluded: decomposition redesigned union geometry",
            },
            "gt_dedupe": None,
            "pred_dedupe": None,
            "per_part": [],
            "per_part_mean": {
                "chamfer_distance": None,
                "f_score": None,
                "f_score_normalized": None,
                "f_precision": None,
                "f_recall": None,
            },
            "fidelity_excluded": True,
            "fidelity_gate": fidelity_gate,
        }

    geom = align_assembly_parts_by_geometry(
        gt_parts_unique, pred_parts_unique,
        gt_union_path=gt_union_path,
        pred_union_path=pred_union_path,
        align_transform_4x4=align_transform_4x4,
        n_points=n_points,
        f_score_tau_frac=f_score_tau_frac,
        f_score_min=f_score_min,
    )

    aligned = geom["aligned"]                # ALL Hungarian pairs
    unmatched_gt = geom["unmatched_gt"]       # dimension excess only
    unmatched_pred = geom["unmatched_pred"]
    geom_error = geom.get("error")

    # Count-level metrics use the acceptance filter (f_score >= f_score_min).
    accepted_count = sum(1 for pair in aligned if pair.get("accepted"))
    if gt_count == 0:
        match_recall = None
        match_precision = None
        match_f1 = None
    else:
        match_recall = accepted_count / gt_count
        match_precision = (accepted_count / pred_count) if pred_count > 0 else 0.0
        denom = match_precision + match_recall
        match_f1 = (2.0 * match_precision * match_recall / denom) if denom > 0 else 0.0

    # Per-part list: one entry per Hungarian-assigned pair (acceptance is a flag,
    # not a filter). Quality means aggregate over ALL of them.
    per_part: List[Dict[str, Any]] = []
    cd_vals: List[float] = []
    fs_vals: List[float] = []
    fp_vals: List[float] = []
    fr_vals: List[float] = []
    for pair in aligned:
        gi, pi = pair["gt_idx"], pair["pred_idx"]
        gt_entry = gt_parts_unique[gi]
        pred_entry = pred_parts_unique[pi]
        cd_val = pair["cd_match"]
        fs_val = pair["f_score"]
        fp_val = pair["f_precision"]
        fr_val = pair["f_recall"]
        per_part.append({
            "gt_idx": gi,
            "pred_idx": pi,
            "gt_name": gt_entry.get("role_name") or gt_entry.get("name"),
            "pred_name": pred_entry.get("name"),
            "gt_instance_count": int(gt_entry.get("instance_count", 1)),
            "pred_instance_count": int(pred_entry.get("instance_count", 1)),
            "gt_aliases": gt_entry.get("aliases") or [],
            "pred_aliases": pred_entry.get("aliases") or [],
            "cd_match": cd_val,
            "f_score": fs_val,
            "f_precision": fp_val,
            "f_recall": fr_val,
            "tau": pair.get("tau"),
            "accepted": bool(pair.get("accepted")),
        })
        cd_vals.append(cd_val)
        fs_vals.append(fs_val)
        fp_vals.append(fp_val)
        fr_vals.append(fr_val)

    cd_mean = (sum(cd_vals) / len(cd_vals)) if cd_vals else None
    fs_mean = (sum(fs_vals) / len(fs_vals)) if fs_vals else None
    fp_mean = (sum(fp_vals) / len(fp_vals)) if fp_vals else None
    fr_mean = (sum(fr_vals) / len(fr_vals)) if fr_vals else None

    # Size-aware aggregate: sum of F-Scores over Hungarian matches divided by
    # max(N, M). Penalizes both kinds of dimension error. Equals the per-match
    # mean only when N == M.
    denom_norm = max(gt_count, pred_count)
    fs_normalized = (sum(fs_vals) / denom_norm) if (fs_vals and denom_norm > 0) else None

    return {
        "alignment": {
            "gt_count": gt_count,
            "pred_count": pred_count,
            "gt_input_count": (gt_dedupe_info["input_count"]
                               if gt_dedupe_info else gt_count),
            "pred_input_count": (pred_dedupe_info["input_count"]
                                 if pred_dedupe_info else pred_count),
            "hungarian_count": len(aligned),
            "matched_count": accepted_count,
            "unmatched_gt": len(unmatched_gt),
            "unmatched_pred": len(unmatched_pred),
            "match_recall": round(match_recall, 4) if match_recall is not None else None,
            "match_precision": round(match_precision, 4) if match_precision is not None else None,
            "match_f1": round(match_f1, 4) if match_f1 is not None else None,
            "f_score_tau_frac": f_score_tau_frac,
            "f_score_min": f_score_min,
            "error": geom_error,
        },
        "gt_dedupe": gt_dedupe_info,
        "pred_dedupe": pred_dedupe_info,
        "per_part": per_part,
        "per_part_mean": {
            "chamfer_distance": cd_mean,
            "f_score": fs_mean,
            "f_score_normalized": fs_normalized,
            "f_precision": fp_mean,
            "f_recall": fr_mean,
        },
        "fidelity_excluded": False,
        "fidelity_gate": fidelity_gate,
    }


# ==========================================================================
# Part metric bucket — runs the single-call decomposition, then evaluates
# ==========================================================================
_PART_KEYS = ("part_match_f1", "part_fs")


def _empty(note: Optional[str] = None) -> dict:
    out: dict = {"part_match_f1": None, "part_fs": None}
    if note:
        out["part_note"] = note
    return out


def _stage1_code(ctx: ScoreContext) -> Optional[str]:
    """The stage-1 (unified) program text. ScoreContext does not carry the raw
    code directly, so consume it defensively: prefer ``ctx.compiled['code']``
    (if a future compile stage embeds it), else ``ctx.shared['stage1_code']``
    (the pipeline seeds this from the prediction record)."""
    code = ctx.compiled.get("code")
    if code:
        return code
    return ctx.shared.get("stage1_code")


def _align_transform_for_part(ctx: ScoreContext):
    """Fetch the whole-assembly ``align_transform_4x4`` mapping stage-1 pred into
    the GT-normalised frame. Prefer the geometry bucket's cached product (paper:
    one transform per side, shared); if the geometry bucket did not run on this
    case (e.g. ``--metric part`` alone), recompute it by aligning the stage-1
    union vs the GT union.

    Returns ``(align_transform_4x4 | None, note | None)``."""
    geo = ctx.shared.get("geometry")
    if isinstance(geo, dict) and geo.get("align_transform_4x4") is not None:
        return geo["align_transform_4x4"], None

    # Recompute: align stage-1 pred union vs GT union (Assembly-3D geometry
    # config: exterior sampling, normalize+scale, voxel IoU).
    pred_stl = ctx.compiled.get("stl")
    gt_path = ctx.case.gt_step or ctx.case.gt_mesh
    if not pred_stl or not gt_path:
        return None, "no stage-1 union / GT mesh to derive align_transform_4x4"
    from .geometry import align_and_compute

    mesh_pred = _load_mesh(str(pred_stl))
    mesh_gt = _load_mesh(str(gt_path))
    if mesh_pred is None or mesh_gt is None:
        return None, "could not load stage-1 union / GT mesh for alignment"
    _metrics, align_transform_4x4, _aligned = align_and_compute(
        mesh_pred, mesh_gt,
        sampling_mode="exterior",
        skip_normalize_and_scale=False,
        compute_iou_variant="voxel",
    )
    return align_transform_4x4, None


def _compute_stage2_fidelity(stage1_stl: str, stage2_stl: str) -> Optional[Dict[str, Any]]:
    """CD + IoU-V of the decomposition union (stage-2) vs the stage-1 union.

    Returns ``{"cd", "iou_v"}`` (pred = stage-2, gt = stage-1, mirroring the
    source) or ``None`` if either mesh fails to load — in which case the gate is
    disabled (it only fires on a well-formed numeric dict)."""
    mesh1 = _load_mesh(str(stage1_stl))
    mesh2 = _load_mesh(str(stage2_stl))
    if mesh1 is None or mesh2 is None:
        return None
    from .geometry import align_and_compute

    try:
        # Assembly-3D uses exterior sampling (matching the geometry bucket), so the
        # fidelity gate is measured under the same regime as the rest of the task.
        metrics, _t, _a = align_and_compute(
            mesh2, mesh1,
            sampling_mode="exterior",
            skip_normalize_and_scale=False,
            compute_iou_variant="voxel",
        )
    except Exception as exc:
        logger.warning("stage-2 fidelity computation failed: %s", exc)
        return None
    return {"cd": metrics.get("chamfer_distance"), "iou_v": metrics.get("iou")}


class _PartBucket(MetricBucket):
    bucket = "part"
    requires: set[str] = {"mesh", "parts"}

    def score(self, ctx: ScoreContext) -> dict:
        # Surface a clear MissingDependencyError early if the geometry extra is
        # absent (numpy/scipy/trimesh power both the decomposition fidelity and
        # the part metric).
        _np()
        _trimesh()
        _ckdtree()

        # 1. Decomposition requires a model client (no retry).
        if ctx.decompose_client is None:
            return _empty("no decompose_client configured")

        # 2. Need the stage-1 unified program to decompose, and a stage-1 union.
        stage1_code = _stage1_code(ctx)
        if not stage1_code:
            return _empty("stage-1 code unavailable (cannot run decomposition)")
        stage1_stl = ctx.compiled.get("stl")
        if not stage1_stl:
            return _empty("no valid stage-1 union mesh (compile invalid)")

        from ..formats import get_format
        from ..tasks.assembly_3d import TASK

        fmt = get_format(ctx.fmt)

        # Single render of the pred geometry as an optional visual cue. The demo
        # harness does not require a render; infer part boundaries from the code.
        has_image = False

        # 2. Build the stage-2 decomposition prompt + 3. call the client.
        prompt = TASK.build_decompose_prompt(fmt, stage1_code, has_image)
        try:
            resp = ctx.decompose_client.generate(
                prompt, system=fmt.system_guidelines, timeout=300,
            )
        except Exception as exc:
            return _empty(f"decomposition call failed: {type(exc).__name__}: {exc}")

        stage2_code = fmt.extract_code(resp.text)
        if not stage2_code.strip():
            return _empty("empty decomposition code extraction")

        # 3. Compile the parts-structured program -> parts_meta.json + parts/*.stl.
        stage2_dir = Path(ctx.work_dir) / "stage2"
        stage2_dir.mkdir(parents=True, exist_ok=True)
        try:
            cr = fmt.compile(stage2_code, stage2_dir)
        except Exception as exc:
            return _empty(f"decomposition compile failed: {type(exc).__name__}: {exc}")
        if not cr.stl:
            return _empty("decomposition produced no union STL")
        if not cr.parts_meta:
            return _empty("decomposition produced no parts_meta.json")

        pred_union_path = cr.stl
        pred_parts = load_pred_parts_from_dir(Path(cr.parts_meta).parent)
        if not pred_parts:
            return _empty("no predicted parts after decomposition")

        # 4. Stage-2 fidelity (CD/IoU-V of stage2-union vs stage1-union) + gate.
        stage2_fidelity = _compute_stage2_fidelity(stage1_stl, pred_union_path)

        # 5. GT parts + the shared pred-side align transform.
        gt_part_paths = [p for p in ctx.case.gt_parts if p]
        if not gt_part_paths:
            return _empty("case has no GT parts")
        gt_parts = [{"stl_path": str(p)} for p in gt_part_paths]
        gt_union_path = ctx.case.gt_step or ctx.case.gt_mesh
        if not gt_union_path:
            return _empty("case has no GT union mesh")

        align_transform_4x4, note = _align_transform_for_part(ctx)
        if align_transform_4x4 is None:
            return _empty(note or "no align_transform_4x4 available")

        # 6. Evaluate.
        result = evaluate_assembly_parts(
            gt_parts, pred_parts,
            gt_union_path=str(gt_union_path),
            pred_union_path=str(pred_union_path),
            align_transform_4x4=align_transform_4x4,
            stage2_fidelity=stage2_fidelity,
        )

        alignment = result.get("alignment", {})
        per_part_mean = result.get("per_part_mean", {})
        out: dict = {
            "part_match_f1": alignment.get("match_f1"),
            "part_fs": per_part_mean.get("f_score"),
            # diagnostics (ignored by aggregation)
            "part_match_p": alignment.get("match_precision"),
            "part_match_r": alignment.get("match_recall"),
            "part_fs_normalized": per_part_mean.get("f_score_normalized"),
        }
        if result.get("fidelity_excluded"):
            out["part_note"] = "fidelity_excluded"
        elif alignment.get("error"):
            out["part_note"] = alignment["error"]
        return out


BUCKET = _PartBucket()
