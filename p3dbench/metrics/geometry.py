"""Geometry metrics + mesh alignment core for P3D-Bench.

Computes Chamfer Distance, F-scores, Normal Consistency and IoU (voxel/CSG) on a
single shared, aligned 8192-point cloud per mesh. The alignment is a
``pca+discrete`` search (48 signed-permutation rotations + a PCA-canonicalized
path) followed by a bounded L-BFGS-B scale/translation refine.

Faithful port of ``cadbenchmark/metrics/geometry_metrics.py`` with production
cruft stripped (thread RNG lock, dead STEP B-Rep analysis, legacy validity /
single-case drivers, visible-view metrics, env-var overrides). Algorithms,
constants and thresholds are preserved verbatim.

Heavy dependencies (numpy, scipy, trimesh, manifold3d) are imported lazily via
:func:`p3dbench.utils.require` so importing this module never requires the
``geometry`` extra to be installed; a bucket whose dependency is absent lets the
``MissingDependencyError`` propagate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple, TYPE_CHECKING

from ..utils import require
from .base import MetricBucket, ScoreContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np  # noqa: F401
    import trimesh  # noqa: F401

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Lazy heavy-dependency accessors (so module import never needs the extra)
# --------------------------------------------------------------------------
def _np():
    return require("numpy", "geometry", "Geometry metrics")


def _trimesh():
    return require("trimesh", "geometry", "Geometry metrics")


def _ckdtree():
    return require("scipy", "geometry", "Geometry metrics").spatial.cKDTree


# --------------------------------------------------------------------------
# Constants (verbatim from the source; env-var override hooks dropped)
# --------------------------------------------------------------------------
EXTERIOR_VOXEL_RESOLUTION = 256
IOU_VOXEL_RESOLUTION = 128
EXTERIOR_OVERSAMPLE_FACTOR = 8
# Cells of slack (Chebyshev) allowed when matching a sample point to exterior-air
# voxels: trimesh surface voxelisation can be ~3 cells thick on near-tangent
# regions of curved surfaces, so a point on the outer skin may land 2-3 voxels
# deep inside the "filled" mask. 3 cells of dilation recovers ~100% of points on
# convex curved geometry while staying safe for internal faces.
EXTERIOR_DILATION_ITERS = 3
_EXTERIOR_MAX_ROUNDS = 8
_EXTERIOR_PAD = 1 + max(EXTERIOR_DILATION_ITERS, 1)


# ==========================================================================
# Normalization helpers
# ==========================================================================
def _pca_axes(pts):
    """PCA eigenvectors sorted by descending eigenvalue (right-handed frame).

    Returns (eigenvalues, eigenvectors) where eigenvector columns are principal
    axes in descending variance order, forming a right-handed frame.
    """
    np = _np()
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending order; reverse to descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Ensure right-handed coordinate system
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1
    return eigvals, eigvecs


def normalize_mesh(mesh):
    """Normalize mesh to fit in [-0.5, 0.5]^3 centered at origin."""
    np = _np()
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    mesh.vertices -= center
    scale = np.max(bounds[1] - bounds[0])
    if scale > 0:
        mesh.vertices /= scale
    return mesh


def center_mesh(mesh):
    """Center mesh at origin without scaling (bbox center subtracted)."""
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    mesh.vertices -= center
    return mesh


# ==========================================================================
# Point sampling
# ==========================================================================
def _sample_points(mesh, n_points: int = 8192, seed: int = 42):
    """Sample points uniformly from mesh surface (deterministic, seed=42)."""
    np = _np()
    trimesh = _trimesh()
    np.random.seed(seed)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def _sample_points_with_normals(mesh, n_points: int = 8192, seed: int = 42):
    """Sample points and their face normals from mesh surface (seed=42)."""
    np = _np()
    trimesh = _trimesh()
    np.random.seed(seed)
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_indices]
    return points, normals


# ==========================================================================
# Exterior-only point sampling (voxel flood-fill based)
# ==========================================================================
#
# Motivation: trimesh.sample.sample_surface samples *every* triangle including
# internal faces of unfused assemblies (fusion360 GTs are concatenated per-solid;
# json/threejs outputs preserve assembly structure; cadquery/openscad fuse to a
# single solid with no internal faces). Comparing the former against the latter
# with chamfer/F-score systematically penalises the fused outputs for "missing"
# internal points that have no semantic meaning.
#
# This sampler restricts samples to the *exterior* surface by:
#   1. voxelising the mesh, padding with background, flood-filling to identify
#      the interior, then computing the exterior-air voxel mask;
#   2. uniformly oversampling the full surface and rejecting any point whose
#      voxel cell is not adjacent to an exterior-air voxel.
@dataclass(frozen=True)
class _MeshVoxelState:
    """Cached voxelisation + flood-fill products for a single mesh.

    All arrays share the same shape — the trimesh bounding-box voxelisation
    padded by ``_EXTERIOR_PAD`` layers of background on every side.
    ``vox_origin_unpadded`` / ``vox_pitch`` describe the *unpadded* grid, so
    consumers that need unpadded indices must crop
    ``[_EXTERIOR_PAD:-_EXTERIOR_PAD, ...]`` before mapping through
    ``vox.transform``-derived math.
    """

    filled_padded: Any          # bool ndarray; binary_fill_holes result
    ext_dilated_padded: Any     # bool ndarray; ~filled then dilated
    vox_origin_unpadded: Any    # world corner of unpadded grid
    vox_pitch: float            # uniform pitch


def _build_mesh_voxel_state(mesh, resolution: int = EXTERIOR_VOXEL_RESOLUTION):
    """Voxelise + pad + flood-fill *mesh* once and cache the results.

    Pitch is derived from ``mesh.extents.max() / resolution`` so this helper is
    independent of any external normalisation. Returns ``None`` on any failure
    (missing scipy, empty voxelisation, degenerate extent, …) so callers can
    fall back independently.
    """
    np = _np()
    try:
        ndimage = require("scipy", "geometry", "Geometry metrics").ndimage
        binary_fill_holes = ndimage.binary_fill_holes
        binary_dilation = ndimage.binary_dilation
    except Exception:
        return None

    try:
        extent = float(np.max(mesh.extents))
        if extent <= 0:
            return None

        vox = mesh.voxelized(extent / resolution)
        surface = np.asarray(vox.matrix, dtype=bool)
        if surface.size == 0 or not surface.any():
            return None

        # Pad with background so binary_fill_holes sees exterior air connected
        # to the array boundary (required — after normalize_mesh the mesh always
        # hugs its bbox along the longest axis) and so subsequent dilation never
        # wraps around the array edge.
        surface_padded = np.pad(
            surface, pad_width=_EXTERIOR_PAD, mode="constant", constant_values=False)
        filled_padded = binary_fill_holes(surface_padded)
        if filled_padded is None:
            return None

        ext = ~filled_padded
        if EXTERIOR_DILATION_ITERS > 0:
            ext_dilated_padded = binary_dilation(
                ext, iterations=EXTERIOR_DILATION_ITERS)
        else:
            ext_dilated_padded = ext

        return _MeshVoxelState(
            filled_padded=filled_padded,
            ext_dilated_padded=ext_dilated_padded,
            vox_origin_unpadded=np.asarray(vox.transform[:3, 3], dtype=float),
            vox_pitch=float(vox.transform[0, 0]),
        )
    except Exception as e:
        logger.debug(f"_build_mesh_voxel_state failed: {e}")
        return None


def _filter_points_to_exterior(points, ext_padded, vox_origin_unpadded, vox_pitch):
    """Return a bool array marking which points lie on the exterior surface.

    Each point is mapped to its enclosing voxel and looked up in ``ext_padded``.
    The grid is non-cubic for elongated meshes (one cell per axis-pitch along
    mesh extents), so each axis has its own size.
    """
    np = _np()
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    R0 = ext_padded.shape[0] - 2 * _EXTERIOR_PAD
    R1 = ext_padded.shape[1] - 2 * _EXTERIOR_PAD
    R2 = ext_padded.shape[2] - 2 * _EXTERIOR_PAD
    idx_unpadded = np.floor(
        (points - vox_origin_unpadded) / vox_pitch).astype(np.int64)
    ii = np.clip(idx_unpadded[:, 0], 0, R0 - 1) + _EXTERIOR_PAD
    jj = np.clip(idx_unpadded[:, 1], 0, R1 - 1) + _EXTERIOR_PAD
    kk = np.clip(idx_unpadded[:, 2], 0, R2 - 1) + _EXTERIOR_PAD
    return ext_padded[ii, jj, kk]


def _sample_points_exterior(
    mesh,
    n_points: int = 8192,
    seed: int = 42,
    resolution: int = EXTERIOR_VOXEL_RESOLUTION,
    oversample: int = EXTERIOR_OVERSAMPLE_FACTOR,
    has_open_edges: bool = False,
    state: Optional[_MeshVoxelState] = None,
):
    """Sample ``n_points`` points restricted to the mesh's exterior surface.

    Uses voxel flood-fill to detect the interior, then oversamples the full
    surface and rejects points whose voxel neighbourhood is not exterior air.
    Falls back to whole-surface sampling when the mesh has open edges (flood
    fill cannot be trusted).

    Returns ``(points, face_indices, info)`` where ``info`` carries diagnostic
    fields used by :func:`align_and_compute` (``fallback``, ``kept_ratio``).
    """
    np = _np()
    trimesh = _trimesh()
    info: dict = {"fallback": None, "kept_ratio": 1.0}

    if has_open_edges:
        info["fallback"] = "open_mesh"
        np.random.seed(seed)
        pts, fidx = trimesh.sample.sample_surface(mesh, n_points)
        return pts, fidx, info

    if state is None:
        state = _build_mesh_voxel_state(mesh, resolution)
    if state is None:
        info["fallback"] = "voxelize_failed"
        np.random.seed(seed)
        pts, fidx = trimesh.sample.sample_surface(mesh, n_points)
        return pts, fidx, info

    ext_padded = state.ext_dilated_padded
    vox_origin = state.vox_origin_unpadded
    vox_pitch = state.vox_pitch

    kept_pts_chunks: list = []
    kept_fidx_chunks: list = []
    total_drawn = 0
    total_kept = 0
    target = n_points
    chunk_size = max(n_points * max(oversample, 1), n_points)

    for round_idx in range(_EXTERIOR_MAX_ROUNDS):
        np.random.seed(seed + round_idx)
        pts, fidx = trimesh.sample.sample_surface(mesh, chunk_size)
        mask = _filter_points_to_exterior(
            pts, ext_padded, vox_origin, vox_pitch)
        kept_pts_chunks.append(pts[mask])
        kept_fidx_chunks.append(fidx[mask])
        total_drawn += chunk_size
        total_kept += int(mask.sum())
        if total_kept >= target:
            break

    if total_kept == 0:
        # Pathological: nothing classified as exterior. Fall back so the
        # downstream metric still produces a number.
        logger.warning(
            "_sample_points_exterior: 0 exterior points after %d rounds; "
            "falling back to surface sampling", _EXTERIOR_MAX_ROUNDS)
        info["fallback"] = "no_exterior_points"
        np.random.seed(seed)
        pts, fidx = trimesh.sample.sample_surface(mesh, n_points)
        return pts, fidx, info

    pts_all = np.concatenate(kept_pts_chunks, axis=0)
    fidx_all = np.concatenate(kept_fidx_chunks, axis=0)

    # Down/up-sample to exactly n_points (with replacement only if necessary).
    rng = np.random.default_rng(seed)
    if len(pts_all) >= n_points:
        sel = rng.choice(len(pts_all), size=n_points, replace=False)
    else:
        logger.warning(
            "_sample_points_exterior: only %d exterior points after %d rounds "
            "(<%d requested); padding with replacement",
            len(pts_all), _EXTERIOR_MAX_ROUNDS, n_points)
        sel = rng.choice(len(pts_all), size=n_points, replace=True)
        info["fallback"] = "insufficient_exterior_points"

    info["kept_ratio"] = float(total_kept) / float(max(total_drawn, 1))
    return pts_all[sel], fidx_all[sel], info


# ==========================================================================
# Alignment
# ==========================================================================
def align_meshes(mesh_pred, mesh_gt, n_points: int = 4096,
                 method: str = "pca+discrete",
                 max_scale_delta: float = 0.3,
                 max_translate: float = 0.2):
    """Align pred mesh to GT.

    Both inputs should already be independently normalized (or centered). GT is
    kept fixed in the normalized frame; pred is transformed.

    Methods:
        "pca+discrete" (default) — runs both the plain-discrete and the
            PCA-canonicalized 48-rotation searches, biased toward plain-discrete.
        "discrete" — 48 axis permutation/sign combos only, min CD.
        "pca" — principal-axis alignment with 4 sign-flip candidates.

    ``max_scale_delta`` is the maximum deviation from scale=1.0 during
    refinement (0.0 disables scale refinement, e.g. for text2cad);
    ``max_translate`` is the per-axis translation bound.

    Returns ``(aligned_pred, aligned_gt, best_transform_4x4)``. If scipy is
    unavailable the inputs are returned unchanged with an identity transform.
    """
    try:
        _ckdtree()
    except Exception:
        np = _np()
        return mesh_pred, mesh_gt, np.eye(4)

    if method == "pca":
        return _align_meshes_pca(
            mesh_pred, mesh_gt, n_points,
            max_scale_delta=max_scale_delta,
            max_translate=max_translate)
    if method == "pca+discrete":
        return _align_meshes_pca_discrete(
            mesh_pred, mesh_gt, n_points,
            max_scale_delta=max_scale_delta,
            max_translate=max_translate)
    return _align_meshes_discrete(
        mesh_pred, mesh_gt, n_points,
        max_scale_delta=max_scale_delta,
        max_translate=max_translate)


def _sample_for_alignment(mesh, n_points: int, seed: int = 42):
    """Sample points for alignment.

    Uses global surface sampling (all triangles uniformly) rather than
    exterior-only sampling. Interior surfaces rotate together with exterior
    surfaces, so the optimal alignment rotation is the same regardless of
    whether interior faces are included. This avoids the expensive voxel
    flood-fill during the alignment phase.
    """
    return _sample_points(mesh, n_points, seed=seed)


def _select_best_rotation_with_identity_bias(candidates, identity_bias_tol=0.05):
    """From a list of (cd, R) candidates, select the best rotation.

    When multiple candidates have CD within *identity_bias_tol* of the minimum,
    prefer the one closest to the identity matrix (Frobenius norm). This avoids
    flipping near-symmetric shapes into semantically wrong orientations when the
    CD improvement is marginal.
    """
    np = _np()
    if not candidates:
        return np.eye(3)
    best_cd = min(c[0] for c in candidates)
    tol = best_cd * (1.0 + identity_bias_tol)
    near_best = [(cd, R) for cd, R in candidates if cd <= tol]
    # Among near-best, pick closest to identity
    best_R = min(near_best, key=lambda x: np.linalg.norm(x[1] - np.eye(3), 'fro'))[1]
    return best_R


def _search_48_rotations(pts_pred, pts_gt, tree_gt, identity_bias_tol=0.05):
    """Search 48 discrete rotations and return (best_R, best_cd).

    Enumerates all 6 axis permutations × 8 sign combinations = 48 signed
    permutation matrices (includes reflections; dets are ±1, not filtered).
    """
    np = _np()
    cKDTree = _ckdtree()
    perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    signs = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]

    candidates = []
    for perm in perms:
        for sign in signs:
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, sign)):
                R[i, p] = s

            pts = pts_pred @ R.T
            tree_pred = cKDTree(pts)
            d_pred_to_gt, _ = tree_gt.query(pts)
            d_gt_to_pred, _ = tree_pred.query(pts_gt)
            cd = (np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2)) / 2.0
            candidates.append((cd, R))

    best_R = _select_best_rotation_with_identity_bias(candidates, identity_bias_tol)
    best_cd = min(c[0] for c in candidates)
    return best_R, best_cd


def _align_meshes_discrete(mesh_pred, mesh_gt, n_points: int = 4096,
                           identity_bias_tol: float = 0.05,
                           max_scale_delta: float = 0.3,
                           max_translate: float = 0.2):
    """48 axis-permutation × sign-flip alignment, minimizing bidirectional CD.

    When multiple rotations give similar CD (within *identity_bias_tol* of the
    minimum), the rotation closest to identity is preferred to avoid flipping
    near-symmetric shapes into semantically wrong orientations.
    """
    np = _np()
    cKDTree = _ckdtree()
    pts_gt = _sample_for_alignment(mesh_gt, n_points)
    pts_pred_base = _sample_for_alignment(mesh_pred, n_points)
    tree_gt = cKDTree(pts_gt)

    best_R, _ = _search_48_rotations(pts_pred_base, pts_gt, tree_gt, identity_bias_tol)
    best_transform = np.eye(4)
    best_transform[:3, :3] = best_R

    # --- Refine alignment with scale + translation optimization ---
    pts_rotated = pts_pred_base @ best_R.T
    try:
        s_opt, t_opt = _refine_scale_translation(pts_rotated, pts_gt, tree_gt,
                                                  max_scale_delta=max_scale_delta,
                                                  max_translate=max_translate)
        T = np.eye(4)
        T[:3, :3] = s_opt * best_R
        T[:3, 3] = t_opt
        best_transform = T
    except Exception as e:
        logger.debug(f"Scale/translation refinement failed, using rotation only: {e}")

    result = mesh_pred.copy()
    result.apply_transform(best_transform)
    return result, mesh_gt, best_transform


def _align_meshes_pca(mesh_pred, mesh_gt, n_points: int = 4096,
                      max_scale_delta: float = 0.3,
                      max_translate: float = 0.2):
    """Principal-axis alignment with 4 sign-flip candidates (CAD-Coder style).

    1. Sample surface points, compute PCA eigenvectors for each shape.
    2. Build rotation R = V_gt @ inv(V_pred) to align principal axes.
    3. Generate 4 candidates by flipping eigenvector signs (handles sign
       ambiguity of PCA).
    4. Pick the candidate with lowest bidirectional CD.
    5. Refine with scale + translation.
    """
    np = _np()
    cKDTree = _ckdtree()
    pts_gt = _sample_for_alignment(mesh_gt, n_points)
    pts_pred_base = _sample_for_alignment(mesh_pred, n_points)

    # Center both point clouds
    c_gt = pts_gt.mean(axis=0)
    c_pred = pts_pred_base.mean(axis=0)
    pts_gt_c = pts_gt - c_gt
    pts_pred_c = pts_pred_base - c_pred

    _, v_gt = _pca_axes(pts_gt_c)
    _, v_pred = _pca_axes(pts_pred_c)

    # 4 candidate rotations: base alignment + 3 sign flips
    # Sign ambiguity: each eigenvector can be ±, but we must keep det=+1
    Rs = []
    # Candidate 0: direct alignment
    Rs.append(v_gt @ np.linalg.inv(v_pred))
    # Candidates 1-3: flip pairs of axes (flipping one axis would give det=-1,
    # so we flip two at a time to preserve orientation)
    for flip_axis in range(3):
        flip = np.ones(3)
        # Flip the two axes that are NOT flip_axis
        for j in range(3):
            if j != flip_axis:
                flip[j] = -1
        v_pred_flipped = v_pred * flip[np.newaxis, :]
        # Re-ensure right-handedness
        if np.linalg.det(v_pred_flipped) < 0:
            v_pred_flipped[:, -1] *= -1
        Rs.append(v_gt @ np.linalg.inv(v_pred_flipped))

    # Evaluate each candidate
    tree_gt = cKDTree(pts_gt)
    best_cd = float('inf')
    best_R = np.eye(3)

    for R in Rs:
        # Apply rotation around pred centroid, then translate to gt centroid
        pts = (pts_pred_base - c_pred) @ R.T + c_gt
        tree_pred = cKDTree(pts)
        d_pred_to_gt, _ = tree_gt.query(pts)
        d_gt_to_pred, _ = tree_pred.query(pts_gt)
        cd = (np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2)) / 2.0
        if cd < best_cd:
            best_cd = cd
            best_R = R

    # Build full transform: translate to origin, rotate, translate to gt center
    best_transform = np.eye(4)
    best_transform[:3, :3] = best_R
    best_transform[:3, 3] = c_gt - best_R @ c_pred

    # Refine with scale + translation
    pts_rotated = (pts_pred_base - c_pred) @ best_R.T + c_gt
    try:
        s_opt, t_opt = _refine_scale_translation(pts_rotated, pts_gt, tree_gt,
                                                  max_scale_delta=max_scale_delta,
                                                  max_translate=max_translate)
        T = np.eye(4)
        T[:3, :3] = s_opt * best_R
        T[:3, 3] = s_opt * (c_gt - best_R @ c_pred) + t_opt
        best_transform = T
    except Exception as e:
        logger.debug(f"PCA scale/translation refinement failed: {e}")

    result = mesh_pred.copy()
    result.apply_transform(best_transform)
    return result, mesh_gt, best_transform


def _align_meshes_pca_discrete(mesh_pred, mesh_gt, n_points: int = 4096,
                               identity_bias_tol: float = 0.05,
                               pca_bias_tol: float = 0.10,
                               max_scale_delta: float = 0.3,
                               max_translate: float = 0.2):
    """PCA canonicalization of pred + 48 discrete rotations, with fallback.

    Runs two alignment paths and picks the one with lower CD:
      A) Plain discrete: 48 rotations on original pred points.
      B) PCA+discrete: canonicalize pred to PCA frame, then 48 rotations.

    Only pred is PCA-canonicalized (GT is assumed to be in standard pose). Path
    B handles arbitrary tilts (e.g. from Three.js STL exports) that path A alone
    cannot correct. Path A is kept as fallback so PCA instability (e.g.
    near-spherical shapes) never causes regression.

    *pca_bias_tol* (0.10) biases the winner selection toward the plain-discrete
    path: PCA only wins when ``cd_pca < cd_disc * (1 - pca_bias_tol)``. Same
    spirit as ``identity_bias_tol``: when the two paths are within tolerance,
    prefer the more conservative (non-PCA) one.
    """
    np = _np()
    cKDTree = _ckdtree()
    pts_gt = _sample_for_alignment(mesh_gt, n_points)
    pts_pred_base = _sample_for_alignment(mesh_pred, n_points)
    tree_gt = cKDTree(pts_gt)

    # --- Path A: plain discrete (same as _align_meshes_discrete) ---
    R_disc, _ = _search_48_rotations(
        pts_pred_base, pts_gt, tree_gt, identity_bias_tol)

    # --- Path B: PCA canonicalization + discrete ---
    c_pred = pts_pred_base.mean(axis=0)
    pts_pred_c = pts_pred_base - c_pred
    _, v_pred = _pca_axes(pts_pred_c)
    pts_pred_canon = pts_pred_c @ v_pred

    R_pca_disc, _ = _search_48_rotations(
        pts_pred_canon, pts_gt, tree_gt, identity_bias_tol)

    R_world_pca = R_pca_disc @ v_pred.T
    t_pca = -R_world_pca @ c_pred

    def _build_and_refine(R, t_vec):
        """Build 4x4 transform, refine scale+translation, return (transform, cd)."""
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t_vec
        pts_rot = pts_pred_base @ R.T + t_vec
        try:
            s_opt, t_opt = _refine_scale_translation(pts_rot, pts_gt, tree_gt,
                                                      max_scale_delta=max_scale_delta,
                                                      max_translate=max_translate)
            transform[:3, :3] = s_opt * R
            transform[:3, 3] = s_opt * t_vec + t_opt
            pts_final = s_opt * pts_rot + t_opt
        except Exception:
            pts_final = pts_rot
        tree_p = cKDTree(pts_final)
        d_p2g, _ = tree_gt.query(pts_final)
        d_g2p, _ = tree_p.query(pts_gt)
        cd = (np.mean(d_p2g ** 2) + np.mean(d_g2p ** 2)) / 2.0
        return transform, cd

    transform_disc, cd_disc = _build_and_refine(R_disc, np.zeros(3))
    transform_pca, cd_pca = _build_and_refine(R_world_pca, t_pca)

    # Asymmetric tiebreak: PCA must beat discrete by >= pca_bias_tol to win,
    # otherwise discrete is preferred.
    pca_wins = cd_pca < cd_disc * (1.0 - pca_bias_tol)
    best_transform = transform_pca if pca_wins else transform_disc
    logger.debug(
        f"PCA+discrete: {'PCA' if pca_wins else 'discrete'} path selected "
        f"(cd_pca={cd_pca:.6f}, cd_disc={cd_disc:.6f}, tol={pca_bias_tol})")

    result = mesh_pred.copy()
    result.apply_transform(best_transform)
    return result, mesh_gt, best_transform


def _refine_scale_translation(pts_pred_rotated, pts_gt, tree_gt,
                              max_scale_delta=0.3, max_translate=0.2):
    """Optimize uniform scale and translation to minimize Chamfer Distance.

    Finds optimal (s, tx, ty, tz) such that ``s * pts_pred_rotated + t``
    minimizes bidirectional squared CD against pts_gt. Warm-started from the
    centroid offset (clipped to ±max_translate) and the RMS-radius ratio
    (clipped to [1-Δ, 1+Δ]). When *max_scale_delta* is 0, scale is locked at 1.0
    but translation is still optimized via L-BFGS-B.
    """
    np = _np()
    cKDTree = _ckdtree()
    minimize = require("scipy", "geometry", "Geometry metrics").optimize.minimize

    centroid_pred = pts_pred_rotated.mean(axis=0)
    centroid_gt = pts_gt.mean(axis=0)
    t_init = centroid_gt - centroid_pred

    # Warm start for scale
    if max_scale_delta > 0:
        rms_pred = np.sqrt(np.mean(np.sum((pts_pred_rotated - centroid_pred) ** 2, axis=1)))
        rms_gt = np.sqrt(np.mean(np.sum((pts_gt - centroid_gt) ** 2, axis=1)))
        s_init = rms_gt / rms_pred if rms_pred > 1e-10 else 1.0
        s_init = np.clip(s_init, 1.0 - max_scale_delta, 1.0 + max_scale_delta)
    else:
        s_init = 1.0

    # Clip initial translation to bounds
    t_init = np.clip(t_init, -max_translate, max_translate)

    x0 = np.array([s_init, t_init[0], t_init[1], t_init[2]])

    def objective(x):
        s, tx, ty, tz = x
        pts_t = s * pts_pred_rotated + np.array([tx, ty, tz])
        d_pred_to_gt, _ = tree_gt.query(pts_t)
        tree_pred_tmp = cKDTree(pts_t)
        d_gt_to_pred, _ = tree_pred_tmp.query(pts_gt)
        return (np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2)) / 2.0

    scale_lo = 1.0 - max_scale_delta if max_scale_delta > 0 else 1.0
    scale_hi = 1.0 + max_scale_delta if max_scale_delta > 0 else 1.0
    bounds = [
        (scale_lo, scale_hi),
        (-max_translate, max_translate),
        (-max_translate, max_translate),
        (-max_translate, max_translate),
    ]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 50, 'ftol': 1e-10})

    return result.x[0], result.x[1:]


# ==========================================================================
# IoU — voxel (surface voxelization + fill) and CSG (manifold3d boolean)
# ==========================================================================
def _voxelize_and_fill(mesh, resolution: int = EXTERIOR_VOXEL_RESOLUTION,
                       state: Optional[_MeshVoxelState] = None,
                       world_bounds=None):
    """Voxelize mesh surface, fill interior, and remap onto a fixed cubic grid.

    Uses trimesh surface voxelization + scipy ``binary_fill_holes`` instead of
    ray-parity containment, so meshes with internal overlapping surfaces
    (unfused assemblies) are handled regardless of watertight status.

    ``world_bounds`` ``(min_xyz, max_xyz)`` describes the world-space cube the
    output grid should cover; when ``None`` the canonical ``[-0.5, 0.5]^3`` cube
    is used (valid only for normalized meshes). Returns a ``(resolution,)*3``
    boolean array. Raises ``RuntimeError`` if voxel state is unavailable.
    """
    np = _np()
    if state is None:
        state = _build_mesh_voxel_state(mesh, resolution)
    if state is None:
        raise RuntimeError("voxelize+fill state unavailable")

    # Crop pad so indices line up with vox.transform (unpadded origin/pitch).
    p = _EXTERIOR_PAD
    filled_unpadded = state.filled_padded[p:-p, p:-p, p:-p]

    if world_bounds is None:
        fixed_origin = np.array([-0.5, -0.5, -0.5], dtype=float)
        fixed_extent = 1.0
    else:
        wmin = np.asarray(world_bounds[0], dtype=float)
        wmax = np.asarray(world_bounds[1], dtype=float)
        fixed_extent = float(np.max(wmax - wmin))
        if fixed_extent <= 0:
            fixed_extent = 1.0
        fixed_center = (wmin + wmax) / 2.0
        fixed_origin = fixed_center - fixed_extent / 2.0

    fixed_pitch = fixed_extent / resolution
    result = np.zeros((resolution, resolution, resolution), dtype=bool)

    filled_indices = np.argwhere(filled_unpadded)
    if len(filled_indices) == 0:
        return result

    # trimesh maps index i to corner origin + i*scale; center = origin + (i + 0.5) * scale.
    origin = state.vox_origin_unpadded
    scale = state.vox_pitch
    world_coords = (filled_indices + 0.5) * scale + origin

    fixed_indices = np.floor((world_coords - fixed_origin) / fixed_pitch).astype(int)

    valid = np.all(fixed_indices >= 0, axis=1) & np.all(fixed_indices < resolution, axis=1)
    vi = fixed_indices[valid]
    result[vi[:, 0], vi[:, 1], vi[:, 2]] = True

    return result


def intersection_over_union(mesh_pred, mesh_gt,
                            resolution: int = IOU_VOXEL_RESOLUTION,
                            skip_normalize: bool = False,
                            pred_state: Optional[_MeshVoxelState] = None,
                            gt_state: Optional[_MeshVoxelState] = None,
                            world_bounds=None):
    """Estimate volumetric IoU via surface voxelization + interior fill (128³).

    Uses trimesh voxelization + scipy ``binary_fill_holes`` instead of
    ray-parity containment (``mesh.contains``), so meshes with internal
    overlapping surfaces (unfused assemblies) are handled correctly. Inputs are
    independently normalized to ``[-0.5, 0.5]^3`` unless ``skip_normalize=True``;
    when ``skip_normalize`` and ``world_bounds`` is ``None`` the evaluation cube
    is auto-derived from the union of the two input bboxes. Falls back to
    ray-parity containment on failure; returns ``None`` if that also fails.
    """
    np = _np()
    if not skip_normalize:
        mesh_pred = normalize_mesh(mesh_pred)
        mesh_gt = normalize_mesh(mesh_gt)
    elif world_bounds is None:
        lo = np.minimum(mesh_pred.bounds[0], mesh_gt.bounds[0])
        hi = np.maximum(mesh_pred.bounds[1], mesh_gt.bounds[1])
        world_bounds = (lo, hi)

    try:
        grid_pred = _voxelize_and_fill(mesh_pred, resolution, state=pred_state,
                                       world_bounds=world_bounds)
        grid_gt = _voxelize_and_fill(mesh_gt, resolution, state=gt_state,
                                     world_bounds=world_bounds)

        intersection = np.logical_and(grid_pred, grid_gt).sum()
        union = np.logical_or(grid_pred, grid_gt).sum()

        if union == 0:
            return 0.0
        return float(intersection / union)
    except Exception as e:
        logger.warning(f"Voxel IoU (voxelize+fill) failed: {e}, "
                       "falling back to ray-parity containment")

    # Fallback: original ray-parity approach via mesh.contains()
    try:
        if world_bounds is None:
            wmin = np.array([-0.5, -0.5, -0.5], dtype=float)
            wmax = np.array([0.5, 0.5, 0.5], dtype=float)
        else:
            wmin = np.asarray(world_bounds[0], dtype=float)
            wmax = np.asarray(world_bounds[1], dtype=float)
        wcenter = (wmin + wmax) / 2.0
        wextent = float(np.max(wmax - wmin))
        if wextent <= 0:
            wextent = 1.0
        pitch = wextent / resolution
        axis_lo = wcenter - wextent / 2.0 + pitch / 2.0
        axis_hi = wcenter + wextent / 2.0 - pitch / 2.0
        lin_x = np.linspace(axis_lo[0], axis_hi[0], resolution)
        lin_y = np.linspace(axis_lo[1], axis_hi[1], resolution)
        lin_z = np.linspace(axis_lo[2], axis_hi[2], resolution)
        X, Y, Z = np.meshgrid(lin_x, lin_y, lin_z, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        grid_pred = mesh_pred.contains(points).reshape(resolution, resolution, resolution)
        grid_gt = mesh_gt.contains(points).reshape(resolution, resolution, resolution)

        intersection = np.logical_and(grid_pred, grid_gt).sum()
        union = np.logical_or(grid_pred, grid_gt).sum()

        if union == 0:
            return 0.0
        return float(intersection / union)
    except Exception as e:
        logger.warning(f"Voxel IoU fallback also failed: {e}")
        return None


def _fix_winding_if_needed(mesh):
    """Fix face winding on a watertight mesh so it qualifies as a volume.

    Returns ``(fixed_mesh, was_fixed)`` where *was_fixed* is True when
    ``trimesh.repair.fix_normals`` was applied to make the mesh a valid volume.
    """
    trimesh = _trimesh()
    if mesh.is_volume:
        return mesh, False
    fixed = mesh.copy()
    trimesh.repair.fix_normals(fixed)
    return fixed, True


def _to_manifold(mesh):
    """Convert a trimesh mesh to a manifold3d Manifold object."""
    np = _np()
    manifold3d = require("manifold3d", "geometry", "IoU (CSG)")
    return manifold3d.Manifold(mesh=manifold3d.Mesh(
        vert_properties=np.array(mesh.vertices, dtype=np.float32),
        tri_verts=np.array(mesh.faces, dtype=np.uint32)))


def iou_csg(mesh_pred, mesh_gt, skip_normalize: bool = False):
    """Compute IoU via mesh boolean operations (cad-recode style).

    Uses manifold3d for boolean operations (auto-cleans non-manifold input).
    Inputs should be closed meshes (no open edges). If ``skip_normalize`` is
    False, both meshes are independently normalized to ``[-0.5, 0.5]^3`` first.

    Returns ``(iou_value, pred_winding_fixed, gt_winding_fixed)``.
    """
    np = _np()
    if not skip_normalize:
        mesh_pred = normalize_mesh(mesh_pred)
        mesh_gt = normalize_mesh(mesh_gt)

    mesh_pred, pred_fixed = _fix_winding_if_needed(mesh_pred)
    mesh_gt, gt_fixed = _fix_winding_if_needed(mesh_gt)

    try:
        m_pred = _to_manifold(mesh_pred)
        m_gt = _to_manifold(mesh_gt)

        m_inter = m_pred ^ m_gt  # intersection
        intersection_volume = abs(m_inter.volume())

        pred_volume = abs(m_pred.volume())
        gt_volume = abs(m_gt.volume())
        union_volume = pred_volume + gt_volume - intersection_volume

        if union_volume <= 0:
            return 0.0, pred_fixed, gt_fixed
        return float(np.clip(intersection_volume / union_volume, 0.0, 1.0)), pred_fixed, gt_fixed
    except Exception as e:
        logger.warning(f"iou_csg failed: {e}")
        return 0.0, pred_fixed, gt_fixed


# ==========================================================================
# Topology metrics (shared with metrics/topology.py)
# ==========================================================================
def compute_topology_metrics(mesh) -> dict:
    """Compute fine-grained topology metrics for a mesh.

    Returns dict with:
        open_edge_ratio: fraction of unique edges shared by only 1 face.
        inverted_normal_ratio: fraction of manifold edges with inconsistent winding.
        non_manifold_edge_ratio: fraction of unique edges shared by >= 3 faces.
    """
    np = _np()
    from collections import Counter

    edges_sorted = np.sort(mesh.edges, axis=1)
    edge_tuples = list(map(tuple, edges_sorted))
    edge_counts = Counter(edge_tuples)

    total_unique = len(edge_counts)
    if total_unique == 0:
        return {"open_edge_ratio": 0.0, "inverted_normal_ratio": 0.0,
                "non_manifold_edge_ratio": 0.0}

    open_edges = sum(1 for c in edge_counts.values() if c == 1)
    non_manifold_edges = sum(1 for c in edge_counts.values() if c >= 3)

    # Winding consistency: for edges shared by exactly 2 faces, check that the
    # two half-edges have opposite direction (i.e. [a,b] and [b,a]). Same
    # direction means inconsistent winding (inverted normal).
    manifold_edge_set = {k for k, c in edge_counts.items() if c == 2}
    inverted = 0
    if manifold_edge_set:
        from collections import defaultdict
        edge_directions = defaultdict(list)
        for e in mesh.edges:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key in manifold_edge_set:
                edge_directions[key].append((e[0], e[1]))
        for key, dirs in edge_directions.items():
            if len(dirs) == 2:
                # Consistent winding: [a,b] and [b,a] (opposite directions)
                # Inverted: [a,b] and [a,b] (same direction)
                if dirs[0] == dirs[1]:
                    inverted += 1

    return {
        "open_edge_ratio": round(open_edges / total_unique, 6),
        "inverted_normal_ratio": round(inverted / total_unique, 6),
        "non_manifold_edge_ratio": round(non_manifold_edges / total_unique, 6),
    }


# ==========================================================================
# Master entry point — normalize -> align -> sample once -> score
# ==========================================================================
def align_and_compute(
    mesh_pred,
    mesh_gt,
    *,
    sampling_mode: str = "surface",
    skip_normalize_and_scale: bool = False,
    compute_iou_variant: Optional[str] = "voxel",
):
    """Align pred to GT and compute all geometry metrics on one shared cloud.

    Pipeline (canonical normalize -> align -> sample-once -> score):
      1. Topology on raw pred & GT, deriving open-edge gates.
      2. Normalize both (or center-only when ``skip_normalize_and_scale``).
      3. Align pred to GT with ``pca+discrete`` + bounded scale/translate refine.
      4. Sample one shared 8192-point cloud per mesh (surface or exterior) that
         feeds CD, both F-scores and NC.
      5. CD / F@.05 / F@.01 / NC on the aligned clouds.
      6. IoU (gated on both meshes having zero open edges); ``compute_iou_variant``
         selects ``'voxel'`` (128³), ``'csg'`` (manifold boolean) or ``None``.

    Args:
        sampling_mode: ``'surface'`` (all triangles) or ``'exterior'`` (voxel
            flood-fill restricts CD/F/NC sampling to the outer skin).
        skip_normalize_and_scale: center-only + scale refinement disabled
            (text2cad, where exact dimensions make absolute scale meaningful).
        compute_iou_variant: ``'voxel'`` | ``'csg'`` | ``None``.

    Returns ``(metrics, align_transform_4x4, aligned_pred)`` where ``metrics``
    carries the RAW geometry keys (``chamfer_distance``, ``f_score_005``,
    ``f_score_001``, ``normal_consistency``, ``iou``) plus diagnostic and
    topology-raw keys for reuse. ``iou`` is ``None`` when the gate fails or the
    requested variant could not be computed; ``aligned_gt`` is included in the
    metrics dict for part.py to reuse.
    """
    np = _np()
    cKDTree = _ckdtree()
    result: dict = {}

    # --- 1. Topology on raw meshes + open-edge gates ---
    pred_topo = compute_topology_metrics(mesh_pred)
    gt_topo = compute_topology_metrics(mesh_gt)
    result["pred_open_edge_ratio"] = pred_topo["open_edge_ratio"]
    result["pred_inverted_normal_ratio"] = pred_topo["inverted_normal_ratio"]
    result["pred_non_manifold_edge_ratio"] = pred_topo["non_manifold_edge_ratio"]
    result["gt_open_edge_ratio"] = gt_topo["open_edge_ratio"]
    result["gt_inverted_normal_ratio"] = gt_topo["inverted_normal_ratio"]
    result["gt_non_manifold_edge_ratio"] = gt_topo["non_manifold_edge_ratio"]

    pred_has_open_edges = pred_topo["open_edge_ratio"] > 0
    gt_has_open_edges = gt_topo["open_edge_ratio"] > 0
    both_no_open_edges = not pred_has_open_edges and not gt_has_open_edges

    # Record pred normalization params for composing the full transform later.
    pred_bounds = mesh_pred.bounds
    pred_center = (pred_bounds[0] + pred_bounds[1]) / 2.0
    result["skip_normalize_and_scale"] = skip_normalize_and_scale

    # --- 2. Normalize (or center-only) + refinement bounds + metric_scale ---
    if skip_normalize_and_scale:
        mesh_gt_n = center_mesh(mesh_gt)
        mesh_pred_n = center_mesh(mesh_pred)
        pred_scale = 1.0
        refine_max_scale_delta = 0.0
        gt_extent = float(np.max(mesh_gt_n.bounds[1] - mesh_gt_n.bounds[0]))
        pred_extent = float(np.max(mesh_pred_n.bounds[1] - mesh_pred_n.bounds[0]))
        refine_max_translate = max(gt_extent, pred_extent) * 0.4
        metric_scale = gt_extent if gt_extent > 0 else 1.0
    else:
        mesh_gt_n = normalize_mesh(mesh_gt)
        mesh_pred_n = normalize_mesh(mesh_pred)
        pred_scale = float(np.max(pred_bounds[1] - pred_bounds[0]))
        if pred_scale <= 0:
            pred_scale = 1.0
        refine_max_scale_delta = 0.3
        refine_max_translate = 0.2
        metric_scale = 1.0

    # --- 3. Align pred to GT (identity transform on failure) ---
    best_transform = np.eye(4)
    try:
        mesh_pred_n, mesh_gt_n, best_transform = align_meshes(
            mesh_pred_n, mesh_gt_n,
            method="pca+discrete",
            max_scale_delta=refine_max_scale_delta,
            max_translate=refine_max_translate)
    except Exception as e:
        logger.warning(f"Alignment failed, using unaligned meshes: {e}")

    # Pred voxel state must be built *after* alignment because pred has been
    # rotated/scaled by the alignment transform.
    if sampling_mode == "exterior":
        pred_voxel_state = (None if pred_has_open_edges
                            else _build_mesh_voxel_state(mesh_pred_n))
        gt_voxel_state = (None if gt_has_open_edges
                          else _build_mesh_voxel_state(mesh_gt_n))
    else:
        pred_voxel_state = None
        gt_voxel_state = None

    # --- 4. Sample one shared cloud per mesh; feeds CD/F/NC ---
    fidx_pred_cached = None
    fidx_gt_cached = None
    pred_sampling_info: dict = {"fallback": None, "kept_ratio": 1.0}
    gt_sampling_info: dict = {"fallback": None, "kept_ratio": 1.0}
    result["sampling_mode"] = sampling_mode
    n_points = 8192
    try:
        if sampling_mode == "exterior":
            pts_pred, fidx_pred_cached, pred_sampling_info = _sample_points_exterior(
                mesh_pred_n, n_points,
                has_open_edges=pred_has_open_edges,
                state=pred_voxel_state)
            pts_gt, fidx_gt_cached, gt_sampling_info = _sample_points_exterior(
                mesh_gt_n, n_points,
                has_open_edges=gt_has_open_edges,
                state=gt_voxel_state)
        else:
            pts_pred = _sample_points(mesh_pred_n, n_points)
            pts_gt = _sample_points(mesh_gt_n, n_points)
        result["pred_sampling_fallback"] = pred_sampling_info["fallback"]
        result["gt_sampling_fallback"] = gt_sampling_info["fallback"]
        result["pred_exterior_kept_ratio"] = round(pred_sampling_info["kept_ratio"], 4)
        result["gt_exterior_kept_ratio"] = round(gt_sampling_info["kept_ratio"], 4)
        tree_pred = cKDTree(pts_pred)
        tree_gt = cKDTree(pts_gt)
        d_pred_to_gt, _ = tree_gt.query(pts_pred)
        d_gt_to_pred, _ = tree_pred.query(pts_gt)

        # --- 5. CD / Hausdorff / F-scores ---
        result["chamfer_distance"] = round(float(
            (np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2)) / 2.0), 6)

        result["hausdorff_distance"] = round(float(
            max(np.max(d_pred_to_gt), np.max(d_gt_to_pred))), 6)

        thr_005 = 0.05 * metric_scale
        thr_001 = 0.01 * metric_scale
        result["f_score_threshold_ref_scale"] = round(float(metric_scale), 6)

        precision = float(np.mean(d_pred_to_gt < thr_005))
        recall = float(np.mean(d_gt_to_pred < thr_005))
        fs = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result["f_score_005"] = round(fs, 4)
        result["f_precision_005"] = round(precision, 4)
        result["f_recall_005"] = round(recall, 4)

        precision_001 = float(np.mean(d_pred_to_gt < thr_001))
        recall_001 = float(np.mean(d_gt_to_pred < thr_001))
        fs_001 = 2 * precision_001 * recall_001 / (precision_001 + recall_001) if (precision_001 + recall_001) > 0 else 0.0
        result["f_score_001"] = round(fs_001, 4)
        result["f_precision_001"] = round(precision_001, 4)
        result["f_recall_001"] = round(recall_001, 4)
    except Exception as e:
        logger.warning(f"CD/F-score failed: {e}")

    # --- NC: reuse the exterior cloud when available so NC uses the same points ---
    try:
        if sampling_mode == "exterior" and fidx_pred_cached is not None and fidx_gt_cached is not None:
            pts_pred_n = pts_pred
            pts_gt_n = pts_gt
            normals_pred = mesh_pred_n.face_normals[fidx_pred_cached]
            normals_gt = mesh_gt_n.face_normals[fidx_gt_cached]
        else:
            pts_pred_n, normals_pred = _sample_points_with_normals(mesh_pred_n, n_points)
            pts_gt_n, normals_gt = _sample_points_with_normals(mesh_gt_n, n_points)

        # pred->gt (precision): are the normals I have correct?
        tree_gt_n = cKDTree(pts_gt_n)
        _, idx_p2g = tree_gt_n.query(pts_pred_n)
        dots_p2g = np.abs(np.sum(normals_pred * normals_gt[idx_p2g], axis=1))
        nc_precision = float(np.mean(dots_p2g))

        # gt->pred (recall): do I cover all GT normals?
        tree_pred_n = cKDTree(pts_pred_n)
        _, idx_g2p = tree_pred_n.query(pts_gt_n)
        dots_g2p = np.abs(np.sum(normals_gt * normals_pred[idx_g2p], axis=1))
        nc_recall = float(np.mean(dots_g2p))

        result["normal_consistency"] = round((nc_precision + nc_recall) / 2.0, 4)
        result["nc_precision"] = round(nc_precision, 4)
        result["nc_recall"] = round(nc_recall, 4)
    except Exception as e:
        logger.warning(f"Normal consistency failed: {e}")

    # --- 6. IoU (gated on both meshes having zero open edges) ---
    # Under skip_normalize_and_scale the meshes keep absolute world units so the
    # voxel grid tracks the union bbox instead of the [-0.5, 0.5]^3 cube.
    if skip_normalize_and_scale:
        iou_world_bounds = (
            np.minimum(mesh_pred_n.bounds[0], mesh_gt_n.bounds[0]),
            np.maximum(mesh_pred_n.bounds[1], mesh_gt_n.bounds[1]),
        )
    else:
        iou_world_bounds = None

    iou_value = None
    gate_ok = both_no_open_edges and compute_iou_variant is not None
    if gate_ok and compute_iou_variant == "voxel":
        try:
            iou_val = intersection_over_union(
                mesh_pred_n, mesh_gt_n, skip_normalize=True,
                pred_state=pred_voxel_state, gt_state=gt_voxel_state,
                world_bounds=iou_world_bounds)
            if iou_val is not None:
                iou_value = round(iou_val, 4)
                result["iou_voxel"] = iou_value
            else:
                logger.warning("IoU voxel: both primary and fallback methods failed")
        except Exception as e:
            logger.warning(f"IoU voxel failed: {e}")
    elif gate_ok and compute_iou_variant == "csg":
        try:
            csg_val, pred_winding_fixed, gt_winding_fixed = iou_csg(
                mesh_pred_n, mesh_gt_n, skip_normalize=True)
            iou_value = round(csg_val, 4)
            result["iou_csg"] = iou_value
            result["iou_csg_pred_winding_fixed"] = pred_winding_fixed
            result["iou_csg_gt_winding_fixed"] = gt_winding_fixed
        except Exception as e:
            logger.warning(f"IoU CSG failed: {e}")
    elif compute_iou_variant is not None:
        logger.warning(
            f"Skipping IoU ({compute_iou_variant}): pred_open_edges={pred_has_open_edges}, "
            f"gt_open_edges={gt_has_open_edges}")

    # Canonical raw geometry key (task-appropriate variant under a single key).
    result["iou"] = iou_value

    # --- Compose full transform: original mesh coords -> aligned coords ---
    # normalize: v' = (v - center) / scale ; align: v'' = best_transform @ v'
    norm_mat = np.eye(4)
    norm_mat[:3, :3] = np.eye(3) / pred_scale
    norm_mat[:3, 3] = -pred_center / pred_scale
    align_transform_4x4 = best_transform @ norm_mat

    # Expose aligned GT alongside aligned pred so part.py can reuse them.
    result["aligned_gt"] = mesh_gt_n

    return result, align_transform_4x4, mesh_pred_n


# ==========================================================================
# Geometry metric bucket
# ==========================================================================
def _load_mesh(path):
    """Load a mesh via compile.step_mesh.load_mesh (lazy import).

    STEP via OCP tessellation; STL/OBJ via trimesh. Imported lazily so this
    module imports cleanly even before its sibling exists.
    """
    from ..compile.step_mesh import load_mesh
    return load_mesh(path)


def _geometry_params_for_task(task: str) -> Tuple[str, bool, Optional[str]]:
    """Pick (sampling_mode, skip_normalize_and_scale, iou_variant) for a task.

    - text-to-3d (parametric; geometry only runs in parametric mode):
        surface sampling, center-only (exact dimensions), exact CSG IoU.
    - image-to-3d: surface sampling, normalize+scale, voxel IoU.
    - assembly-3d: exterior sampling (unfused assemblies), normalize+scale,
        voxel IoU.
    """
    if task == "text-to-3d":
        return "surface", True, "csg"
    if task == "image-to-3d":
        return "surface", False, "voxel"
    if task == "assembly-3d":
        return "exterior", False, "voxel"
    # Unknown task: default to the image-to-3d-style configuration.
    return "surface", False, "voxel"


# Raw geometry keys the aggregation understands (see metrics.base.METRIC_SPECS).
_GEOMETRY_KEYS = (
    "chamfer_distance",
    "f_score_005",
    "f_score_001",
    "normal_consistency",
    "iou",
)


class _GeometryBucket(MetricBucket):
    bucket = "geometry"
    requires: set[str] = {"mesh"}

    def score(self, ctx: ScoreContext) -> dict:
        # Surface a clear MissingDependencyError early if the extra is absent.
        _np()
        _trimesh()
        _ckdtree()

        empty = {k: None for k in _GEOMETRY_KEYS}

        pred_path = ctx.compiled.get("stl")
        if not pred_path:
            return empty
        gt_path = ctx.case.gt_step or ctx.case.gt_mesh
        if not gt_path:
            return empty

        mesh_pred = _load_mesh(str(pred_path))
        mesh_gt = _load_mesh(str(gt_path))
        if mesh_pred is None or mesh_gt is None:
            return empty

        sampling_mode, skip_norm, iou_variant = _geometry_params_for_task(ctx.task)

        metrics, align_transform_4x4, aligned_pred = align_and_compute(
            mesh_pred, mesh_gt,
            sampling_mode=sampling_mode,
            skip_normalize_and_scale=skip_norm,
            compute_iou_variant=iou_variant,
        )

        # Store alignment products for part.py to reuse (paper: one transform
        # per side). aligned_gt lives inside the metrics dict.
        ctx.shared["geometry"] = {
            "align_transform_4x4": align_transform_4x4,
            "aligned_pred": aligned_pred,
            "aligned_gt": metrics.get("aligned_gt"),
            "gt_mesh": mesh_gt,
            "pred_mesh": mesh_pred,
            "sampling_mode": sampling_mode,
            "skip_normalize_and_scale": skip_norm,
            "iou_variant": iou_variant,
        }

        # Return only the canonical raw geometry keys (iou=None when gated out).
        return {k: metrics.get(k) for k in _GEOMETRY_KEYS}


BUCKET = _GeometryBucket()
