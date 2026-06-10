"""Pyrender multiview renderer for the Judge bucket (extra: ``render``).

Ports the canonical render geometry from the source ``render/mesh_utils.py`` +
``render/render.py``:

  * ``VIEW_ANGLES`` — the 4 tetrahedral judge views (elev, azim).
  * ``ORIENT_MAT`` — the (x,y,z)->(z,x,y) orientation applied to every mesh so
    single-view / multiview / OCC / Blender renders are pixel-aligned.
  * camera helpers ``orient_mesh_for_render``, ``eye_from_angles``,
    ``fit_perspective_camera`` (projected-bbox fit).
  * ``PyrenderSceneConfig`` — silver material, white bg, key + fill lights.

The public entry point is :func:`render_multiview`:

    render_multiview(mesh_or_step_path, output_dir, *, n_views=4) -> list[str]

It loads the mesh via :func:`p3dbench.compile.step_mesh.load_mesh` (STEP via OCP
tessellation; STL/OBJ/PLY via trimesh), orients it, fits the perspective camera
ONCE at view 0 (elev 30 / azim 45) so the projected scale is constant across the
set, then renders ``n_views`` PNGs (``view_000.png`` .. ``view_003.png``).

Returns ``[]`` on any failure or when pyrender / PIL are not installed (the
Judge bucket then simply skips this case rather than crashing the run).

Heavy deps (numpy / trimesh / pyrender / PIL) are imported lazily through
``utils.require`` / guarded imports so importing this module never pulls them in.
``PYOPENGL_PLATFORM=egl`` is set before pyrender is ever imported, as headless
EGL rendering requires.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

from ..utils import require

logger = logging.getLogger(__name__)

# Must be set before any OpenGL / pyrender import for headless EGL rendering.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_RENDER_EXTRA = "render"

# ---------------------------------------------------------------------------
# Canonical constants (ported verbatim from render/mesh_utils.py + render.py)
# ---------------------------------------------------------------------------

# 4 tetrahedral judge view angles (elevation_deg, azimuth_deg):
# 2 top diagonals looking down + 2 bottom diagonals looking up at +-30 deg
# (matching the single-view elevation so view 0 aligns with the single render).
# Azimuths staggered (top 45/225, bottom 135/315) so each camera sits on a
# distinct horizontal direction -- top, bottom, and all four sides of the
# object end up visible across the set, instead of only the top half-sphere.
VIEW_ANGLES = [
    (30, 45),    # 0: top-front-right diagonal (above, looking down)
    (30, 225),   # 1: top-back-left diagonal (above, looking down)
    (-30, 135),  # 2: bottom-back-right diagonal (below, looking up)
    (-30, 315),  # 3: bottom-front-left diagonal (below, looking up)
]

DEFAULT_RENDER_RESOLUTION = 1024
PYRENDER_DEFAULT_YFOV = 0.6
PYRENDER_FIT_PADDING = 1.03

# Orientation matrix: (x,y,z) -> (z,x,y). Applies Rx(90) then Rz(90) to align
# renders across all pipelines. Stored as a plain nested list so importing this
# module needs no numpy; promoted to an ndarray lazily by ``_orient_mat()``.
ORIENT_MAT = [
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
]

# Thread-safe lock for the pyrender EGL renderer (not thread-safe). Kept for
# correctness even in the single-process demo: any callers sharing this process
# must not enter the offscreen renderer concurrently.
PYRENDER_EGL_RENDER_LOCK = threading.Lock()


def _np():
    """Lazy numpy handle (extra: ``render``)."""
    return require("numpy", _RENDER_EXTRA, "Multiview rendering")


def _orient_mat():
    """Return ``ORIENT_MAT`` as a float numpy array."""
    np = _np()
    return np.array(ORIENT_MAT, dtype=float)


# ---------------------------------------------------------------------------
# Camera helpers (ported from render/mesh_utils.py)
# ---------------------------------------------------------------------------

def orient_mesh_for_render(mesh):
    """Rotate a mesh to keep single-view / multi-view / judge renders aligned.

    Applies Rx(90) then Rz(90) so (x,y,z) maps to (z, x, y). Returns a copy.
    """
    np = _np()
    orient = np.eye(4)
    orient[:3, :3] = _orient_mat()
    mesh = mesh.copy()
    mesh.apply_transform(orient)
    return mesh


def eye_from_angles(center, distance: float, elev_deg: float, azim_deg: float):
    """Compute camera eye position from elevation / azimuth (up = +Z)."""
    np = _np()
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)
    return center + distance * np.array([
        np.cos(elev) * np.sin(azim),
        np.cos(elev) * np.cos(azim),
        np.sin(elev),
    ])


def look_at_axes(eye, center):
    """Return forward / right / up axes for a look-at camera (up = +Z)."""
    np = _np()
    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return forward, right, up


def look_at_pose(eye, center):
    """Build a 4x4 camera pose for a look-at camera."""
    np = _np()
    forward, right, up = look_at_axes(eye, center)
    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = up
    cam_pose[:3, 2] = -forward
    cam_pose[:3, 3] = eye
    return cam_pose


def fit_perspective_camera(mesh, elev_deg: float, azim_deg: float,
                           yfov: float = PYRENDER_DEFAULT_YFOV,
                           aspect_ratio: float = 1.0,
                           padding: float = PYRENDER_FIT_PADDING):
    """Fit a perspective camera using projected bbox extents for tight framing.

    For each of the 8 oriented-bbox corners compute the distance that keeps it
    inside the frustum for the requested view direction; take the max and scale
    by ``padding``. Returns ``(center, distance, eye)``.
    """
    np = _np()
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0

    eye_unit = eye_from_angles(np.zeros(3), 1.0, elev_deg, azim_deg)
    view_dir = eye_unit / np.linalg.norm(eye_unit)
    eye = center + view_dir
    _, right, up = look_at_axes(eye, center)

    tan_y = np.tan(yfov / 2.0)
    tan_x = tan_y * aspect_ratio
    corners = np.array(list(itertools.product(*zip(bounds[0], bounds[1]))), dtype=float)
    offsets = corners - center

    view_offsets = offsets @ view_dir
    horiz = np.abs(offsets @ right) / tan_x
    vert = np.abs(offsets @ up) / tan_y
    required_distance = view_offsets + np.maximum(horiz, vert)
    distance = float(required_distance.max() * padding)
    eye = eye_from_angles(center, distance, elev_deg, azim_deg)
    return center, distance, eye


# ---------------------------------------------------------------------------
# Pyrender scene (ported from render/render.py)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PyrenderSceneConfig:
    """Configuration for building a pyrender scene (silver material, white bg)."""
    base_color: tuple = (0.6, 0.6, 0.6, 1.0)
    metallic: float = 0.2
    roughness: float = 0.6
    ambient: tuple = (0.15, 0.15, 0.15)
    bg_color: tuple = (255, 255, 255, 255)
    camera_type: str = "perspective"  # "perspective" or "orthographic"
    yfov: float = PYRENDER_DEFAULT_YFOV
    ortho_padding: float = 1.4
    elev_deg: float = 30.0
    azim_deg: float = 45.0
    key_light_intensity: float = 2.5
    fill_light_intensity: float = 1.5
    resolution: tuple = (DEFAULT_RENDER_RESOLUTION, DEFAULT_RENDER_RESOLUTION)


def build_pyrender_scene(mesh, config: Optional[PyrenderSceneConfig] = None,
                         eye=None, center=None, distance=None):
    """Build a fully assembled pyrender Scene ready for OffscreenRenderer.

    Returns ``(scene, resolution_tuple)``. ``eye``/``center``/``distance`` may be
    pre-computed (so a multiview set shares one fit); otherwise they are derived
    from the config view via :func:`fit_perspective_camera`.
    """
    pyrender = require("pyrender", _RENDER_EXTRA, "Multiview rendering")
    np = _np()

    if config is None:
        config = PyrenderSceneConfig()

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=list(config.base_color),
        metallicFactor=config.metallic,
        roughnessFactor=config.roughness,
    )

    scene = pyrender.Scene(bg_color=list(config.bg_color),
                           ambient_light=list(config.ambient))
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False, material=material))

    # Camera position
    if center is None or distance is None or eye is None:
        center, distance, eye = fit_perspective_camera(
            mesh, elev_deg=config.elev_deg, azim_deg=config.azim_deg,
            yfov=config.yfov)

    cam_pose = look_at_pose(eye, center)

    if config.camera_type == "orthographic":
        bounds = mesh.bounds
        half_diag = np.linalg.norm(bounds[1] - bounds[0]) / 2.0
        xmag = ymag = half_diag * config.ortho_padding
        camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)
    else:
        camera = pyrender.PerspectiveCamera(yfov=config.yfov)

    scene.add(camera, pose=cam_pose)

    # Key light from camera direction.
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0],
                                      intensity=config.key_light_intensity)
    scene.add(light, pose=cam_pose)

    # Fill light from above.
    top_pose = np.eye(4)
    top_pose[:3, 3] = center + np.array([0, 0, distance])
    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0],
                                           intensity=config.fill_light_intensity)
    scene.add(fill_light, pose=top_pose)

    return scene, config.resolution


def render_scene_to_file(scene, output_path: str, resolution: tuple) -> bool:
    """Render a pyrender scene to a PNG file (holds the EGL render lock)."""
    pyrender = require("pyrender", _RENDER_EXTRA, "Multiview rendering")
    Image = require("PIL.Image", _RENDER_EXTRA, "Multiview rendering")

    with PYRENDER_EGL_RENDER_LOCK:
        r = pyrender.OffscreenRenderer(*resolution)
        try:
            color, _ = r.render(scene)
            Image.fromarray(color).save(output_path)
            return True
        finally:
            r.delete()


def render_pyrender(mesh, output_path: str,
                    config: Optional[PyrenderSceneConfig] = None,
                    eye=None, center=None, distance=None) -> bool:
    """One-call pyrender rendering: build scene + render to file."""
    scene, res = build_pyrender_scene(mesh, config, eye=eye,
                                      center=center, distance=distance)
    return render_scene_to_file(scene, output_path, res)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_multiview(mesh_or_step_path, output_dir, *, n_views: int = 4) -> List[str]:
    """Render ``n_views`` judge views of a mesh / STEP file via pyrender (EGL).

    Loads the geometry with :func:`p3dbench.compile.step_mesh.load_mesh`, orients
    it with :func:`orient_mesh_for_render`, fits the perspective camera ONCE at
    view 0 (elev 30 / azim 45) so all views share the same projected scale, then
    renders ``view_000.png`` .. ``view_{n-1:03d}.png`` into ``output_dir``.

    Returns the list of written PNG paths, or ``[]`` on any failure (missing
    pyrender / PIL, mesh load failure, or a per-view render error). The Judge
    bucket treats ``[]`` as "skip this side".
    """
    output_dir = Path(output_dir)

    # Guard heavy deps -- a missing optional extra means "no renders", not a crash.
    try:
        require("pyrender", _RENDER_EXTRA, "Multiview rendering")
        require("PIL.Image", _RENDER_EXTRA, "Multiview rendering")
        _np()
        require("trimesh", _RENDER_EXTRA, "Multiview rendering")
    except Exception as exc:
        logger.warning("Multiview render unavailable (%s); returning []", exc)
        return []

    if n_views < 1:
        return []
    n_views = min(n_views, len(VIEW_ANGLES))

    # Shared STEP/STL/OBJ loader (one tessellation path keeps renders consistent
    # with the geometry metrics).
    from ..compile.step_mesh import load_mesh

    try:
        mesh = load_mesh(str(mesh_or_step_path))
    except Exception as exc:
        logger.warning("Failed to load mesh from %s: %s", mesh_or_step_path, exc)
        return []
    if mesh is None:
        logger.warning("Failed to load mesh from %s", mesh_or_step_path)
        return []

    try:
        mesh = orient_mesh_for_render(mesh)
        # Camera distance is fit once at view 0 (elev 30 / azim 45) and reused so
        # all views show the model at the same projected scale.
        center, distance, _ = fit_perspective_camera(
            mesh, elev_deg=30.0, azim_deg=45.0, yfov=PYRENDER_DEFAULT_YFOV)
    except Exception as exc:
        logger.warning("Failed to prepare camera for %s: %s", mesh_or_step_path, exc)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    config = PyrenderSceneConfig(
        resolution=(DEFAULT_RENDER_RESOLUTION, DEFAULT_RENDER_RESOLUTION))

    output_paths: List[str] = []
    for i in range(n_views):
        elev, azim = VIEW_ANGLES[i]
        out_path = str(output_dir / f"view_{i:03d}.png")
        try:
            eye = eye_from_angles(center, distance, elev, azim)
            ok = render_pyrender(mesh, out_path, config,
                                 eye=eye, center=center, distance=distance)
        except Exception as exc:
            logger.warning("Error rendering view %d for %s: %s",
                           i, mesh_or_step_path, exc)
            ok = False
        if not ok:
            # All-or-nothing: a ragged view set is never returned to the judge.
            logger.warning(
                "Multiview render incomplete for %s (view %d failed); returning []",
                mesh_or_step_path, i)
            return []
        output_paths.append(out_path)

    return output_paths
