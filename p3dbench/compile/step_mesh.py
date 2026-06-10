"""Shared STEP -> trimesh/STL meshing pipeline.

Per-solid pipeline (:func:`_load_step_via_ocp_aqd_per_solid`):
  For each ``TopAbs_SOLID`` in the STEP:
    1. OCP AQD: BRepMesh + ``AllowQualityDecrease=True``, with a
       ``ShapeUpgrade_ShapeDivideClosed`` retry on closed seam faces / timeouts.
    2. If the OCP mesh is closed (no open edges) -> use it.
    3. Else -> sandboxed gmsh (subprocess with a wall-clock timeout, because
       gmsh is initialized with ``interruptible=False`` and has been observed
       to infinite-loop on pathological CAD).
         - gmsh succeeds -> use its watertight mesh.
         - gmsh fails / times out -> keep the OCP open-edge mesh rather than
           drop the solid. Surface metrics (CD, NC, F-score, renders) stay
           correct; only voxel-IoU loses fidelity on that solid.
    4. If neither OCP nor gmsh produced anything -> drop that solid with a
       warning; surviving solids are still concatenated and returned.

Free shells / free faces (``TopAbs_SHELL`` with no SOLID ancestor;
``TopAbs_FACE`` with no SHELL ancestor) are meshed separately by
:func:`mesh_free_shells_and_faces` and concatenated onto the main mesh.
Closed-solid STEPs are unaffected because that function returns ``None`` when
there are no free units.

Optional heavy deps (numpy / trimesh / gmsh / OCP) are imported lazily via
``utils.require(..., extra="geometry")`` so importing this module never pulls
them in; a missing dep raises a clear :class:`MissingDependencyError`.
"""

from __future__ import annotations

import logging
import signal
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional, Tuple

from ..utils import require

logger = logging.getLogger(__name__)
_GMSH_SESSION_LOCK = threading.Lock()  # gmsh has process-global state.

DEFAULT_STEP_LINEAR_DEFLECTION = 0.01
DEFAULT_STEP_ANGULAR_DEFLECTION = 0.5
DEFAULT_STEP_GMSH_SIZE = 0.02
DEFAULT_OCP_MESH_TIMEOUT = 60  # seconds per solid (OCP AQD, SIGALRM)
# gmsh is initialized with interruptible=False and has no internal time limit;
# it has been observed to infinite-loop on pathological CAD (13h+ hangs). We
# run it in a subprocess per solid with this wall-clock budget — on timeout we
# fall back to the OCP open-edge mesh for that solid rather than drop the case.
DEFAULT_PER_SOLID_GMSH_TIMEOUT = 300  # seconds per solid

_GEOMETRY_EXTRA = "geometry"


def _np():
    """Lazy numpy handle."""
    return require("numpy", _GEOMETRY_EXTRA, "STEP meshing")


def _trimesh():
    """Lazy trimesh handle."""
    return require("trimesh", _GEOMETRY_EXTRA, "STEP meshing")


# ---------------------------------------------------------------------------
# STEP I/O helpers
# ---------------------------------------------------------------------------

def read_step_shape(step_path: Path):
    """Read a STEP file into an OCC ``TopoDS_Shape``."""
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if int(status) != 1:
        raise RuntimeError(f"STEP read failed for {step_path} with status {status}")

    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"Empty shape from STEP file: {step_path}")
    return shape


def iter_occ_solids(shape):
    """Yield all ``TopoDS_Solid`` children from an OCC shape."""
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        yield explorer.Current()
        explorer.Next()


# ---------------------------------------------------------------------------
# BBox / gmsh size helpers
# ---------------------------------------------------------------------------

def _get_step_bbox_max_extent(step_path: Path) -> Optional[float]:
    """Return the STEP bbox max extent in model units."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    np = _np()
    try:
        shape = read_step_shape(step_path)
        bbox = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        extents = np.array(
            [xmax - xmin, ymax - ymin, zmax - zmin],
            dtype=np.float64,
        )
        extents = np.maximum(extents, 0.0)
        if not np.all(np.isfinite(extents)):
            return None
        return float(np.max(extents))
    except Exception as e:
        logger.debug(f"Failed to compute STEP bbox for {step_path}: {e}")
        return None


def _resolve_gmsh_size(
    size_value: float | None,
    bbox_max_extent: Optional[float],
    size_mode: str,
) -> Optional[float]:
    """Resolve a requested gmsh size into an absolute model-space length."""
    if size_value is None:
        return None

    requested = float(size_value)
    if requested <= 0:
        return None

    mode = (size_mode or "relative").strip().lower()
    if mode == "absolute":
        return requested

    if mode != "relative":
        raise ValueError(f"Unsupported gmsh size mode: {size_mode}")

    if bbox_max_extent is None or bbox_max_extent <= 0:
        logger.debug(
            "STEP bbox unavailable or degenerate; falling back to absolute "
            f"gmsh size {requested}"
        )
        return requested

    return requested * bbox_max_extent


# ---------------------------------------------------------------------------
# Mesh quality check
# ---------------------------------------------------------------------------

def _mesh_has_open_edges(mesh) -> bool:
    """Return True if the mesh has any open (boundary) edges."""
    from collections import Counter

    np = _np()
    edges_sorted = np.sort(mesh.edges, axis=1)
    edge_counts = Counter(map(tuple, edges_sorted))
    return any(c == 1 for c in edge_counts.values())


# ---------------------------------------------------------------------------
# OCP AQD per-solid meshing (primary strategy)
# ---------------------------------------------------------------------------

class _MeshTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _MeshTimeout("BRepMesh timeout")


def _has_untriangulated_faces(shape) -> bool:
    """Return True if any face in *shape* lacks triangulation."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            return True
        exp.Next()
    return False


def _occ_meshed_shape_to_trimesh(shape):
    """Extract triangulation from an already-meshed OCC shape via StlAPI.

    Using ``StlAPI_Writer`` (binary STL round-trip) is more robust than
    manually collecting per-face triangulations because StlAPI handles face
    orientation and location transforms consistently.
    """
    from OCP.StlAPI import StlAPI_Writer

    trimesh = _trimesh()

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp_path = f.name

    try:
        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        if not writer.Write(shape, tmp_path):
            return None
        mesh = trimesh.load(tmp_path, file_type="stl")
        if isinstance(mesh, trimesh.Scene):
            geos = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geos:
                return None
            mesh = trimesh.util.concatenate(geos)
        if len(mesh.faces) == 0:
            return None
        return mesh
    except Exception:
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _ocp_aqd_mesh_solid(
    solid,
    linear_deflection: float = DEFAULT_STEP_LINEAR_DEFLECTION,
    angular_deflection: float = DEFAULT_STEP_ANGULAR_DEFLECTION,
    timeout: int = DEFAULT_OCP_MESH_TIMEOUT,
) -> Tuple[Optional[object], bool, bool]:
    """Mesh a single OCC solid with BRepMesh + AllowQualityDecrease.

    Returns ``(meshed_shape_or_None, used_divide_closed, all_faces_ok)``.

    Strategy:
      1. BRepMesh with ``AllowQualityDecrease=True``.
      2. If timeout or faces fail -> ShapeDivideClosed + re-mesh.
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.IMeshTools import IMeshTools_Parameters

    params = IMeshTools_Parameters()
    params.Deflection = linear_deflection
    params.Angle = angular_deflection
    params.InParallel = False
    params.AllowQualityDecrease = True

    # --- helpers to arm/disarm SIGALRM (main-thread only) ---
    is_main_thread = threading.current_thread() is threading.main_thread()

    def _arm_alarm():
        if is_main_thread and timeout > 0:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)

    def _disarm_alarm():
        if is_main_thread and timeout > 0:
            signal.alarm(0)

    # Step 1: try direct AQD mesh
    BRepTools.Clean_s(solid)
    _arm_alarm()
    try:
        BRepMesh_IncrementalMesh(solid, params).Perform()
        _disarm_alarm()
    except _MeshTimeout:
        _disarm_alarm()
        # Timeout -> go to ShapeDivideClosed path
        return _ocp_aqd_divide_closed_mesh(solid, params, timeout, is_main_thread)

    if not _has_untriangulated_faces(solid):
        return solid, False, True

    # Step 2: some faces failed -> ShapeDivideClosed + re-mesh
    return _ocp_aqd_divide_closed_mesh(solid, params, timeout, is_main_thread)


def _ocp_aqd_divide_closed_mesh(solid, params, timeout, is_main_thread):
    """Apply ShapeDivideClosed and re-mesh with AQD."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosed

    BRepTools.Clean_s(solid)
    div = ShapeUpgrade_ShapeDivideClosed(solid)
    div.SetNbSplitPoints(1)
    div.Perform()
    divided = div.Result()

    BRepTools.Clean_s(divided)

    if is_main_thread and timeout > 0:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)
    try:
        BRepMesh_IncrementalMesh(divided, params).Perform()
        if is_main_thread and timeout > 0:
            signal.alarm(0)
    except _MeshTimeout:
        if is_main_thread and timeout > 0:
            signal.alarm(0)
        return divided, True, False

    ok = not _has_untriangulated_faces(divided)
    return divided, True, ok


def _load_step_via_ocp_aqd_per_solid(
    step_path: Path,
    linear_deflection: float = DEFAULT_STEP_LINEAR_DEFLECTION,
    angular_deflection: float = DEFAULT_STEP_ANGULAR_DEFLECTION,
    timeout: int = DEFAULT_OCP_MESH_TIMEOUT,
    *,
    gmsh_size: float = DEFAULT_STEP_GMSH_SIZE,
    gmsh_size_mode: str = "relative",
    gmsh_timeout_s: int = DEFAULT_PER_SOLID_GMSH_TIMEOUT,
    occ_shape=None,
):
    """Per-solid meshing with OCP AQD + sandboxed gmsh fallback.

    For each solid in the STEP file:
      1. OCP AQD (BRepMesh + AllowQualityDecrease, ShapeDivideClosed retry).
      2. If the OCP mesh is closed (no open edges) -> use it.
      3. Else -> sandboxed gmsh for this solid (subprocess, wall-clock bounded).
           - gmsh succeeds -> prefer its watertight mesh.
           - gmsh fails / times out -> keep the OCP open-edge mesh anyway
             rather than drop the whole solid.
      4. If OCP produced nothing either -> drop this solid with a warning,
         surviving solids are still concatenated and returned.

    ``occ_shape`` lets the caller pass an already-read shape to avoid a
    redundant ``read_step_shape`` round-trip. Returns ``None`` only when
    *every* solid failed.
    """
    trimesh = _trimesh()

    if occ_shape is None:
        occ_shape = read_step_shape(step_path)
    solids = list(iter_occ_solids(occ_shape))

    if not solids:
        return None

    meshes: list = []

    for i, solid in enumerate(solids):
        meshed_shape, _used_dc, _all_ok = _ocp_aqd_mesh_solid(
            solid,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
            timeout=timeout,
        )
        ocp_mesh = (
            _occ_meshed_shape_to_trimesh(meshed_shape)
            if meshed_shape is not None else None
        )

        chosen = None
        if ocp_mesh is not None and not _mesh_has_open_edges(ocp_mesh):
            chosen = ocp_mesh
        else:
            # OCP mesh has open edges (or is missing) -> try gmsh in a
            # subprocess; on failure we still prefer the OCP open-edge mesh
            # over dropping the solid entirely.
            gmsh_mesh = _gmsh_mesh_occ_solid_sandboxed(
                solid,
                size_min=gmsh_size,
                size_max=gmsh_size,
                size_mode=gmsh_size_mode,
                timeout_s=gmsh_timeout_s,
            )
            if gmsh_mesh is not None and len(gmsh_mesh.faces) > 0:
                chosen = gmsh_mesh
            elif ocp_mesh is not None and len(ocp_mesh.faces) > 0:
                logger.warning(
                    f"Solid {i}/{len(solids)} of {step_path}: gmsh fallback "
                    f"failed; keeping OCP open-edge mesh "
                    f"(voxel-IoU on this solid will be affected)"
                )
                chosen = ocp_mesh
            else:
                logger.warning(
                    f"Solid {i}/{len(solids)} of {step_path}: both OCP AQD "
                    f"and gmsh failed; dropping this solid"
                )

        if chosen is not None:
            chosen.process(validate=True)
            chosen.merge_vertices()
            if len(chosen.faces) > 0:
                meshes.append(chosen)

    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


# ---------------------------------------------------------------------------
# Free shell / free face meshing
# ---------------------------------------------------------------------------

def _iter_free_shells_and_faces(shape):
    """Yield ``(unit, kind)`` for every free shell / free face in *shape*.

    Free shell = ``TopAbs_SHELL`` with no ``TopAbs_SOLID`` ancestor.
    Free face  = ``TopAbs_FACE`` with no ``TopAbs_SHELL`` ancestor.

    These are units that the per-solid main pipeline would silently skip
    (because it iterates ``TopAbs_SOLID`` only).
    """
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    m_shell = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_SHELL, TopAbs_SOLID, m_shell)
    for i in range(1, m_shell.Extent() + 1):
        if m_shell.FindFromIndex(i).Size() == 0:
            yield TopoDS.Shell_s(m_shell.FindKey(i)), "shell"

    m_face = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_FACE, TopAbs_SHELL, m_face)
    for i in range(1, m_face.Extent() + 1):
        if m_face.FindFromIndex(i).Size() == 0:
            yield TopoDS.Face_s(m_face.FindKey(i)), "face"


def mesh_free_shells_and_faces(
    shape,
    *,
    linear_deflection: float = DEFAULT_STEP_LINEAR_DEFLECTION,
    angular_deflection: float = DEFAULT_STEP_ANGULAR_DEFLECTION,
):
    """Mesh every free shell / free face in *shape* and return the merged trimesh.

    Returns ``None`` when there are no free shells/faces — callers can simply
    keep their main per-solid mesh in that case (zero-cost no-op for closed
    STEPs).

    Strategy: BRepMesh AQD on each unit, then StlAPI round-trip via
    :func:`_occ_meshed_shape_to_trimesh`. The output is intentionally
    non-watertight; closure is not required for surface bodies (e.g.
    eyeglass lenses imported as open shells).
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.IMeshTools import IMeshTools_Parameters

    trimesh = _trimesh()

    units = list(_iter_free_shells_and_faces(shape))
    if not units:
        return None

    params = IMeshTools_Parameters()
    params.Deflection = linear_deflection
    params.Angle = angular_deflection
    params.InParallel = False
    params.AllowQualityDecrease = True

    meshes: list = []
    for unit, kind in units:
        try:
            BRepTools.Clean_s(unit)
            BRepMesh_IncrementalMesh(unit, params).Perform()
            mesh = _occ_meshed_shape_to_trimesh(unit)
            if mesh is None or len(mesh.faces) == 0:
                logger.debug(f"Free {kind}: StlAPI round-trip empty")
                continue
            mesh.process(validate=True)
            mesh.merge_vertices()
            if len(mesh.faces) > 0:
                meshes.append(mesh)
        except Exception as e:
            logger.debug(f"Free {kind} meshing failed: {e}")

    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


# ---------------------------------------------------------------------------
# gmsh meshing
# ---------------------------------------------------------------------------

def load_step_as_gmsh_surface_mesh(
    step_path: Path,
    size_min: float | None,
    size_max: float | None,
    size_mode: str = "relative",
    verbose: bool = False,
):
    """Mesh a STEP file with gmsh and return a trimesh surface mesh.

    gmsh uses process-global state, so the full
    initialize/import/mesh/finalize session is serialized under a lock.

    Args:
        size_min: Characteristic length lower bound. In ``relative`` mode this
            is a fraction of the STEP bbox max extent.
        size_max: Characteristic length upper bound. In ``relative`` mode this
            is a fraction of the STEP bbox max extent.
        size_mode: ``relative`` (default) scales by the STEP bbox max extent;
            ``absolute`` forwards the values as-is.
    """
    gmsh = require("gmsh", _GEOMETRY_EXTRA, "STEP meshing (gmsh fallback)")
    np = _np()
    trimesh = _trimesh()

    bbox_max_extent = _get_step_bbox_max_extent(step_path)
    size_min = _resolve_gmsh_size(size_min, bbox_max_extent, size_mode)
    size_max = _resolve_gmsh_size(size_max, bbox_max_extent, size_mode)

    with _GMSH_SESSION_LOCK:
        initialized_here = False
        try:
            if not gmsh.isInitialized():
                gmsh.initialize(
                    argv=[],
                    readConfigFiles=False,
                    run=False,
                    interruptible=False,
                )
                initialized_here = True

            gmsh.clear()
            gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
            gmsh.model.add(step_path.stem)

            gmsh.model.occ.importShapes(str(step_path))
            gmsh.model.occ.synchronize()

            if size_min is not None:
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_min)
            if size_max is not None:
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_max)

            gmsh.model.mesh.generate(2)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            vertices = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
            tag_to_index = {int(tag): i for i, tag in enumerate(node_tags)}

            faces: list = []
            for dim, tag in gmsh.model.getEntities(2):
                types, _, node_tag_blocks = gmsh.model.mesh.getElements(dim, tag)
                for elem_type, node_tags_for_type in zip(types, node_tag_blocks):
                    if elem_type != 2:
                        continue
                    triangles = np.array(node_tags_for_type, dtype=np.int64).reshape(-1, 3)
                    faces.extend(
                        [tag_to_index[a], tag_to_index[b], tag_to_index[c]]
                        for a, b, c in triangles
                    )

            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=np.array(faces, dtype=np.int64),
                process=False,
            )
            mesh.merge_vertices(digits_vertex=6)
            mesh.remove_unreferenced_vertices()
            return mesh
        finally:
            try:
                if gmsh.isInitialized():
                    gmsh.clear()
            finally:
                if initialized_here and gmsh.isInitialized():
                    gmsh.finalize()


def _per_solid_gmsh_child(step_path_str: str,
                          size_min: float | None,
                          size_max: float | None,
                          size_mode: str,
                          result_path: str) -> None:
    """Subprocess entrypoint. Pickles the mesh (or an error payload) to disk."""
    import pickle
    from pathlib import Path as _Path
    try:
        mesh = load_step_as_gmsh_surface_mesh(
            _Path(step_path_str),
            size_min=size_min,
            size_max=size_max,
            size_mode=size_mode,
        )
        payload = {"status": "ok", "mesh": mesh}
    except BaseException as e:
        payload = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    with open(result_path, "wb") as f:
        pickle.dump(payload, f)


def _gmsh_mesh_occ_solid_sandboxed(
    solid,
    size_min: float | None,
    size_max: float | None,
    size_mode: str = "relative",
    *,
    timeout_s: int = DEFAULT_PER_SOLID_GMSH_TIMEOUT,
):
    """Mesh a single OCC solid via gmsh inside a subprocess with a wall-clock timeout.

    gmsh is initialized with ``interruptible=False`` and has no internal time
    limit, so a pathological solid can hang the parent indefinitely. We
    serialize the solid to a temporary STEP in the parent, then spawn a
    ``forkserver`` child that runs :func:`load_step_as_gmsh_surface_mesh`.
    On timeout the child is killed and this function returns ``None`` — the
    caller is expected to fall back to an OCP mesh rather than drop the solid.
    """
    import multiprocessing as _mp
    import os
    import pickle

    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    with tempfile.NamedTemporaryFile(suffix=".step", delete=True) as tmp:
        solid_step = tmp.name

    try:
        writer = STEPControl_Writer()
        writer.Transfer(solid, STEPControl_AsIs)
        if writer.Write(solid_step) != IFSelect_RetDone:
            logger.debug("per-solid gmsh: failed to write solid to temp STEP")
            return None

        fd, result_path = tempfile.mkstemp(suffix=".per_solid_gmsh.pkl")
        os.close(fd)

        ctx = _mp.get_context("forkserver")
        proc = ctx.Process(
            target=_per_solid_gmsh_child,
            args=(solid_step, size_min, size_max, size_mode, result_path),
        )
        proc.start()
        proc.join(timeout=timeout_s)

        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                logger.debug(f"per-solid gmsh subprocess exceeded {timeout_s}s")
                return None

            if proc.exitcode != 0:
                logger.debug(f"per-solid gmsh subprocess exited {proc.exitcode}")
                return None

            try:
                with open(result_path, "rb") as f:
                    payload = pickle.load(f)
            except Exception as e:
                logger.debug(f"per-solid gmsh subprocess produced no result: {e}")
                return None

            if payload.get("status") != "ok":
                logger.debug(f"per-solid gmsh subprocess error: {payload.get('error')}")
                return None
            return payload.get("mesh")
        finally:
            Path(result_path).unlink(missing_ok=True)
    finally:
        Path(solid_step).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Mesh statistics
# ---------------------------------------------------------------------------

def build_mesh_stats(
    mesh,
    *,
    source: Optional[str] = None,
    gmsh_size: Optional[float] = None,
    gmsh_size_requested: Optional[float] = None,
    gmsh_size_mode: Optional[str] = None,
    bbox_max_extent: Optional[float] = None,
) -> dict[str, Any]:
    """Build a consistent mesh statistics dict for exports and debug output."""
    stats = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "components": int(len(mesh.split(only_watertight=False))),
        "bounds": mesh.bounds.tolist(),
    }
    if source is not None:
        stats["source"] = source
    if gmsh_size is not None:
        stats["size"] = gmsh_size
    if gmsh_size_requested is not None:
        stats["size_requested"] = gmsh_size_requested
    if gmsh_size_mode is not None:
        stats["size_mode"] = gmsh_size_mode
    if bbox_max_extent is not None:
        stats["bbox_max_extent"] = bbox_max_extent
    return stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_step_as_trimesh_with_source(
    step_path: Path,
    *,
    gmsh_size: float = DEFAULT_STEP_GMSH_SIZE,
    gmsh_size_mode: str = "relative",
    linear_deflection: float = DEFAULT_STEP_LINEAR_DEFLECTION,
    angular_deflection: float = DEFAULT_STEP_ANGULAR_DEFLECTION,
    gmsh_verbose: bool = False,
) -> Tuple[Optional[Any], Optional[str]]:
    """Load a STEP file as trimesh, returning ``(mesh|None, source|None)``.

    ``source`` is one of ``per_solid``, ``free_shells_only``,
    ``per_solid+free_shells`` (or ``None``).

    Main pipeline: :func:`_load_step_via_ocp_aqd_per_solid` — per-solid OCP
    AQD with a sandboxed gmsh fallback for solids whose OCP mesh has open
    edges, and a final OCP open-edge fallback so individual solids are never
    silently dropped when gmsh fails.

    Free-shell augmentation (orthogonal): after the main pipeline produces
    ``main_mesh`` (which only sees ``TopAbs_SOLID`` units), every free shell /
    free face is also meshed and concatenated. STEPs without free shells/faces
    are unaffected (no-op).
    """
    trimesh = _trimesh()

    step_path = Path(step_path)
    if not step_path.exists():
        logger.error(f"File not found: {step_path}")
        return None, None

    # --- Adapt linear_deflection to model size (bbox-relative) ---
    # The default 0.01 is absolute, far too coarse for small models (e.g.
    # Text2CAD bbox ~0.75 -> 0.01 is 1.3% of bbox). Scale to 0.05% of bbox for
    # smooth tessellation of curved surfaces. Only ever refines.
    _ADAPTIVE_LD_FRACTION = 0.0005  # 0.05% of bbox max extent
    bbox_ext = _get_step_bbox_max_extent(step_path)
    if bbox_ext is not None and bbox_ext > 0:
        adaptive_ld = _ADAPTIVE_LD_FRACTION * bbox_ext
        if adaptive_ld < linear_deflection:
            logger.debug(
                f"Adapting linear_deflection: {linear_deflection} -> "
                f"{adaptive_ld:.6f} (bbox={bbox_ext:.4f})"
            )
            linear_deflection = adaptive_ld

    # Read the shape once and reuse it for both the main per-solid pipeline
    # and the free-shell augmentation below.
    try:
        occ_shape = read_step_shape(step_path)
    except Exception as e:
        logger.debug(f"STEP read failed for {step_path}: {e}")
        return None, None

    main_mesh = None
    main_source = None

    # --- Per-solid pipeline (OCP AQD + sandboxed gmsh fallback per solid) ---
    try:
        main_mesh = _load_step_via_ocp_aqd_per_solid(
            step_path,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
            gmsh_size=gmsh_size,
            gmsh_size_mode=gmsh_size_mode,
            occ_shape=occ_shape,
        )
        if main_mesh is not None:
            main_source = "per_solid"
    except Exception as e:
        logger.debug(f"Per-solid meshing failed for {step_path}: {e}")

    # --- Free-shell augmentation (independent of main pipeline) ---
    free_mesh = None
    try:
        free_mesh = mesh_free_shells_and_faces(
            occ_shape,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
        )
    except Exception as e:
        logger.debug(f"Free-shell scan failed for {step_path}: {e}")

    if free_mesh is not None:
        if main_mesh is None:
            return free_mesh, "free_shells_only"
        merged = trimesh.util.concatenate([main_mesh, free_mesh])
        return merged, f"{main_source}+free_shells"

    if main_mesh is not None:
        return main_mesh, main_source

    return None, None


def load_step_as_trimesh(
    step_path: Path,
    *,
    gmsh_size: float = DEFAULT_STEP_GMSH_SIZE,
    gmsh_size_mode: str = "relative",
    linear_deflection: float = DEFAULT_STEP_LINEAR_DEFLECTION,
    angular_deflection: float = DEFAULT_STEP_ANGULAR_DEFLECTION,
    gmsh_verbose: bool = False,
):
    """Load a STEP file as trimesh, reusing the shared pipeline."""
    mesh, _ = load_step_as_trimesh_with_source(
        step_path,
        gmsh_size=gmsh_size,
        gmsh_size_mode=gmsh_size_mode,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        gmsh_verbose=gmsh_verbose,
    )
    return mesh


def export_step_to_stl(
    step_path: Path,
    stl_path: Path,
    *,
    gmsh_size: float = DEFAULT_STEP_GMSH_SIZE,
    gmsh_size_mode: str = "relative",
    linear_deflection: float = DEFAULT_STEP_LINEAR_DEFLECTION,
    angular_deflection: float = DEFAULT_STEP_ANGULAR_DEFLECTION,
    gmsh_verbose: bool = False,
) -> Tuple[Optional[str], Optional[dict[str, Any]]]:
    """Export a STEP file to STL using the shared pipeline.

    Returns ``(error|None, stats|None)``. Writes the STL via ``mesh.export``.
    A non-watertight result is logged but does not fail.
    """
    mesh, source = load_step_as_trimesh_with_source(
        step_path,
        gmsh_size=gmsh_size,
        gmsh_size_mode=gmsh_size_mode,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        gmsh_verbose=gmsh_verbose,
    )
    if mesh is None:
        return f"STEP->mesh conversion failed for {step_path}", None

    stl_path = Path(stl_path)
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(stl_path)

    is_gmsh_source = bool(source and "gmsh" in source)
    bbox_max_extent = _get_step_bbox_max_extent(step_path) if is_gmsh_source else None
    resolved_gmsh_size = (
        _resolve_gmsh_size(gmsh_size, bbox_max_extent, gmsh_size_mode)
        if is_gmsh_source else None
    )
    stats = build_mesh_stats(
        mesh,
        source=source,
        gmsh_size=resolved_gmsh_size,
        gmsh_size_requested=gmsh_size if is_gmsh_source else None,
        gmsh_size_mode=gmsh_size_mode if is_gmsh_source else None,
        bbox_max_extent=bbox_max_extent,
    )
    if not mesh.is_watertight:
        logger.warning(f"STEP export produced non-watertight mesh for {step_path} via {source}")
    return None, stats


def load_mesh(path):
    """Load any supported geometry file as a single ``trimesh.Trimesh``.

    - ``.step`` / ``.stp`` -> :func:`load_step_as_trimesh` (OCP tessellation).
    - anything else (STL / OBJ / PLY / GLB / ...) -> ``trimesh.load(force='mesh')``,
      concatenating a ``Scene`` into one mesh.

    Returns ``None`` on failure or empty geometry. This is the shared loader
    imported by ``metrics/geometry.py`` and ``metrics/part.py`` for both GT and
    predicted meshes.
    """
    trimesh = _trimesh()

    path = Path(path)
    if not path.exists():
        logger.error(f"Mesh file not found: {path}")
        return None

    suffix = path.suffix.lower()
    if suffix in (".step", ".stp"):
        return load_step_as_trimesh(path)

    try:
        loaded = trimesh.load(str(path), force="mesh")
    except Exception as e:
        logger.debug(f"trimesh.load failed for {path}: {e}")
        return None

    if loaded is None:
        return None
    if isinstance(loaded, trimesh.Scene):
        geos = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geos:
            return None
        loaded = trimesh.util.concatenate(geos)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        return None
    return loaded
