"""Text2CAD minimal-JSON -> CadQuery solids -> STEP/STL interpreter.

Converts the Text2CAD ``minimal_json`` format (top-level ``parts`` dict of
features, each with ``coordinate_system`` / ``sketch`` / ``extrusion``) into
CadQuery solids, then writes a STEP compound (assembly mode) and tessellates it
to STL via :mod:`p3dbench.compile.step_mesh`.

Coordinate system: the ``Euler Angles`` are ZYX intrinsic in degrees, computed
by the dataset as ``R.from_matrix(np.vstack((x_axis, y_axis, z_axis))).as_euler("zyx")``.
We rebuild via ``R.from_euler("zyx", angles, degrees=True)`` and read the matrix
*rows* as ``x_axis``, ``y_axis``, ``normal``.

Faithfulness flags (do NOT silently "fix"):
  * Two-sided extrusion (both depths > 0) uses CadQuery ``extrude(both=True)``,
    which is *symmetric* — asymmetric fwd/rev depths are approximated.
  * Hole loops are extruded single-direction only
    (``depth_fwd if depth_fwd > 0 else depth_rev``).
  * Parts are iterated in **insertion order** (not lexicographic ``sorted`` as in
    the research code), which matches the ``parts_meta`` insertion-order contract
    and is correct past ``part_9`` / ``part_10``.
  * ``scale = 1.0`` is hard-coded; Text2CAD coordinates are already in final
    physical units, nothing rescales.

cadquery / OCP are imported lazily so importing this module does not require the
``geometry`` extra to be installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from ..utils import require
from .step_mesh import export_step_to_stl

logger = logging.getLogger(__name__)

_GEOMETRY_EXTRA = "geometry"


def _cq():
    """Lazy cadquery handle."""
    return require("cadquery", _GEOMETRY_EXTRA, "Text2CAD minimal-JSON compile")


def _build_workplane(euler_angles_deg, translation):
    """Build a CadQuery workplane from Text2CAD's coordinate system.

    Args:
        euler_angles_deg: ``[angle_z, angle_y, angle_x]`` in degrees (ZYX).
        translation: ``[tx, ty, tz]`` origin offset.
    """
    from scipy.spatial.transform import Rotation as R

    cq = _cq()
    np = require("numpy", _GEOMETRY_EXTRA, "Text2CAD minimal-JSON compile")

    # Matrix rows are the x-axis, y-axis, normal of the sketch plane (the
    # dataset built the Euler angles as R.from_matrix(vstack(x, y, z)).as_euler).
    rot_matrix = R.from_euler("zyx", euler_angles_deg, degrees=True).as_matrix()
    x_axis = rot_matrix[0]
    normal = rot_matrix[2]
    origin = np.array(translation, dtype=float)

    plane = cq.Plane(
        origin=cq.Vector(*origin),
        xDir=cq.Vector(*x_axis),
        normal=cq.Vector(*normal),
    )
    return cq.Workplane(plane)


def _build_wire(wp, loop, scale):
    """Draw a closed wire from a loop of lines, arcs, and circles."""
    curves = list(loop.items())
    if not curves:
        return wp

    # Single full circle (Center + Radius) — already closed.
    if len(curves) == 1:
        curve_name, curve = curves[0]
        if curve_name.startswith('circle') and 'Center' in curve and 'Radius' in curve:
            cx, cy = curve['Center']
            r = curve['Radius']
            wp = wp.moveTo(cx * scale, cy * scale).circle(r * scale)
            return wp

    # Move to start point of first curve.
    first_curve = curves[0][1]
    sp = first_curve['Start Point']
    wp = wp.moveTo(sp[0] * scale, sp[1] * scale)

    for curve_name, curve in curves:
        if curve_name.startswith('line'):
            ep = curve['End Point']
            wp = wp.lineTo(ep[0] * scale, ep[1] * scale)
        elif curve_name.startswith('arc'):
            mp = curve['Mid Point']
            ep = curve['End Point']
            wp = wp.threePointArc(
                (mp[0] * scale, mp[1] * scale),
                (ep[0] * scale, ep[1] * scale)
            )
        elif curve_name.startswith('circle'):
            if 'Center' in curve and 'Radius' in curve:
                cx, cy = curve['Center']
                r = curve['Radius']
                wp = wp.moveTo(cx * scale, cy * scale).circle(r * scale)
            elif 'Mid Point' in curve:
                mp = curve['Mid Point']
                ep = curve['End Point']
                wp = wp.threePointArc(
                    (mp[0] * scale, mp[1] * scale),
                    (ep[0] * scale, ep[1] * scale)
                )
            else:
                ep = curve['End Point']
                wp = wp.lineTo(ep[0] * scale, ep[1] * scale)

    # Only close if not a standalone circle (circle() already closes).
    if not (len(curves) == 1 and curves[0][0].startswith('circle') and 'Center' in curves[0][1]):
        wp = wp.close()
    return wp


def _build_part_solid(part_name: str, part: dict, json_path: str):
    """Build the solid contributed by a single minimal_json feature."""
    ext = part.get('extrusion', {})
    if not ext:
        logger.warning(f"No extrusion info for {part_name} in {json_path}")
        return None

    scale = 1.0
    depth_fwd = ext.get('extrude_depth_towards_normal', 0)
    depth_rev = ext.get('extrude_depth_opposite_normal', 0)
    trans = part.get('coordinate_system', {}).get('Translation Vector', [0, 0, 0])
    euler = part.get('coordinate_system', {}).get('Euler Angles', [0, 0, 0])

    sketch = part.get('sketch', {})
    if not sketch:
        return None

    try:
        wp = _build_workplane(euler, trans)

        # Union all extruded faces that belong to the same feature.
        part_solid = None
        for face_name, face_data in sketch.items():
            loops = list(face_data.items())
            if not loops:
                continue

            outer_loop_name, outer_loop = loops[0]
            wp_sketch = _build_wire(wp, outer_loop, scale)

            if depth_fwd > 0 and depth_rev > 0:
                # NOTE (faithfulness): both=True extrudes symmetrically, so an
                # asymmetric fwd/rev split is approximated. Do not "fix".
                solid = wp_sketch.extrude(depth_fwd + depth_rev, both=True)
            elif depth_fwd > 0:
                solid = wp_sketch.extrude(depth_fwd)
            elif depth_rev > 0:
                solid = wp_sketch.extrude(-depth_rev)
            else:
                continue

            for inner_loop_name, inner_loop in loops[1:]:
                try:
                    hole_wp = wp
                    hole_wp = _build_wire(hole_wp, inner_loop, scale)
                    # NOTE (faithfulness): holes extrude single-direction only.
                    hole_solid = hole_wp.extrude(depth_fwd if depth_fwd > 0 else depth_rev)
                    solid = solid.cut(hole_solid)
                except Exception as e:
                    logger.debug(f"Inner loop {inner_loop_name} failed: {e}")

            if part_solid is None:
                part_solid = solid
            else:
                part_solid = part_solid.union(solid)

        return part_solid
    except Exception as e:
        logger.warning(f"Failed to build {part_name}: {e}")
        return None


def minimal_json_to_solids_assembly(json_path: str) -> List:
    """Convert a Text2CAD minimal JSON file to a list of CadQuery solids.

    Preserves assembly structure at the body level:
      - ``NewBodyFeatureOperation`` starts a new exported body.
      - ``Join`` / ``Cut`` / ``Intersect`` modify the current body in place.

    Parts are iterated in **insertion order** (matches the ``parts_meta``
    contract and is correct past ``part_9``/``part_10``). Returns one Workplane
    per independent body.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    parts = data.get('parts', {})
    if not parts:
        logger.warning(f"No parts found in {json_path}")
        return []

    completed_bodies = []
    current_body = None

    for part_name in parts.keys():
        part = parts[part_name]
        operation = part.get('extrusion', {}).get('operation', 'NewBodyFeatureOperation')
        part_solid = _build_part_solid(part_name, part, json_path)
        if part_solid is None:
            continue

        if operation == 'NewBodyFeatureOperation':
            if current_body is not None:
                completed_bodies.append(current_body)
            current_body = part_solid
            continue

        if current_body is None:
            logger.warning(
                f"Encountered {operation} before any NewBodyFeatureOperation in {json_path}; "
                "treating it as the start of a new body."
            )
            current_body = part_solid
            continue

        if operation == 'JoinFeatureOperation':
            current_body = current_body.union(part_solid)
        elif operation == 'CutFeatureOperation':
            current_body = current_body.cut(part_solid)
        elif operation == 'IntersectFeatureOperation':
            current_body = current_body.intersect(part_solid)
        else:
            logger.warning(
                f"Unknown operation {operation} for {part_name} in {json_path}; "
                "starting a new body."
            )
            completed_bodies.append(current_body)
            current_body = part_solid

    if current_body is not None:
        completed_bodies.append(current_body)

    return completed_bodies


def minimal_json_to_solid(json_path: str):
    """Convert a Text2CAD minimal JSON file to one fused CadQuery solid (legacy).

    Fuses all bodies via union. For assembly-aware export that preserves
    internal faces, use :func:`minimal_json_to_solids_assembly`.
    """
    bodies = minimal_json_to_solids_assembly(json_path)
    if not bodies:
        return None

    fused = bodies[0]
    for body in bodies[1:]:
        fused = fused.union(body)
    return fused


def export_minimal_json(json_path: str, output_dir: str, assembly_mode: bool = True,
                        output_prefix: str = "gt_model") -> dict:
    """Export a Text2CAD minimal JSON to STL + STEP.

    Args:
        json_path: Path to the minimal JSON file.
        output_dir: Directory for ``{output_prefix}.stl`` / ``{output_prefix}.step``.
        assembly_mode: True keeps internal faces via a STEP compound; False fuses
            all bodies into one solid (legacy).
        output_prefix: Output filename stem.

    Returns:
        ``{"stl": ..., "step": ...}`` on success, or ``{"error": ...}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stl_path = str(output_dir / f"{output_prefix}.stl")
    step_path = str(output_dir / f"{output_prefix}.step")

    try:
        if assembly_mode:
            solids = minimal_json_to_solids_assembly(json_path)
            if not solids:
                return {"error": "Failed to build solids from minimal JSON"}

            from OCP.IFSelect import IFSelect_RetDone
            from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
            from OCP.TopoDS import TopoDS_Builder, TopoDS_Compound

            compound = TopoDS_Compound()
            builder = TopoDS_Builder()
            builder.MakeCompound(compound)
            for solid in solids:
                builder.Add(compound, solid.val().wrapped)

            writer = STEPControl_Writer()
            writer.Transfer(compound, STEPControl_AsIs)
            status = writer.Write(step_path)
            if status != IFSelect_RetDone:
                return {"error": "STEP export failed"}

            error, _ = export_step_to_stl(Path(step_path), Path(stl_path))
            if error:
                raise RuntimeError(error)

        else:
            solid = minimal_json_to_solid(json_path)
            if solid is None:
                return {"error": "Failed to build solid from minimal JSON"}

            solid.val().exportStep(step_path)
            error, _ = export_step_to_stl(Path(step_path), Path(stl_path))
            if error:
                raise RuntimeError(error)

        return {"stl": stl_path, "step": step_path}
    except Exception as e:
        return {"error": f"Export failed: {e}"}


def export_minimal_json_to_step(json_path: str, step_path: str, assembly_mode: bool = True) -> str:
    """Export a Text2CAD minimal JSON directly to a STEP file.

    Used by GT preprocessing. Returns ``step_path`` on success; raises
    ``RuntimeError`` on failure.
    """
    Path(step_path).parent.mkdir(parents=True, exist_ok=True)

    if assembly_mode:
        solids = minimal_json_to_solids_assembly(json_path)
        if not solids:
            raise RuntimeError(f"Failed to build solids from {json_path}")

        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
        from OCP.TopoDS import TopoDS_Builder, TopoDS_Compound

        compound = TopoDS_Compound()
        builder = TopoDS_Builder()
        builder.MakeCompound(compound)
        for solid in solids:
            builder.Add(compound, solid.val().wrapped)

        writer = STEPControl_Writer()
        writer.Transfer(compound, STEPControl_AsIs)
        status = writer.Write(step_path)
        if status != IFSelect_RetDone:
            raise RuntimeError(f"STEP export failed for {json_path}")
    else:
        solid = minimal_json_to_solid(json_path)
        if solid is None:
            raise RuntimeError(f"Failed to build solid from {json_path}")
        solid.val().exportStep(step_path)

    return step_path
