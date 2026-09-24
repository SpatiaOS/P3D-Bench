"""Validated profile reconstruction for checksum-registered CAD references."""
from __future__ import annotations

import copy
import json
from pathlib import Path


def _is_solid(shape):
    solids = shape.Solids()
    return bool(solids) and shape.isValid() and all(s.Volume() > 0 for s in solids)


def _heal(shape):
    """Repair BRep orientation/connectivity; reject empty or invalid results."""
    if _is_solid(shape):
        return shape
    fixed = shape.copy().fix()
    if not _is_solid(fixed):
        raise ValueError("Reference shape remains invalid or empty after healing")
    return fixed


def _boolean(left, right, operation):
    """Retry failed Booleans with bounded tolerance and volume checks."""
    last_error = None
    for tolerance in (None, 1e-8, 1e-7, 1e-6, 1e-5):
        try:
            options = {} if tolerance is None else {"tol": tolerance}
            result = getattr(left.copy(), operation)(right.copy(), **options).clean()
            result = _heal(result)
            volume, va, vb = result.Volume(), left.Volume(), right.Volume()
            epsilon = max(1e-10, (va + vb) * 1e-6)
            if operation == "fuse" and not max(va, vb) - epsilon <= volume <= va + vb + epsilon:
                raise ValueError("Union volume violates operand bounds")
            if operation == "cut" and not 0 < volume <= va + epsilon:
                raise ValueError("Cut volume violates operand bounds")
            if operation == "intersect" and not 0 < volume <= min(va, vb) + epsilon:
                raise ValueError("Intersection volume violates operand bounds")
            return result
        except Exception as error:
            last_error = error
    raise ValueError(f"Reference {operation} failed after bounded retries") from last_error


def _restore_precision(data, digest):
    """Apply source-verified sketch precision; every value must round back."""
    path = Path(__file__).with_name("reference_profile_precision.json")
    record = json.loads(path.read_text(encoding="utf-8"))["programs"].get(digest)
    if record is None:
        return data
    restored = copy.deepcopy(data)
    for part_name, sketch in record["sketches"].items():
        old_sketch = data["parts"][part_name]["sketch"]
        if old_sketch.keys() != sketch.keys():
            raise ValueError("Precision record face mismatch")
        for face_name, face in sketch.items():
            if old_sketch[face_name].keys() != face.keys():
                raise ValueError("Precision record loop mismatch")
            for loop_name, loop in face.items():
                old_loop = old_sketch[face_name][loop_name]
                if old_loop.keys() != loop.keys():
                    raise ValueError("Precision record curve mismatch")
                for curve_name, curve in loop.items():
                    original = old_loop[curve_name]
                    if original.keys() != curve.keys():
                        raise ValueError("Precision record field mismatch")
                    for key, value in curve.items():
                        before = original[key]
                        pairs = zip(before, value) if isinstance(value, list) else [(before, value)]
                        if isinstance(value, list) and len(before) != len(value):
                            raise ValueError("Precision record coordinate mismatch")
                        if any(round(new, 4) != old for old, new in pairs):
                            raise ValueError("Precision record exceeds published rounding")
        restored["parts"][part_name]["sketch"] = sketch
    return restored


def build_reference_geometry(data, digest, build_workplane, build_wire, extrude_interval):
    """Reconstruct registered references without discarding failed features."""
    import cadquery as cq

    data = _restore_precision(data, digest)
    completed, current = [], None
    operations = {"JoinFeatureOperation": "fuse", "CutFeatureOperation": "cut",
                  "IntersectFeatureOperation": "intersect"}
    for part_name, part in data["parts"].items():
        extrusion = part["extrusion"]
        feature = None
        for face in part["sketch"].values():
            frame = part["coordinate_system"]
            sketch = build_workplane(frame["Euler Angles"], frame["Translation Vector"])
            for loop in face.values():
                sketch = build_wire(sketch, loop, 1.0)
            solid = _heal(extrude_interval(
                sketch, extrusion["extrude_depth_towards_normal"],
                extrusion["extrude_depth_opposite_normal"],
            ).val())
            feature = solid if feature is None else _boolean(feature, solid, "fuse")
        if feature is None:
            raise ValueError(f"Empty reference feature: {part_name}")
        operation = extrusion["operation"]
        if current is None or operation == "NewBodyFeatureOperation":
            if current is not None:
                completed.append(current)
            current = feature
        elif operation in operations:
            current = _boolean(current, feature, operations[operation])
        else:
            raise ValueError(f"Unknown reference operation: {operation}")
    if current is not None:
        completed.append(current)
    if not completed or not all(_is_solid(body) for body in completed):
        raise ValueError("Reference reconstruction produced no valid solid")
    return [cq.Workplane("XY").newObject([body]) for body in completed]
