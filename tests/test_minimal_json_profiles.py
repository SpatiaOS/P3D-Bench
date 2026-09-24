"""Geometry-only regression tests; no datasets, models or renders required."""
import math
import json
from pathlib import Path

import pytest

pytest.importorskip("cadquery")

from p3dbench.compile.text2cad_interpreter import _build_part_solid_repaired as _build_part_solid


def circle(radius):
    return {"circle_1": {"Center": [0, 0], "Radius": radius}}


def feature(loops, forward, reverse):
    return {
        "coordinate_system": {"Euler Angles": [0, 0, 0], "Translation Vector": [0, 0, 0]},
        "sketch": {"face_1": loops},
        "extrusion": {"extrude_depth_towards_normal": forward,
                      "extrude_depth_opposite_normal": reverse,
                      "operation": "NewBodyFeatureOperation"},
    }


@pytest.mark.parametrize("inner_first", [False, True])
@pytest.mark.parametrize("depths", [(2, 0), (0, 2), (2, 1)])
def test_hole_order_and_depth_interval(inner_first, depths):
    radii = [1, 2] if inner_first else [2, 1]
    loops = {f"loop_{i}": circle(r) for i, r in enumerate(radii, 1)}
    shape = _build_part_solid("part_1", feature(loops, *depths), "test").val()
    assert shape.isValid() and len(shape.Solids()) == 1
    assert shape.Volume() == pytest.approx(3 * math.pi * sum(depths))
    bbox = shape.BoundingBox()
    assert bbox.zmin == pytest.approx(-depths[1], abs=1e-8)
    assert bbox.zmax == pytest.approx(depths[0], abs=1e-8)


def test_invalid_loop_is_not_silently_dropped():
    loops = {"outer": circle(2), "broken": {"line_1": {"Start Point": [0, 0]}}}
    with pytest.raises(ValueError, match="Failed to build"):
        _build_part_solid("part_1", feature(loops, 1, 0), "test")


@pytest.mark.parametrize("path", sorted((Path(__file__).parent / "fixtures/reference_profiles").glob("*.json")))
def test_reviewed_reference_has_valid_nonempty_bodies(path):
    from p3dbench.compile.text2cad_interpreter import minimal_json_to_solids_assembly
    from p3dbench.compile.reference_profile_repairs import needs_profile_repair
    data = json.loads(path.read_text())
    assert needs_profile_repair(data)
    bodies = minimal_json_to_solids_assembly(str(path))
    assert bodies
    assert all(b.val().Solids() and b.val().isValid() and b.val().Volume() > 0 for b in bodies)
    # An altered program must not inherit a reference-only repair.
    data["final_name"] = "modified program"
    assert not needs_profile_repair(data)


def test_unregistered_program_uses_original_builder(tmp_path, monkeypatch):
    import p3dbench.compile.text2cad_interpreter as interpreter
    calls = []
    sentinel = object()
    monkeypatch.setattr(interpreter, "_build_part_solid", lambda *args: calls.append("legacy") or sentinel)
    monkeypatch.setattr(interpreter, "_build_part_solid_repaired", lambda *args: pytest.fail("unexpected repair"))
    p = tmp_path / "program.json"
    p.write_text(json.dumps({"parts": {"part_1": feature({"loop_1": circle(2)}, 1, 0)}}))
    assert interpreter.minimal_json_to_solids_assembly(str(p)) == [sentinel]
    assert calls == ["legacy"]


def test_precision_records_round_back_and_preserve_features():
    from p3dbench.compile.reference_geometry import _restore_precision
    from p3dbench.compile.reference_profile_repairs import program_digest
    for uid in ["0043_00437520", "0013_00135195", "0009_00098309", "0072_00723126"]:
        path = Path(__file__).parent / "fixtures/reference_profiles" / (uid + ".json")
        data = json.loads(path.read_text())
        before = json.dumps(data)
        restored = _restore_precision(data, program_digest(data))
        assert json.dumps(data) == before
        assert list(restored["parts"]) == list(data["parts"])
        for key in data["parts"]:
            for field in ["coordinate_system", "extrusion", "description"]:
                assert restored["parts"][key][field] == data["parts"][key][field]
        bad = json.loads(before)
        first_part = next(iter(bad["parts"].values()))
        first_curve = next(iter(next(iter(next(iter(first_part["sketch"].values())).values())).values()))
        first_field = next(iter(first_curve))
        if isinstance(first_curve[first_field], list):
            first_curve[first_field][0] += 0.01
        else:
            first_curve[first_field] += 0.01
        with pytest.raises(ValueError, match="published rounding"):
            _restore_precision(bad, program_digest(data))


def test_healing_rejects_empty_shape():
    import cadquery as cq
    from p3dbench.compile.reference_geometry import _heal
    with pytest.raises(ValueError, match="invalid or empty"):
        _heal(cq.Compound.makeCompound([]))


def test_boolean_preserves_analytic_volume():
    import cadquery as cq
    from p3dbench.compile.reference_geometry import _boolean
    a = cq.Workplane("XY").box(2, 2, 2).val()
    b = cq.Workplane("XY").box(2, 2, 2).translate((1, 0, 0)).val()
    before = (a.Volume(), b.Volume())
    assert _boolean(a, b, "fuse").Volume() == pytest.approx(12)
    assert _boolean(a, b, "cut").Volume() == pytest.approx(4)
    assert _boolean(a, b, "intersect").Volume() == pytest.approx(4)
    assert (a.Volume(), b.Volume()) == before


@pytest.mark.parametrize("uid", ["0009_00098309", "0072_00723126"])
def test_precision_repair_survives_step_and_stl_roundtrip(uid, tmp_path):
    import cadquery as cq
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("gmsh")
    from p3dbench.compile.text2cad_interpreter import export_minimal_json
    from p3dbench.compile.step_mesh import read_step_shape
    path = Path(__file__).parent / "fixtures/reference_profiles" / (uid + ".json")
    result = export_minimal_json(str(path), str(tmp_path))
    assert "error" not in result, result
    shape = cq.Shape.cast(read_step_shape(Path(result["step"])))
    assert shape.isValid() and len(shape.Solids()) == 1 and shape.Volume() > 0
    mesh = trimesh.load(result["stl"], force="mesh")
    assert len(mesh.faces) > 0 and mesh.is_watertight
