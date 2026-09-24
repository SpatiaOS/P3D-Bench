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
