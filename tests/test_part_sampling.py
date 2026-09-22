"""Measurement reproducibility and default Part protocol."""

import pytest

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")

from p3dbench.metrics import part


def test_part_sampling_matches_legacy_uniform_sequence():
    # On one unit right triangle, legacy surface samples are just reflected
    # uniform barycentric coordinates; the first draws select the sole face.
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                           faces=[[0, 1, 2]], process=False)
    legacy = np.random.RandomState(42)
    legacy.random_sample(2048)
    uv = legacy.random_sample((2048, 2))
    outside = uv.sum(axis=1) > 1.
    uv[outside] = 1. - uv[outside]
    expected = np.column_stack([uv, np.zeros(2048)])
    np.testing.assert_array_equal(part._sample_points(mesh, 2048), expected)


def test_part_sampling_is_repeatable_and_preserves_global_rng(tmp_path):
    path = tmp_path / "part.stl"
    trimesh.creation.box(extents=[1., 2., 3.]).export(path)
    transform = np.array([[-.2, 0, 0, .1], [0, .2, 0, -.1],
                          [0, 0, .2, .3], [0, 0, 0, 1]])
    saved = np.random.get_state()
    try:
        np.random.seed(8123)
        before = np.random.get_state()
        first = part._sample_part_with_transform(str(path), 2048, transform)
        after = np.random.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]
        np.random.seed(713)
        second = part._sample_part_with_transform(str(path), 2048, transform)
        np.testing.assert_array_equal(first, second)
        assert first.shape == (2048, 3)
    finally:
        np.random.set_state(saved)


def test_identical_part_is_matched_at_revised_defaults(tmp_path, monkeypatch):
    path = tmp_path / "box.stl"
    trimesh.creation.box(extents=[1., .6, .3]).export(path)
    pred_path = tmp_path / "pred.stl"
    pred_path.write_bytes(path.read_bytes())
    observed_sizes = []
    sampler = part._sample_points

    def record_samples(mesh, n_pts):
        observed_sizes.append(n_pts)
        return sampler(mesh, n_pts)

    monkeypatch.setattr(part, "_sample_points", record_samples)
    parts = [{"stl_path": str(path), "instance_count": 1}]
    result = part.evaluate_assembly_parts(
        parts, [{"stl": str(pred_path)}],
        gt_union_path=str(path), pred_union_path=str(pred_path),
        align_transform_4x4=np.eye(4),
    )
    assert observed_sizes and set(observed_sizes) == {2048}
    assert result["alignment"]["f_score_tau_frac"] == .03
    assert result["alignment"]["f_score_min"] == .7
    assert result["alignment"]["match_f1"] == 1.
    assert result["per_part_mean"]["f_score"] == pytest.approx(1.)
