import numpy as np

import config as cfg
import cppn
import track


def circle(r=200.0, n=360):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def figure_eight(r=100.0, n=360):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([r * np.sin(2.0 * t), r * np.sin(t)], axis=1)


def test_centerline_shape():
    rng = np.random.default_rng(0)
    g = cppn.random_genome(cfg.TRACK_CPPN_LAYERS, rng)
    assert track.centerline(g).shape == (cfg.TRACK_POINTS, 2)


def test_smoothing_keeps_circle_a_circle():
    c = circle()
    s = track.smooth_closed(c, 15)
    r = np.linalg.norm(s, axis=1)
    assert r.std() < 0.1


def test_smoothing_does_not_break_the_seam():
    c = circle()
    s = track.smooth_closed(c, 15)
    steps = np.linalg.norm(np.roll(s, -1, axis=0) - s, axis=1)
    assert steps.max() < steps.mean() * 1.05


def test_smoothing_window_one_is_identity():
    c = circle()
    assert np.allclose(track.smooth_closed(c, 1), c)


def test_tangents_are_unit_length():
    t = track.tangents(circle())
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0)


def test_normals_are_perpendicular_to_tangents():
    c = circle()
    dot = np.sum(track.tangents(c) * track.normals(c), axis=1)
    assert np.allclose(dot, 0.0, atol=1e-12)


def test_walls_are_offset_by_half_width():
    c = circle()
    left, right = track.offset_walls(c, 45.0)
    assert np.allclose(np.linalg.norm(left - c, axis=1), 45.0)
    assert np.allclose(np.linalg.norm(right - c, axis=1), 45.0)


def test_walls_are_on_opposite_sides():
    c = circle()
    left, right = track.offset_walls(c, 45.0)
    assert np.allclose(left + right, 2.0 * c)


def test_curvature_of_circle_is_quarter_of_true():
    for r in (100.0, 200.0, 400.0):
        k = track.curvature(circle(r))
        assert abs(k.mean() - 1.0 / (4.0 * r)) < 0.05 / (4.0 * r)


def test_curvature_of_circle_is_constant():
    k = track.curvature(circle(200.0))
    assert k.std() < k.mean() * 1e-6


def test_polyline_length_of_circle():
    assert abs(track.polyline_length(circle(100.0)) - 2.0 * np.pi * 100.0) < 6.3


def test_checkpoints_shape_and_ends_on_walls():
    c = circle()
    left, right = track.offset_walls(c, 45.0)
    cps = track.build_checkpoints(left, right, 8)
    assert cps.shape == (45, 2, 2)
    assert np.allclose(cps[:, 0, :], left[::8])
    assert np.allclose(cps[:, 1, :], right[::8])


def test_checkpoints_go_in_order_along_the_line():
    c = circle()
    left, right = track.offset_walls(c, 45.0)
    mids = track.build_checkpoints(left, right, 8).mean(axis=1)
    angles = np.unwrap(np.arctan2(mids[:, 1], mids[:, 0]))
    assert np.all(np.diff(angles) > 0.0)


def test_circle_has_no_self_intersection():
    assert not track.any_self_intersection(circle())


def test_figure_eight_has_self_intersection():
    assert track.any_self_intersection(figure_eight())


def test_folded_inner_wall_is_detected():
    t = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    r = 200.0 + 150.0 * np.sin(5.0 * t)
    c = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    left, _ = track.offset_walls(c, 45.0)
    assert track.any_self_intersection(left)
