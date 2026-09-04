import time

import numpy as np

import config as cfg
import field
import track


def circle(r=200.0, n=360):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def ring_field(r=200.0, half=45.0, cell=2.0):
    return field.build(circle(r), half, cell=cell)


def test_value_on_the_centerline_is_half_width():
    f = ring_field()
    on_line = f.sample(circle(200.0)[::7])
    assert np.all(np.abs(on_line - 45.0) < 3.0)


def test_value_at_the_wall_is_near_zero():
    f = ring_field()
    assert np.all(np.abs(f.sample(circle(245.0)[::7])) < 3.0)
    assert np.all(np.abs(f.sample(circle(155.0)[::7])) < 3.0)


def test_value_outside_the_track_is_negative():
    f = ring_field()
    assert np.all(f.sample(circle(300.0)[::7]) < 0.0)
    assert np.all(f.sample(circle(100.0)[::7]) < 0.0)


def test_value_drops_by_one_per_pixel_away_from_the_line():
    f = ring_field()
    near = f.sample(np.array([[210.0, 0.0]]))[0]
    far = f.sample(np.array([[230.0, 0.0]]))[0]
    assert abs((near - far) - 20.0) < 3.0


def test_out_of_bounds_reads_as_deep_outside():
    f = ring_field()
    assert f.sample(np.array([[1e6, 1e6]]))[0] == field.OUTSIDE
    assert f.sample(np.array([[-1e6, 0.0]]))[0] == field.OUTSIDE


def test_sample_keeps_the_input_shape():
    f = ring_field(cell=6.0)
    for shape in [(5, 2), (4, 7, 2), (3, 2, 6, 2)]:
        assert f.sample(np.zeros(shape)).shape == shape[:-1]


def test_padding_leaves_room_past_the_outer_wall():
    f = ring_field(cell=6.0)
    outer_wall = 245.0
    reach = outer_wall + cfg.RAY_STEP
    assert f.sample(np.array([[reach, 0.0], [-reach, 0.0]]))[0] != field.OUTSIDE
    assert f.origin[0] < -reach


def test_build_for_track_is_fast_enough():
    trk = track.evolve_track(np.random.default_rng(0))
    t0 = time.perf_counter()
    f = field.build_for_track(trk)
    assert time.perf_counter() - t0 < 2.0
    assert np.all(f.sample(trk.center[::9]) > 0.0)


def test_field_agrees_with_the_walls_of_a_real_track():
    trk = track.evolve_track(np.random.default_rng(1))
    f = field.build_for_track(trk)
    assert np.all(np.abs(f.sample(trk.left[::9])) < 8.0)
    assert np.all(np.abs(f.sample(trk.right[::9])) < 8.0)
