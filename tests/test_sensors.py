import numpy as np

import config as cfg
import field
import sensors
import track


HALF = 45.0
TOL = cfg.RAY_STEP + 1.0


def corridor_field(half=HALF, cell=1.0, nx=1400, ny=900):
    origin = np.array([-700.0, -450.0])
    gy = origin[1] + (np.arange(ny) + 0.5) * cell
    values = np.tile(half - np.abs(gy), (nx, 1))
    return field.Field(values, origin, cell)


def cast_one(fld, x=0.0, y=0.0, heading=0.0):
    return sensors.cast(np.array([[x, y]]), np.array([heading]), fld)[0]


def test_output_shape():
    fld = corridor_field()
    pos = np.zeros((5, 2))
    assert sensors.cast(pos, np.zeros(5), fld).shape == (5, cfg.N_RAYS)


def test_side_rays_see_the_corridor_walls():
    rays = cast_one(corridor_field())
    assert abs(rays[0] - HALF) < TOL
    assert abs(rays[6] - HALF) < TOL


def test_forward_ray_sees_nothing_in_a_straight_corridor():
    assert cast_one(corridor_field())[3] == cfg.RAY_MAX


def test_diagonal_rays_follow_the_sine_rule():
    rays = cast_one(corridor_field())
    for i, deg in ((1, 60.0), (2, 30.0), (4, 30.0), (5, 60.0)):
        assert abs(rays[i] - HALF / np.sin(np.radians(deg))) < TOL


def test_rays_are_symmetric_about_the_nose():
    rays = cast_one(corridor_field())
    assert np.allclose(rays[:3], rays[6:3:-1], atol=TOL)


def test_turning_the_car_rotates_the_readings():
    fld = corridor_field()
    straight = cast_one(fld)
    turned = cast_one(fld, heading=np.pi / 2.0)
    assert abs(turned[3] - straight[0]) < TOL
    assert abs(turned[0] - straight[3]) < TOL


def test_car_near_a_wall_sees_it_close():
    rays = cast_one(corridor_field(), y=30.0)
    assert abs(rays[6] - 15.0) < TOL
    assert abs(rays[0] - 75.0) < TOL


def test_car_outside_the_track_sees_walls_at_minimum_distance():
    rays = cast_one(corridor_field(), y=200.0)
    assert np.all(rays == cfg.RAY_STEP)


def test_readings_never_exceed_the_declared_range():
    fld = corridor_field()
    rng = np.random.default_rng(0)
    pos = rng.uniform(-200.0, 200.0, (40, 2))
    rays = sensors.cast(pos, rng.uniform(-np.pi, np.pi, 40), fld)
    assert np.all(rays > 0.0) and np.all(rays <= cfg.RAY_MAX)


def test_on_a_real_track_the_car_sees_both_walls():
    trk = track.evolve_track(np.random.default_rng(0))
    fld = field.build_for_track(trk)
    rays = sensors.cast(trk.center[:1], np.array([trk.start_angle]), fld)[0]
    assert rays[0] < trk.width
    assert rays[6] < trk.width
