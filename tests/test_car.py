import numpy as np

import car
import config as cfg
import cppn


def many_cars(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return [car.random_car(rng) for _ in range(n)]


def genome(seed=0):
    return cppn.random_genome(cfg.CAR_CPPN_LAYERS, np.random.default_rng(seed), cfg.CAR_INIT_SCALE)


def test_body_has_the_configured_number_of_points():
    assert car.random_car(np.random.default_rng(0)).shape.shape == (cfg.CAR_POINTS, 2)


def test_polygon_area_of_a_unit_square():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert abs(car.polygon_area(square) - 1.0) < 1e-12


def test_polygon_area_ignores_winding_direction():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert car.polygon_area(square) == car.polygon_area(square[::-1])


def test_body_outline_is_closed_and_centred_on_the_axis():
    body = car.body_outline(genome(1), 30.0, 14.0)
    assert abs(float(body[:, 1].mean())) < 1.0
    assert np.linalg.norm(body[0] - body[-1]) < np.ptp(body[:, 0])


def test_body_is_longer_than_it_is_wide():
    for c in many_cars(100):
        assert c.length > c.width


def test_body_follows_the_requested_proportions():
    body = car.body_outline(np.zeros(cppn.genome_size(cfg.CAR_CPPN_LAYERS)), 40.0, 16.0)
    assert abs(np.ptp(body[:, 1]) - 16.0) < 1.0
    assert 30.0 < np.ptp(body[:, 0]) < 40.0


def test_the_nose_is_narrower_than_the_tail():
    body = car.body_outline(np.zeros(cppn.genome_size(cfg.CAR_CPPN_LAYERS)), 40.0, 16.0)
    front = np.ptp(body[body[:, 0] > 12.0][:, 1])
    back = np.ptp(body[body[:, 0] < -12.0][:, 1])
    assert front < back


def test_there_are_four_wheels_at_the_four_corners():
    c = car.random_car(np.random.default_rng(0))
    assert len(c.wheels) == 4
    centres = np.array([w.mean(axis=0) for w in c.wheels])
    assert set(np.sign(centres[:, 0]).astype(int)) == {-1, 1}
    assert set(np.sign(centres[:, 1]).astype(int)) == {-1, 1}


def test_wheels_stick_out_past_the_body():
    for c in many_cars(50):
        widest = max(float(np.abs(w[:, 1]).max()) for w in c.wheels)
        assert widest > c.half_width


def test_cockpit_sits_inside_the_body():
    for c in many_cars(50):
        assert np.abs(c.cockpit[:, 0]).max() < c.length * 0.5
        assert np.abs(c.cockpit[:, 1]).max() < c.half_width


def test_all_parameters_stay_in_sane_ranges():
    for c in many_cars():
        assert car.MASS_MIN <= c.mass <= car.MASS_MAX
        assert car.SPEED_BASE <= c.max_speed <= car.SPEED_BASE + car.SPEED_GAIN
        assert car.STEER_BASE - car.STEER_DROP <= c.max_steer <= car.STEER_BASE
        assert c.area > 0.0 and c.length > 0.0 and c.width > 0.0


def test_car_fits_inside_the_track():
    for c in many_cars():
        assert c.half_width < cfg.TRACK_WIDTH * 0.5


def test_mass_does_not_saturate_at_the_limits():
    mass = np.array([c.mass for c in many_cars()])
    assert mass.std() > 0.1
    assert np.mean(mass >= car.MASS_MAX - 1e-6) < 0.2
    assert np.mean(mass <= car.MASS_MIN + 1e-6) < 0.2


def test_speed_and_steering_actually_vary():
    cars = many_cars()
    assert np.array([c.max_speed for c in cars]).std() > 10.0
    assert np.array([c.max_steer for c in cars]).std() > 0.2


def test_longer_cars_are_faster_and_turn_worse():
    cars = many_cars()
    aspect = np.array([c.length / c.width for c in cars])
    assert np.corrcoef(aspect, [c.max_speed for c in cars])[0, 1] > 0.8
    assert np.corrcoef(aspect, [c.max_steer for c in cars])[0, 1] < -0.8


def test_mass_and_aspect_are_mostly_independent():
    cars = many_cars()
    aspect = np.array([c.length / c.width for c in cars])
    assert abs(np.corrcoef(aspect, [c.mass for c in cars])[0, 1]) < 0.6


def test_stretch_stays_in_its_range():
    rng = np.random.default_rng(6)
    for _ in range(200):
        s = car.stretch_from(cppn.random_genome(cfg.CAR_CPPN_LAYERS, rng, cfg.CAR_INIT_SCALE))
        assert car.STRETCH_MID - car.STRETCH_SPAN < s < car.STRETCH_MID + car.STRETCH_SPAN


def test_half_size_stays_in_its_range():
    rng = np.random.default_rng(7)
    for _ in range(200):
        h = car.half_size_from(cppn.random_genome(cfg.CAR_CPPN_LAYERS, rng, cfg.CAR_INIT_SCALE))
        assert car.HALF_SIZE_MIN <= h <= car.HALF_SIZE_MAX


def test_power_equals_mass_times_acceleration():
    for c in many_cars(30):
        assert abs(c.power - c.mass * c.accel) < 1e-9


def test_same_genome_gives_the_same_car():
    g = genome(3)
    a, b = car.generate(g), car.generate(g)
    assert np.allclose(a.shape, b.shape)
    assert a.max_speed == b.max_speed and a.color == b.color


def test_different_genomes_give_different_cars():
    rng = np.random.default_rng(4)
    a, b = car.random_car(rng), car.random_car(rng)
    assert not np.allclose(a.shape, b.shape)


def test_colours_are_valid_and_varied():
    colours = {c.color for c in many_cars(60)}
    assert len(colours) > 30
    for r, g, b in colours:
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255
