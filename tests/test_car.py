import numpy as np

import car
import config as cfg
import cppn


def many_cars(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return [car.random_car(rng) for _ in range(n)]


def test_shape_has_the_configured_number_of_points():
    assert car.random_car(np.random.default_rng(0)).shape.shape == (cfg.CAR_POINTS, 2)


def test_polygon_area_of_a_unit_square():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert abs(car.polygon_area(square) - 1.0) < 1e-12


def test_polygon_area_ignores_winding_direction():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert car.polygon_area(square) == car.polygon_area(square[::-1])


def test_orient_puts_the_long_axis_along_x():
    rng = np.random.default_rng(0)
    for _ in range(50):
        pts = cppn.ring_shape(cppn.random_genome(cfg.CAR_CPPN_LAYERS, rng),
                              cfg.CAR_CPPN_LAYERS, cfg.CAR_POINTS, 5.0, 14.0)
        p = car.orient(pts)
        assert np.ptp(p[:, 0]) >= np.ptp(p[:, 1]) - 1e-9


def test_orient_keeps_the_area():
    rng = np.random.default_rng(1)
    pts = cppn.ring_shape(cppn.random_genome(cfg.CAR_CPPN_LAYERS, rng),
                          cfg.CAR_CPPN_LAYERS, cfg.CAR_POINTS, 5.0, 14.0)
    assert abs(car.polygon_area(car.orient(pts)) - car.polygon_area(pts)) < 1e-9


def test_orient_centres_the_shape():
    rng = np.random.default_rng(2)
    pts = cppn.ring_shape(cppn.random_genome(cfg.CAR_CPPN_LAYERS, rng),
                          cfg.CAR_CPPN_LAYERS, cfg.CAR_POINTS, 5.0, 14.0)
    assert np.allclose(car.orient(pts).mean(axis=0), 0.0, atol=1e-9)


def test_all_parameters_stay_in_sane_ranges():
    for c in many_cars():
        assert car.MASS_MIN <= c.mass <= car.MASS_MAX
        assert car.SPEED_BASE <= c.max_speed <= car.SPEED_BASE + car.SPEED_GAIN
        assert car.STEER_BASE - car.STEER_DROP <= c.max_steer <= car.STEER_BASE
        assert c.area > 0.0
        assert c.length > 0.0 and c.width > 0.0


def test_mass_does_not_saturate_at_the_limits():
    mass = np.array([c.mass for c in many_cars()])
    assert mass.std() > 0.2
    assert np.mean(mass >= car.MASS_MAX - 1e-6) < 0.2
    assert np.mean(mass <= car.MASS_MIN + 1e-6) < 0.2


def test_speed_and_steering_actually_vary():
    cars = many_cars()
    assert np.array([c.max_speed for c in cars]).std() > 10.0
    assert np.array([c.max_steer for c in cars]).std() > 0.2


def test_mass_and_aspect_are_mostly_independent():
    cars = many_cars()
    aspect = np.array([c.length / c.width for c in cars])
    mass = np.array([c.mass for c in cars])
    assert abs(np.corrcoef(aspect, mass)[0, 1]) < 0.5


def test_stretching_keeps_the_area():
    rng = np.random.default_rng(5)
    for _ in range(30):
        g = cppn.random_genome(cfg.CAR_CPPN_LAYERS, rng)
        blob = car.orient(cppn.ring_shape(g, cfg.CAR_CPPN_LAYERS, cfg.CAR_POINTS,
                                          cfg.CAR_R_MIN, cfg.CAR_R_MAX))
        assert abs(car.generate(g).area - car.polygon_area(blob)) < 1e-9


def test_stretch_stays_in_its_range():
    rng = np.random.default_rng(6)
    for _ in range(200):
        s = car.stretch_from(cppn.random_genome(cfg.CAR_CPPN_LAYERS, rng))
        assert car.STRETCH_MID - car.STRETCH_SPAN < s < car.STRETCH_MID + car.STRETCH_SPAN


def test_car_fits_inside_the_track():
    for c in many_cars():
        assert c.half_width < cfg.TRACK_WIDTH * 0.5


def test_longer_cars_are_faster_and_turn_worse():
    cars = many_cars()
    aspect = np.array([c.length / c.width for c in cars])
    speed = np.array([c.max_speed for c in cars])
    steer = np.array([c.max_steer for c in cars])
    assert np.corrcoef(aspect, speed)[0, 1] > 0.9
    assert np.corrcoef(aspect, steer)[0, 1] < -0.9


def test_power_equals_mass_times_acceleration():
    for c in many_cars(30):
        assert abs(c.power - c.mass * c.accel) < 1e-9


def test_same_genome_gives_the_same_car():
    g = cppn.random_genome(cfg.CAR_CPPN_LAYERS, np.random.default_rng(3))
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
