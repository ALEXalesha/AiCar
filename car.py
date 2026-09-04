import colorsys

import numpy as np

import config as cfg
import cppn

SPEED_BASE, SPEED_GAIN = 170.0, 90.0
STEER_BASE, STEER_DROP = 3.4, 1.5
MASS_MIN, MASS_MAX = 0.8, 2.5
ASPECT_MIN, ASPECT_MAX = 1.4, 3.0
STRETCH_MID, STRETCH_SPAN = 2.1, 0.7

BODY_POWER = 6.0
BODY_BUMP = 0.09
NOSE_TAPER = 0.74
HALF_SIZE_MIN, HALF_SIZE_MAX = 6.0, 10.0

WHEEL_AT_X, WHEEL_AT_Y = 0.32, 0.54
WHEEL_LEN, WHEEL_WIDTH = 0.30, 0.20
COCKPIT_BACK, COCKPIT_LEN, COCKPIT_WIDTH = 0.08, 0.20, 0.24
COCKPIT_POINTS = 14


def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def stretch_from(genome):
    return STRETCH_MID + STRETCH_SPAN * np.tanh(0.2 * float(np.sum(genome[1::3])))


def half_size_from(genome):
    t = 0.5 + 0.5 * np.tanh(0.2 * float(np.sum(genome[2::3])))
    return HALF_SIZE_MIN + (HALF_SIZE_MAX - HALF_SIZE_MIN) * t


def colour_from(genome):
    hue = (float(np.sum(genome[0::3])) * 0.137) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def body_outline(genome, length, width, n_points=None):
    n_points = cfg.CAR_POINTS if n_points is None else n_points
    phi = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    a, b = length * 0.5, width * 0.5

    corner = (np.abs(np.cos(phi)) / a) ** BODY_POWER + (np.abs(np.sin(phi)) / b) ** BODY_POWER
    base = corner ** (-1.0 / BODY_POWER)

    features = cppn.angle_features(phi, cfg.CAR_CPPN_LAYERS[0])
    bump = cppn.forward(genome, cfg.CAR_CPPN_LAYERS, features)[:, 0]
    taper = 1.0 - (1.0 - NOSE_TAPER) * np.clip(np.cos(phi), 0.0, 1.0)

    r = base * (1.0 + BODY_BUMP * bump) * taper
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


def wheel_polygons(length, width):
    half_l, half_w = length * WHEEL_LEN * 0.5, width * WHEEL_WIDTH * 0.5
    box = np.array([[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]])
    return [box + np.array([sx * length * WHEEL_AT_X, sy * width * WHEEL_AT_Y])
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]


def cockpit_polygon(length, width):
    t = np.linspace(0.0, 2.0 * np.pi, COCKPIT_POINTS, endpoint=False)
    return np.stack([-COCKPIT_BACK * length + length * COCKPIT_LEN * np.cos(t),
                     width * COCKPIT_WIDTH * np.sin(t)], axis=1)


class Car:
    def __init__(self, shape, wheels, cockpit, mass, accel, max_speed, max_steer, color):
        self.shape = shape
        self.wheels = wheels
        self.cockpit = cockpit
        self.parts = list(wheels) + [shape, cockpit]
        self.stacked = np.vstack(self.parts)
        edges = np.cumsum([0] + [len(part) for part in self.parts])
        self.slices = list(zip(edges[:-1], edges[1:]))
        self.mass = mass
        self.accel = accel
        self.max_speed = max_speed
        self.max_steer = max_steer
        self.color = color
        self.length = float(np.ptp(shape[:, 0]))
        self.width = float(np.ptp(shape[:, 1]))
        self.half_width = self.width * 0.5
        self.area = polygon_area(shape)
        self.power = mass * accel


def generate(genome):
    aspect = float(np.clip(stretch_from(genome), ASPECT_MIN, ASPECT_MAX))
    half = half_size_from(genome)
    stretch = np.sqrt(aspect)
    length, width = 2.0 * half * stretch, 2.0 * half / stretch

    shape = body_outline(genome, length, width)
    max_area = np.pi * (HALF_SIZE_MAX * 1.2) ** 2
    filled = polygon_area(shape) / max_area
    mass = MASS_MIN + (MASS_MAX - MASS_MIN) * float(np.clip(filled, 0.0, 1.0))

    reach = (aspect - ASPECT_MIN) / (ASPECT_MAX - ASPECT_MIN)
    max_speed = SPEED_BASE + SPEED_GAIN * reach
    max_steer = STEER_BASE - STEER_DROP * reach
    accel = max_speed * cfg.DRAG * cfg.ACCEL_FACTOR

    return Car(shape, wheel_polygons(length, width), cockpit_polygon(length, width),
               mass, accel, max_speed, max_steer, colour_from(genome))


def random_car(rng):
    return generate(cppn.random_genome(cfg.CAR_CPPN_LAYERS, rng, cfg.CAR_INIT_SCALE))
