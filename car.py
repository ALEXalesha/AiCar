import colorsys

import numpy as np

import config as cfg
import cppn

SPEED_BASE, SPEED_GAIN = 170.0, 90.0
STEER_BASE, STEER_DROP = 3.4, 1.5
MASS_MIN, MASS_MAX = 0.8, 2.5
ASPECT_MIN, ASPECT_MAX = 1.0, 3.0
STRETCH_MID, STRETCH_SPAN = 1.8, 0.8
MAX_AREA = np.pi * cfg.CAR_R_MAX ** 2


def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def orient(pts):
    p = pts - pts.mean(axis=0)
    _, vecs = np.linalg.eigh(p.T @ p)
    ang = np.arctan2(vecs[1, -1], vecs[0, -1])
    c, s = np.cos(ang), np.sin(ang)
    p = p @ np.array([[c, -s], [s, c]])
    if np.ptp(p[:, 1]) > np.ptp(p[:, 0]):
        p = p[:, ::-1] * np.array([1.0, -1.0])
    return p


def stretch_from(genome):
    return STRETCH_MID + STRETCH_SPAN * np.tanh(0.2 * float(np.sum(genome[1::3])))


def colour_from(genome):
    hue = (float(np.sum(genome[0::3])) * 0.137) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


class Car:
    def __init__(self, shape, mass, accel, max_speed, max_steer, color):
        self.shape = shape
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
    blob = orient(cppn.ring_shape(genome, cfg.CAR_CPPN_LAYERS, cfg.CAR_POINTS,
                                  cfg.CAR_R_MIN, cfg.CAR_R_MAX))
    k = np.sqrt(stretch_from(genome))
    shape = blob * np.array([k, 1.0 / k])

    length = float(np.ptp(shape[:, 0]))
    width = float(np.ptp(shape[:, 1]))

    filled = polygon_area(shape) / MAX_AREA
    mass = MASS_MIN + (MASS_MAX - MASS_MIN) * float(np.clip(filled, 0.0, 1.0))
    aspect = float(np.clip(length / width, ASPECT_MIN, ASPECT_MAX))
    stretch = (aspect - ASPECT_MIN) / (ASPECT_MAX - ASPECT_MIN)

    max_speed = SPEED_BASE + SPEED_GAIN * stretch
    max_steer = STEER_BASE - STEER_DROP * stretch
    accel = max_speed * cfg.DRAG * cfg.ACCEL_FACTOR
    return Car(shape, mass, accel, max_speed, max_steer, colour_from(genome))


def random_car(rng):
    return generate(cppn.random_genome(cfg.CAR_CPPN_LAYERS, rng, cfg.CAR_INIT_SCALE))
