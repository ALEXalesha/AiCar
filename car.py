import colorsys

import numpy as np

import config as cfg
import cppn

SPEED_BASE, SPEED_GAIN = 170.0, 90.0
STEER_BASE, STEER_DROP = 3.4, 1.5
MASS_MIN, MASS_MAX = 0.8, 2.5
ASPECT_MIN, ASPECT_MAX = 1.8, 3.2
STRETCH_MID, STRETCH_SPAN = 2.4, 0.6

BODY_POWER = 6.0
BODY_BUMP = 0.05
BUMP_SMOOTH = 9
NOSE_TAPER = 0.80
HALF_SIZE_MIN, HALF_SIZE_MAX = 7.0, 11.5

WHEEL_AT_X, WHEEL_AT_Y = 0.32, 0.47
WHEEL_LEN, WHEEL_WIDTH = 0.22, 0.17

CABIN_BACK, CABIN_FRONT, CABIN_HALF = -0.09, 0.06, 0.33
CABIN_POWER, CABIN_POINTS = 6.0, 16
GLASS_HALF = 0.29
WINDSHIELD_TIP, WINDSHIELD_NARROW = 0.27, 0.62
REAR_WINDOW_TIP, REAR_WINDOW_NARROW = -0.27, 0.80
LAMP_BACK, LAMP_FRONT = 0.41, 0.475
LAMP_AT_Y, LAMP_HALF = 0.215, 0.085


def _smooth_ring(values, w):
    if w <= 1:
        return values
    ext = np.concatenate([values[-w:], values, values[:w]])
    return np.convolve(ext, np.ones(w) / w, mode="same")[w:-w]


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
    bump = _smooth_ring(cppn.forward(genome, cfg.CAR_CPPN_LAYERS, features)[:, 0], BUMP_SMOOTH)
    taper = 1.0 - (1.0 - NOSE_TAPER) * np.clip(np.cos(phi), 0.0, 1.0)

    r = base * (1.0 + BODY_BUMP * bump) * taper
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


def _box(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _trapezoid(x_near, x_far, half_near, half_far):
    return np.array([[x_near, -half_near], [x_far, -half_far],
                     [x_far, half_far], [x_near, half_near]])


def _rounded_box(cx, half_l, half_w, power=CABIN_POWER, points=CABIN_POINTS):
    t = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    corner = (np.abs(np.cos(t)) / half_l) ** power + (np.abs(np.sin(t)) / half_w) ** power
    r = corner ** (-1.0 / power)
    return np.stack([cx + r * np.cos(t), r * np.sin(t)], axis=1)


def wheel_polygons(length, width):
    half_l, half_w = length * WHEEL_LEN * 0.5, width * WHEEL_WIDTH * 0.5
    box = _box(-half_l, half_l, -half_w, half_w)
    return [box + np.array([sx * length * WHEEL_AT_X, sy * width * WHEEL_AT_Y])
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]


def cabin_polygon(length, width):
    half_l = (CABIN_FRONT - CABIN_BACK) * 0.5 * length
    return _rounded_box((CABIN_FRONT + CABIN_BACK) * 0.5 * length, half_l, CABIN_HALF * width)


def windshield_polygon(length, width):
    half = GLASS_HALF * width
    return _trapezoid(CABIN_FRONT * length, WINDSHIELD_TIP * length, half, half * WINDSHIELD_NARROW)


def rear_window_polygon(length, width):
    half = GLASS_HALF * width
    return _trapezoid(CABIN_BACK * length, REAR_WINDOW_TIP * length, half, half * REAR_WINDOW_NARROW)


def lamp_polygons(length, width, front=True):
    side = 1.0 if front else -1.0
    x0, x1 = sorted((side * LAMP_BACK * length, side * LAMP_FRONT * length))
    half = LAMP_HALF * width
    return [_box(x0, x1, sy * LAMP_AT_Y * width - half, sy * LAMP_AT_Y * width + half)
            for sy in (-1.0, 1.0)]


def build_parts(shape, length, width):
    parts = [(poly, "wheel") for poly in wheel_polygons(length, width)]
    parts.append((shape, "body"))
    parts.append((cabin_polygon(length, width), "cabin"))
    parts.append((windshield_polygon(length, width), "glass"))
    parts.append((rear_window_polygon(length, width), "glass"))
    parts += [(poly, "head") for poly in lamp_polygons(length, width, front=True)]
    parts += [(poly, "tail") for poly in lamp_polygons(length, width, front=False)]
    return parts


class Car:
    def __init__(self, shape, parts, mass, accel, max_speed, max_steer, color):
        self.shape = shape
        self.parts = parts
        self.wheels = [poly for poly, kind in parts if kind == "wheel"]
        self.cockpit = next(poly for poly, kind in parts if kind == "cabin")
        self.kinds = [kind for _, kind in parts]
        self.stacked = np.vstack([poly for poly, _ in parts])
        edges = np.cumsum([0] + [len(poly) for poly, _ in parts])
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
    shape[:, 0] -= 0.5 * (shape[:, 0].max() + shape[:, 0].min())
    max_area = np.pi * (HALF_SIZE_MAX * 1.2) ** 2
    filled = polygon_area(shape) / max_area
    mass = MASS_MIN + (MASS_MAX - MASS_MIN) * float(np.clip(filled, 0.0, 1.0))

    reach = (aspect - ASPECT_MIN) / (ASPECT_MAX - ASPECT_MIN)
    max_speed = SPEED_BASE + SPEED_GAIN * reach
    max_steer = STEER_BASE - STEER_DROP * reach
    accel = max_speed * cfg.DRAG * cfg.ACCEL_FACTOR

    extent_l = float(np.ptp(shape[:, 0]))
    extent_w = float(np.ptp(shape[:, 1]))
    parts = build_parts(shape, extent_l, extent_w)
    return Car(shape, parts, mass, accel, max_speed, max_steer, colour_from(genome))


def random_car(rng):
    return generate(cppn.random_genome(cfg.CAR_CPPN_LAYERS, rng, cfg.CAR_INIT_SCALE))
