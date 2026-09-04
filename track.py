import numpy as np

import config as cfg
import cppn
import evolution

CURVATURE_SCALE = 0.25

FOLDED_SCORE = 1.0
VALID_BASE = 2.0


def smooth_closed(pts, w):
    if w <= 1:
        return pts.copy()
    kernel = np.ones(w) / w
    ext = np.vstack([pts[-w:], pts, pts[:w]])
    x = np.convolve(ext[:, 0], kernel, mode="same")[w:-w]
    y = np.convolve(ext[:, 1], kernel, mode="same")[w:-w]
    return np.stack([x, y], axis=1)


def centerline(genome, n_points=None, smooth_window=None):
    n_points = cfg.TRACK_POINTS if n_points is None else n_points
    smooth_window = cfg.SMOOTH_WINDOW if smooth_window is None else smooth_window
    pts = cppn.ring_shape(genome, cfg.TRACK_CPPN_LAYERS, n_points, cfg.R_MIN, cfg.R_MAX)
    return smooth_closed(pts, smooth_window)


def tangents(pts):
    t = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)


def normals(pts):
    t = tangents(pts)
    return np.stack([-t[:, 1], t[:, 0]], axis=1)


def offset_walls(center, half):
    n = normals(center)
    return center + n * half, center - n * half


def curvature(pts):
    nxt = np.roll(pts, -1, axis=0)
    prv = np.roll(pts, 1, axis=0)
    d1 = nxt - prv
    d2 = nxt - 2.0 * pts + prv
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = np.power(np.sum(d1 * d1, axis=1), 1.5) + 1e-12
    return num / den


def polyline_length(pts):
    return float(np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)))


def build_checkpoints(left, right, step):
    idx = np.arange(0, len(left), step)
    return np.stack([left[idx], right[idx]], axis=1)


def any_self_intersection(poly, step=4):
    p = poly[::step]
    n = len(p)
    r = np.roll(p, -1, axis=0) - p

    rr = r[:, None, :]
    ss = r[None, :, :]
    denom = rr[..., 0] * ss[..., 1] - rr[..., 1] * ss[..., 0]
    safe = np.where(np.abs(denom) < 1e-12, np.nan, denom)

    qp = p[None, :, :] - p[:, None, :]
    t = (qp[..., 0] * ss[..., 1] - qp[..., 1] * ss[..., 0]) / safe
    u = (qp[..., 0] * rr[..., 1] - qp[..., 1] * rr[..., 0]) / safe

    eps = 1e-9
    hit = (t > -eps) & (t < 1.0 + eps) & (u > -eps) & (u < 1.0 + eps)

    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    neighbour = (gap <= 1) | (gap >= n - 1)
    return bool(np.any(hit & ~neighbour))


def smooth_signal(v, w):
    if w <= 1:
        return v.copy()
    ext = np.concatenate([v[-w:], v, v[:w]])
    return np.convolve(ext, np.ones(w) / w, mode="same")[w:-w]


def shape_variety(pts):
    r = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    return float(r.std() / (r.mean() + 1e-9))


def corner_variety(pts):
    return float(smooth_signal(curvature(pts), cfg.CURVATURE_WINDOW).std())


def interest(pts):
    return cfg.SHAPE_WEIGHT * shape_variety(pts) + cfg.VARIETY_SCALE * corner_variety(pts)


def min_radius(pts):
    return CURVATURE_SCALE / (float(curvature(pts).max()) + 1e-12)


def radius_factor(difficulty):
    d = float(np.clip(difficulty, 0.0, 1.0))
    return cfg.MIN_RADIUS_EASY + (cfg.MIN_RADIUS_HARD - cfg.MIN_RADIUS_EASY) * d


def walls_are_sane(center, width, difficulty=None):
    difficulty = cfg.DIFFICULTY if difficulty is None else difficulty
    if min_radius(center) < width * radius_factor(difficulty):
        return False
    left, right = offset_walls(center, width * 0.5)
    return not (any_self_intersection(left) or any_self_intersection(right))


def track_fitness(center, width, difficulty):
    k = curvature(center)
    radius = CURVATURE_SCALE / (float(k.max()) + 1e-12)
    needed = width * radius_factor(difficulty)
    if radius < needed:
        return FOLDED_SCORE * radius / needed

    length = polyline_length(center)
    target = 0.5 * (cfg.LEN_MIN + cfg.LEN_MAX)
    if cfg.LEN_MIN <= length <= cfg.LEN_MAX:
        len_factor = 1.0
    else:
        len_factor = max(0.05, 1.0 - abs(length - target) / target)

    left, right = offset_walls(center, width * 0.5)
    if any_self_intersection(left) or any_self_intersection(right):
        return FOLDED_SCORE

    return VALID_BASE + interest(center) * len_factor


def circle_centerline(n_points=cfg.TRACK_POINTS):
    r = 0.5 * (cfg.R_MIN + cfg.R_MAX)
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


class Track:
    def __init__(self, center, width):
        self.center = center
        self.width = width
        self.half = width * 0.5
        self.left, self.right = offset_walls(center, self.half)
        self.checkpoints = build_checkpoints(self.left, self.right, cfg.CHECKPOINT_STEP)
        self.cp_idx = np.arange(0, len(center), cfg.CHECKPOINT_STEP)
        self.cp_mid = center[self.cp_idx]
        self.cp_dir = tangents(center)[self.cp_idx]
        self.length = polyline_length(center)
        self.min_radius = min_radius(center)
        self.variety = interest(center)
        self.shape_variety = shape_variety(center)
        self.lo = center.min(axis=0) - self.half
        self.hi = center.max(axis=0) + self.half

    @property
    def start_pos(self):
        return self.center[0].copy()

    @property
    def start_angle(self):
        d = self.center[1] - self.center[0]
        return float(np.arctan2(d[1], d[0]))

    @property
    def n_checkpoints(self):
        return len(self.cp_mid)


def evolve_track(rng, width=None, difficulty=None, attempts=3):
    width = cfg.TRACK_WIDTH if width is None else width
    difficulty = cfg.DIFFICULTY if difficulty is None else difficulty
    for attempt in range(attempts):
        eased = max(0.0, difficulty - 0.25 * attempt)
        pop = np.stack([cppn.random_genome(cfg.TRACK_CPPN_LAYERS, rng)
                        for _ in range(cfg.TRACK_POP)])
        best_line, best_fit = None, -1.0
        for _ in range(cfg.TRACK_GENS):
            lines = [centerline(g) for g in pop]
            fits = np.array([track_fitness(c, width, eased) for c in lines])
            i = int(np.argmax(fits))
            if fits[i] > best_fit and walls_are_sane(lines[i], width, eased):
                best_fit, best_line = float(fits[i]), lines[i]
            pop = evolution.evolve(pop, fits, rng)
        if best_line is not None:
            return Track(best_line, width)
    return Track(circle_centerline(), width)
