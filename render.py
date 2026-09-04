import numpy as np
import pygame

import config as cfg

BG = (16, 18, 24)
PANEL_BG = (26, 29, 37)
ASPHALT = (48, 52, 63)
WALL = (126, 136, 158)
MIDLINE = (72, 78, 93)
CHECK_TAKEN = (58, 112, 84)
CHECK_NEXT = (206, 184, 92)
START_LINE = (232, 234, 242)
DEAD_CAR = (74, 74, 84)
WHEEL = (22, 23, 29)
DEAD_WHEEL = (46, 46, 54)
GLASS = (196, 214, 232)
RAY_LINE = (86, 198, 158)
LEADER_RING = (255, 238, 120)
TEXT = (222, 226, 236)
TEXT_DIM = (138, 146, 164)

SHOWCASE_SCALE = 2.4


class Camera:
    def __init__(self, rect, lo, hi, margin=24.0):
        self.rect = rect
        self.margin = margin
        self.fit(lo, hi)

    def fit(self, lo, hi):
        span = np.maximum(np.asarray(hi) - np.asarray(lo), 1e-6)
        room = np.array([self.rect.width - 2.0 * self.margin,
                         self.rect.height - 2.0 * self.margin])
        self.scale = float(np.min(room / span))
        self.center = (np.asarray(lo) + np.asarray(hi)) * 0.5

    def follow(self, point, scale):
        self.center = np.asarray(point, dtype=float)
        self.scale = scale

    def to_screen(self, pts):
        p = (np.atleast_2d(pts) - self.center) * self.scale
        return np.stack([self.rect.centerx + p[:, 0], self.rect.centery - p[:, 1]], axis=1)

    def to_world(self, pt):
        dx = pt[0] - self.rect.centerx
        dy = self.rect.centery - pt[1]
        return self.center + np.array([dx, dy]) / self.scale


def _poly(cam, pts):
    return cam.to_screen(pts).astype(np.int32).tolist()


def draw_track(surf, cam, trk):
    ribbon = np.vstack([trk.left, trk.right[::-1]])
    pygame.draw.polygon(surf, ASPHALT, _poly(cam, ribbon))
    pygame.draw.lines(surf, WALL, True, _poly(cam, trk.left), 2)
    pygame.draw.lines(surf, WALL, True, _poly(cam, trk.right), 2)
    pygame.draw.lines(surf, MIDLINE, True, _poly(cam, trk.center), 1)

    start = np.stack([trk.left[0], trk.right[0]])
    pygame.draw.line(surf, START_LINE, *_poly(cam, start), 3)


def draw_checkpoints(surf, cam, trk, taken=0):
    for i, seg in enumerate(trk.checkpoints):
        if i == 0:
            continue
        colour = CHECK_TAKEN if i <= taken else CHECK_NEXT if i == taken + 1 else None
        if colour is not None:
            pygame.draw.line(surf, colour, *_poly(cam, seg), 1)


def car_polygon(shape, pos, angle):
    c, s = np.cos(angle), np.sin(angle)
    return shape @ np.array([[c, s], [-s, c]]) + pos


def nose_point(veh):
    return veh.shape[int(np.argmax(veh.shape[:, 0]))]


def _lighter(colour, k=0.55):
    return tuple(int(c + (255 - c) * k) for c in colour)


def _darker(colour, k=0.45):
    return tuple(int(c * (1.0 - k)) for c in colour)


def draw_one_car(surf, cam, veh, pos, angle, body, wheel, glass=None):
    pts = _poly(cam, car_polygon(veh.stacked, pos, angle))
    colours = [wheel] * 4 + [body, glass]
    for (start, end), colour in zip(veh.slices, colours):
        if colour is not None:
            pygame.draw.polygon(surf, colour, pts[start:end])


def draw_cars(surf, cam, r, veh, leader=None):
    crashed = ~r.alive & (r.finish_step < 0)
    detailed = cam.scale * veh.length > 26.0
    glass = GLASS if detailed else None

    for i in range(r.n):
        if crashed[i]:
            draw_one_car(surf, cam, veh, r.pos[i], r.angle[i], DEAD_CAR, DEAD_WHEEL)

    nose = nose_point(veh)[None, :]
    tip = _lighter(veh.color)
    for i in range(r.n):
        if crashed[i]:
            continue
        draw_one_car(surf, cam, veh, r.pos[i], r.angle[i], veh.color, WHEEL, glass)
        if not detailed:
            x, y = cam.to_screen(car_polygon(nose, r.pos[i], r.angle[i]))[0]
            pygame.draw.circle(surf, tip, (int(x), int(y)), 2)

    if leader is not None:
        x, y = cam.to_screen(r.pos[leader])[0]
        radius = max(6, int(veh.length * cam.scale * 0.8))
        pygame.draw.circle(surf, LEADER_RING, (int(x), int(y)), radius, 2)


def draw_rays(surf, cam, pos, angle, rays):
    ang = angle + cfg.RAY_ANGLES
    tips = pos + np.stack([np.cos(ang), np.sin(ang)], axis=1) * rays[:, None]
    origin = cam.to_screen(pos)[0]
    for tip in cam.to_screen(tips):
        pygame.draw.line(surf, RAY_LINE, origin, tip, 1)


def draw_car_badge(surf, veh, at, scale):
    pts = (veh.stacked * scale + np.asarray(at, dtype=float)).astype(np.int32).tolist()
    for (start, end), colour in zip(veh.slices, [WHEEL] * 4 + [veh.color, GLASS]):
        pygame.draw.polygon(surf, colour, pts[start:end])
