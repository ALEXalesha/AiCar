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
GLASS = (178, 202, 226)
HEADLIGHT = (255, 246, 212)
TAILLIGHT = (218, 76, 68)
DEAD_GLASS = (94, 98, 108)

DETAIL_PX = {"wheel": 0.0, "body": 0.0, "cabin": 10.0, "glass": 10.0, "head": 26.0, "tail": 26.0}
RAY_LINE = (86, 198, 158)
LEADER_RING = (255, 238, 120)
TEXT = (222, 226, 236)
TEXT_DIM = (138, 146, 164)

PATH_WIDTH = 2

HUD_W, HUD_H, HUD_PAD = 236, 122, 12
HUD_BG = (22, 25, 32)
BAR_BG = (54, 59, 71)
BAR_NEAR = (222, 96, 88)
BAR_FAR = (86, 198, 158)
BAR_GAS = (120, 200, 240)
BAR_BRAKE = (222, 140, 90)


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


def palette(veh, crashed=False):
    if crashed:
        return {"wheel": DEAD_WHEEL, "body": DEAD_CAR, "cabin": _darker(DEAD_CAR, 0.25),
                "glass": DEAD_GLASS, "head": DEAD_GLASS, "tail": DEAD_GLASS}
    return {"wheel": WHEEL, "body": veh.color, "cabin": _darker(veh.color, 0.16),
            "glass": GLASS, "head": HEADLIGHT, "tail": TAILLIGHT}


def draw_one_car(surf, cam, veh, pos, angle, colours, px):
    pts = _poly(cam, car_polygon(veh.stacked, pos, angle))
    for (start, end), kind in zip(veh.slices, veh.kinds):
        if px >= DETAIL_PX[kind]:
            pygame.draw.polygon(surf, colours[kind], pts[start:end])


def draw_cars(surf, cam, r, veh, leader=None):
    crashed = ~r.alive & (r.finish_step < 0)
    px = cam.scale * veh.length
    dead, live = palette(veh, True), palette(veh, False)

    for i in range(r.n):
        if crashed[i]:
            draw_one_car(surf, cam, veh, r.pos[i], r.angle[i], dead, px)

    nose = nose_point(veh)[None, :]
    tip = _lighter(veh.color)
    for i in range(r.n):
        if crashed[i]:
            continue
        draw_one_car(surf, cam, veh, r.pos[i], r.angle[i], live, px)
        if px < DETAIL_PX["cabin"]:
            x, y = cam.to_screen(car_polygon(nose, r.pos[i], r.angle[i]))[0]
            pygame.draw.circle(surf, tip, (int(x), int(y)), 2)

    if leader is not None:
        x, y = cam.to_screen(r.pos[leader])[0]
        radius = max(6, int(veh.length * cam.scale * 0.8))
        pygame.draw.circle(surf, LEADER_RING, (int(x), int(y)), radius, 2)


def draw_path(surf, cam, points, colour, width=PATH_WIDTH):
    if len(points) < 2:
        return
    pygame.draw.lines(surf, colour, False, _poly(cam, np.asarray(points)), width)


def draw_rays(surf, cam, pos, angle, rays):
    ang = angle + cfg.RAY_ANGLES
    tips = pos + np.stack([np.cos(ang), np.sin(ang)], axis=1) * rays[:, None]
    origin = cam.to_screen(pos)[0]
    for tip in cam.to_screen(tips):
        pygame.draw.line(surf, RAY_LINE, origin, tip, 1)


def _bar(surf, rect, share, colour, centred=False):
    pygame.draw.rect(surf, BAR_BG, rect, border_radius=2)
    if centred:
        half = rect.width * 0.5
        length = int(abs(share) * half)
        left = rect.centerx if share >= 0 else rect.centerx - length
        pygame.draw.rect(surf, colour, pygame.Rect(left, rect.top, max(length, 1), rect.height),
                         border_radius=2)
    else:
        pygame.draw.rect(surf, colour, pygame.Rect(rect.left, rect.top,
                                                   max(int(share * rect.width), 1), rect.height),
                         border_radius=2)


def hud_rect(view):
    return pygame.Rect(view.left + HUD_PAD, view.bottom - HUD_H - HUD_PAD, HUD_W, HUD_H)


def draw_telemetry(surf, box, font, veh, watch):
    pygame.draw.rect(surf, HUD_BG, box, border_radius=6)
    pygame.draw.rect(surf, MIDLINE, box, 1, border_radius=6)

    x, y = box.left + 10, box.top + 8
    state = "едет" if watch["alive"] else ("финиш" if watch["finished"] else "разбилась")
    surf.blit(font.render(f"машинка #{watch['index']}  {state}", True, veh.color), (x, y))
    grip = pygame.Rect(box.right - 26, box.top + 8, 16, 3)
    for row in range(3):
        pygame.draw.rect(surf, BAR_BG, grip.move(0, row * 5))
    y += 20

    surf.blit(font.render("датчики", True, TEXT_DIM), (x, y))
    y += 18
    slot = (HUD_W - 20) // cfg.N_RAYS
    for i, value in enumerate(watch["rays"]):
        share = float(np.clip(value, 0.0, 1.0))
        colour = BAR_NEAR if share < 0.25 else BAR_FAR
        height = 16
        rect = pygame.Rect(x + i * slot, y, slot - 3, height)
        _bar(surf, rect, share, colour)
    y += 24

    for label, value, colour in (("руль", watch["steer"], BAR_GAS),
                                 ("газ", watch["throttle"],
                                  BAR_GAS if watch["throttle"] >= 0 else BAR_BRAKE)):
        surf.blit(font.render(label, True, TEXT_DIM), (x, y - 2))
        _bar(surf, pygame.Rect(x + 44, y + 2, HUD_W - 64, 10), float(value), colour, centred=True)
        y += 20

    surf.blit(font.render(f"скорость {watch['speed']:.0f}   чекпоинт {watch['cp']}",
                          True, TEXT), (x, y - 2))


def draw_car_badge(surf, veh, at, scale, angle=0.0):
    spun = car_polygon(veh.stacked, np.asarray(at, dtype=float) / scale, angle) * scale
    pts = spun.astype(np.int32).tolist()
    colours = palette(veh)
    for (start, end), kind in zip(veh.slices, veh.kinds):
        pygame.draw.polygon(surf, colours[kind], pts[start:end])
