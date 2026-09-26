"""Камера и отрисовка через QPainter со сглаживанием (с 2.0.0; до того - pygame).

Модуль не хранит состояние игры и ничего не решает. Ему дают трассу, заезд и QPainter,
он рисует. Цвета - кортежи RGB, как были: по ним проверяется палитра; в QColor они
переводятся один раз (`qc`).
"""
import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QImage, QPainter, QPen, QPixmap,
                           QPolygonF, QTransform)

import config as cfg
from rect import Rect

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

HUD_W, HUD_H, HUD_PAD = 252, 134, 12
GRIP_W, HUD_GAP, CLOSE_W = 16, 12, 13
HUD_BG = (22, 25, 32)
BAR_BG = (54, 59, 71)
BAR_NEAR = (222, 96, 88)
BAR_FAR = (86, 198, 158)
BAR_GAS = (120, 200, 240)
BAR_BRAKE = (222, 140, 90)

FONT_FAMILY = "Consolas"

_COLOURS = {}


def qc(colour):
    """QColor из кортежа RGB, один раз на цвет (зовётся сотни раз за кадр)."""
    got = _COLOURS.get(colour)
    if got is None:
        got = _COLOURS[colour] = QColor(*(int(c) for c in colour))
    return got


# --- шрифт и текст ---------------------------------------------------------------------

def font(pixels, bold=False):
    """Моноширинный Consolas размером в пикселях - как SysFont("consolas", 15) у pygame."""
    f = QFont(FONT_FAMILY)
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPixelSize(int(pixels))
    f.setBold(bold)
    return f


_METRICS = {}


def metrics(f):
    """Метрики шрифта, одни на объект шрифта. Ключ - сам объект (id и ссылка, чтобы id
    не достался другому шрифту): font.key() строит строку и дорог для сотни вызовов в кадр."""
    got = _METRICS.get(id(f))
    if got is None or got[0] is not f:
        if len(_METRICS) > 64:
            _METRICS.clear()
        got = _METRICS[id(f)] = (f, QFontMetricsF(f))
    return got[1]


def text_width(f, text):
    """Ширина строки в пикселях. Мерить надо её, а не число символов (docs/bug-hunt.md)."""
    return metrics(f).horizontalAdvance(text)


def line_height(f):
    return metrics(f).height()


def fit_text(f, text, width, tail="…"):
    if f is None or text_width(f, text) <= width:
        return text
    lo, hi = 0, len(text)            # самая длинная голова, которая влезает вместе с хвостом
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text_width(f, text[:mid] + tail) <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + tail


def text(p, f, s, colour, x, y):
    """Строка левым верхним углом в (x, y) - как blit поверхности со шрифтом у pygame."""
    p.setFont(f)
    p.setPen(qc(colour))
    p.drawText(QPointF(x, y + metrics(f).ascent()), s)


# --- холст -----------------------------------------------------------------------------

def painter(device):
    p = QPainter(device)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    return p


def canvas(w, h, ground=BG):
    img = QImage(int(w), int(h), QImage.Format.Format_RGB32)
    img.fill(qc(ground))
    return img


def pixels(image):
    """Картинка в numpy (ширина, высота, 3) - раскладка [x, y], как pygame.surfarray."""
    img = image.convertToFormat(QImage.Format.Format_RGB32)
    w, h, line = img.width(), img.height(), img.bytesPerLine()
    raw = np.frombuffer(img.constBits(), np.uint8, count=line * h).reshape(h, line // 4, 4)
    return np.ascontiguousarray(raw[:, :w, 2::-1].transpose(1, 0, 2))


def ink(image, ground):
    """Маска (ширина, высота): True там, где что-то нарисовано поверх фона."""
    return np.any(pixels(image) != np.array(ground, dtype=np.uint8), axis=2)


# --- многоугольники из numpy -----------------------------------------------------------

def _fast_fill(poly, pts):
    """Записать точки прямо в память QPolygonF: в сотни раз быстрее списка QPointF,
    а путь заезда - до 1800 точек на кадр. Не вышло - вернуть False."""
    try:
        import shiboken6
        buf = shiboken6.VoidPtr(poly.data(), len(pts) * 16, True)
        np.frombuffer(buf, np.float64).reshape(-1, 2)[:] = pts
    except Exception:  # noqa: BLE001 - другая сборка PySide: медленный путь ниже
        return False
    last = poly.at(len(pts) - 1)
    return last.x() == pts[-1, 0] and last.y() == pts[-1, 1]


def qpolygon(pts):
    pts = np.ascontiguousarray(np.asarray(pts, dtype=np.float64).reshape(-1, 2))
    poly = QPolygonF()
    if len(pts) == 0:
        return poly
    poly.resize(len(pts))
    if not _fast_fill(poly, pts):
        poly = QPolygonF([QPointF(x, y) for x, y in pts.tolist()])
    return poly


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

    def transform(self):
        """Мир -> экран одной матрицей: то же, что to_screen, для QPainter."""
        t = QTransform()
        t.translate(self.rect.centerx - self.center[0] * self.scale,
                    self.rect.centery + self.center[1] * self.scale)
        t.scale(self.scale, -self.scale)
        return t

    def key(self):
        return (self.rect.width, self.rect.height, self.rect.centerx, self.rect.centery,
                round(self.scale, 9), round(float(self.center[0]), 6), round(float(self.center[1]), 6))


def _pen(colour, width, cap=Qt.PenCapStyle.RoundCap):
    pen = QPen(qc(colour), width)
    pen.setCapStyle(cap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCosmetic(True)           # толщина в пикселях экрана при любом масштабе мира
    return pen


def draw_track(p, cam, trk):
    p.save()
    p.setTransform(cam.transform(), True)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(qc(ASPHALT))
    p.drawPolygon(qpolygon(np.vstack([trk.left, trk.right[::-1]])), Qt.FillRule.WindingFill)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(_pen(WALL, 2))
    p.drawPolygon(qpolygon(trk.left))
    p.drawPolygon(qpolygon(trk.right))
    p.setPen(_pen(MIDLINE, 1))
    p.drawPolygon(qpolygon(trk.center))
    p.setPen(_pen(START_LINE, 3, Qt.PenCapStyle.FlatCap))
    p.drawLine(QLineF(*trk.left[0], *trk.right[0]))
    p.restore()


class TrackLayer:
    """Трасса нарисована один раз в картинку и дальше только копируется: лента из 720
    точек и три замкнутые линии со сглаживанием - самое дорогое в кадре, а меняется
    только с трассой или с размером поля."""

    def __init__(self):
        self.key = None
        self.pixmap = None

    def draw(self, p, cam, trk):
        dpr = p.device().devicePixelRatioF() if p.device() is not None else 1.0
        key = (id(trk), cam.key(), dpr)
        if key != self.key:
            rect = cam.rect
            pix = QPixmap(max(1, int(rect.width * dpr)), max(1, int(rect.height * dpr)))
            pix.setDevicePixelRatio(dpr)
            pix.fill(qc(BG))
            q = painter(pix)
            q.translate(-rect.left, -rect.top)
            draw_track(q, cam, trk)
            q.end()
            self.pixmap, self.key = pix, key
        p.drawPixmap(QPointF(cam.rect.left, cam.rect.top), self.pixmap)


def draw_checkpoints(p, cam, trk, taken=0):
    lines = {CHECK_TAKEN: [], CHECK_NEXT: []}
    for i, seg in enumerate(trk.checkpoints):
        if i == 0:
            continue
        colour = CHECK_TAKEN if i <= taken else CHECK_NEXT if i == taken + 1 else None
        if colour is not None:
            (x0, y0), (x1, y1) = cam.to_screen(seg)
            lines[colour].append(QLineF(x0, y0, x1, y1))
    for colour, segs in lines.items():
        if segs:
            p.setPen(_pen(colour, 1, Qt.PenCapStyle.FlatCap))
            p.drawLines(segs)


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


_SHAPES = {}


class CarShape:
    """Детали машинки в её собственных координатах, построенные один раз на машинку.
    Подряд идущие детали одного вида (четыре колеса, два стекла, пары фар) слиты в один
    путь: деталей двенадцать, а рисуется шесть. Порядок видов тот же, что в veh.kinds, -
    колёса под кузовом, стёкла и фары поверх."""

    def __init__(self, veh):
        from PySide6.QtGui import QPainterPath
        self.groups = []                 # (вид, QPainterPath)
        for (a, b), kind in zip(veh.slices, veh.kinds):
            if not self.groups or self.groups[-1][0] != kind:
                path = QPainterPath()
                path.setFillRule(Qt.FillRule.WindingFill)
                self.groups.append((kind, path))
            self.groups[-1][1].addPolygon(qpolygon(veh.stacked[a:b]))
            self.groups[-1][1].closeSubpath()
        self.brushes = {}

    def brush_list(self, colours):
        key = tuple(colours[kind] for kind, _ in self.groups)
        got = self.brushes.get(key)
        if got is None:
            got = self.brushes[key] = [qc(colours[kind]) for kind, _ in self.groups]
        return got


def car_shape(veh):
    got = _SHAPES.get(id(veh))
    if got is None or got[0] is not veh:
        if len(_SHAPES) > 16:
            _SHAPES.clear()
        got = _SHAPES[id(veh)] = (veh, CarShape(veh))
    return got[1]


def car_transforms(cam, pos, angle):
    """Матрицы «своё -> экран» для всех машинок разом: поворот, масштаб, переворот Y, сдвиг.
    Qt: x' = m11 x + m21 y + dx, y' = m12 x + m22 y + dy."""
    scr = cam.to_screen(np.asarray(pos, dtype=float).reshape(-1, 2))
    ang = np.asarray(angle, dtype=float).reshape(-1)
    c, s = np.cos(ang) * cam.scale, np.sin(ang) * cam.scale
    return [QTransform(ci, -si, -si, -ci, x, y)
            for ci, si, x, y in zip(c.tolist(), s.tolist(), scr[:, 0].tolist(), scr[:, 1].tolist())]


def _draw_group(p, shape, transforms, brushes, visible, base):
    for t in transforms:
        p.setTransform(t * base)
        for (kind, path), brush, show in zip(shape.groups, brushes, visible):
            if show:
                p.setBrush(brush)
                p.drawPath(path)


def draw_one_car(p, cam, veh, pos, angle, colours, px):
    shape = car_shape(veh)
    visible = [px >= DETAIL_PX[kind] for kind, _ in shape.groups]
    base = p.transform()
    _draw_group(p, shape, car_transforms(cam, pos, angle), shape.brush_list(colours), visible, base)
    p.setTransform(base)


def draw_cars(p, cam, r, veh, leader=None):
    crashed = ~r.alive & (r.finish_step < 0)
    px = cam.scale * veh.length
    dead, live = palette(veh, True), palette(veh, False)
    shape = car_shape(veh)
    visible = [px >= DETAIL_PX[kind] for kind, _ in shape.groups]
    p.save()
    p.setPen(Qt.PenStyle.NoPen)
    base = p.transform()

    wrecked = np.flatnonzero(crashed)
    if len(wrecked):
        _draw_group(p, shape, car_transforms(cam, r.pos[wrecked], r.angle[wrecked]),
                    shape.brush_list(dead), visible, base)
    racing = np.flatnonzero(~crashed)
    if len(racing):
        _draw_group(p, shape, car_transforms(cam, r.pos[racing], r.angle[racing]),
                    shape.brush_list(live), visible, base)
    p.setTransform(base)
    if px < DETAIL_PX["cabin"] and len(racing):
        nose = nose_point(veh)[None, :]
        p.setBrush(qc(_lighter(veh.color)))
        for i in racing:
            x, y = cam.to_screen(car_polygon(nose, r.pos[i], r.angle[i]))[0]
            p.drawEllipse(QPointF(x, y), 2.0, 2.0)

    if leader is not None:
        x, y = cam.to_screen(r.pos[leader])[0]
        radius = max(6, int(veh.length * cam.scale * 0.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(_pen(LEADER_RING, 2))
        p.drawEllipse(QPointF(x, y), radius, radius)
    p.restore()


def draw_path(p, cam, points, colour, width=PATH_WIDTH):
    if len(points) < 2:
        return
    p.save()
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(_pen(colour, width))
    p.drawPolyline(qpolygon(cam.to_screen(np.asarray(points, dtype=float))))
    p.restore()


def draw_rays(p, cam, pos, angle, rays):
    ang = angle + cfg.RAY_ANGLES
    tips = pos + np.stack([np.cos(ang), np.sin(ang)], axis=1) * rays[:, None]
    ox, oy = cam.to_screen(pos)[0]
    p.save()
    p.setPen(_pen(RAY_LINE, 1, Qt.PenCapStyle.FlatCap))
    p.drawLines([QLineF(ox, oy, x, y) for x, y in cam.to_screen(tips)])
    p.restore()


def _rounded(p, rect, colour, radius):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(qc(colour))
    p.drawRoundedRect(QRectF(rect.left, rect.top, rect.width, rect.height), radius, radius)


def _bar(p, rect, share, colour, centred=False):
    _rounded(p, rect, BAR_BG, 2)
    if centred:
        half = rect.width * 0.5
        length = int(abs(share) * half)
        left = rect.centerx if share >= 0 else rect.centerx - length
        _rounded(p, Rect(left, rect.top, max(length, 1), rect.height), colour, 2)
    else:
        _rounded(p, Rect(rect.left, rect.top, max(int(share * rect.width), 1), rect.height), colour, 2)


def hud_rect(view):
    return Rect(view.left + HUD_PAD, view.bottom - HUD_H - HUD_PAD, HUD_W, HUD_H)


def hud_close_rect(box):
    """Крестик слева от хвата: закрывает слежение вместе с окошком."""
    return Rect(box.right - 10 - GRIP_W - 10 - CLOSE_W, box.top + 6, CLOSE_W, CLOSE_W)


def draw_telemetry(p, box, f, veh, watch):
    p.save()
    # Рамка в 1 пиксель - по пикселям внутри окошка: линия по самому краю легла бы
    # половиной за его пределы.
    frame = QRectF(box.left + 0.5, box.top + 0.5, box.width - 1, box.height - 1)
    p.setBrush(qc(HUD_BG))
    p.setPen(QPen(qc(MIDLINE), 1))
    p.drawRoundedRect(frame, 6, 6)
    p.setClipRect(QRectF(box.left, box.top, box.width, box.height))

    x, y = box.left + 10, box.top + 8
    inner = box.width - 20
    state = "едет" if watch["alive"] else ("финиш" if watch["finished"] else "разбилась")
    title = fit_text(f, f"машинка #{watch['index']}  {state}", inner - GRIP_W - CLOSE_W - 20)
    text(p, f, title, veh.color, x, y)
    grip = Rect(box.right - 10 - GRIP_W, box.top + 8, GRIP_W, 3)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(qc(BAR_BG))
    for row in range(3):
        p.drawRect(grip.move(0, row * 5).qrectf())

    cross = hud_close_rect(box)
    pen = QPen(qc(TEXT_DIM), 2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    a, b = cross.left + 1, cross.right - 1
    top, bottom = cross.top + 1, cross.bottom - 1
    p.drawLine(QLineF(a, top, b, bottom))
    p.drawLine(QLineF(a, bottom, b, top))
    y += 20

    text(p, f, "датчики", TEXT_DIM, x, y)
    y += 18
    slot = inner // cfg.N_RAYS
    for i, value in enumerate(watch["rays"]):
        share = float(np.clip(value, 0.0, 1.0))
        colour = BAR_NEAR if share < 0.25 else BAR_FAR
        _bar(p, Rect(x + i * slot, y, slot - 3, 16), share, colour)
    y += 24

    for label, value, colour in (("руль", watch["steer"], BAR_GAS),
                                 ("газ", watch["throttle"],
                                  BAR_GAS if watch["throttle"] >= 0 else BAR_BRAKE)):
        text(p, f, label, TEXT_DIM, x, y - 2)
        _bar(p, Rect(x + 44, y + 2, inner - 44, 10), float(value), colour, centred=True)
        y += 20

    left = fit_text(f, f"скорость {watch['speed']:.0f}", inner)
    right = fit_text(f, f"чекпоинт {watch['cp']}", inner - text_width(f, left) - HUD_GAP)
    text(p, f, left, TEXT, x, y - 2)
    text(p, f, right, TEXT, box.right - 10 - text_width(f, right), y - 2)
    p.restore()


def draw_car_badge(p, veh, at, scale, angle=0.0):
    colours = palette(veh)
    base = p.transform()
    t = QTransform()
    t.translate(float(at[0]), float(at[1]))
    t.scale(scale, scale)
    t.rotateRadians(float(angle))
    p.setTransform(t, True)
    p.setPen(Qt.PenStyle.NoPen)
    shape = car_shape(veh)
    for (kind, path), brush in zip(shape.groups, shape.brush_list(colours)):
        p.setBrush(brush)
        p.drawPath(path)
    p.setTransform(base)


def icon_image(veh, size=256):
    """Значок окна и установщика: машинка носом вверх на тёмном скруглённом квадрате."""
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = painter(img)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(qc(PANEL_BG))
    p.drawRoundedRect(QRectF(0, 0, size, size), size / 6, size / 6)
    scale = size * 0.62 / max(veh.length, veh.width)
    draw_car_badge(p, veh, (size / 2, size / 2), scale, -np.pi / 2)
    p.end()
    return img
