"""Боковая панель: ползунки, переключатели, кнопки, график (рисуются QPainter, с 2.0.0).

Виджеты свои, а не QSlider/QPushButton: вид панели и её законы раскладки остались от
pygame-версии, и на них держатся свойства stress.py. Геометрия - `rect.Rect` с правилами
pygame, события мыши - `Mouse` (окно переводит в них QMouseEvent, тесты создают напрямую).

Панель по высоте сжимается (`Panel.layout`): у каждого элемента естественная высота и
наименьшая. В окне 1280x720 раскладка почти естественная, в окне минимальной высоты все
элементы сжаты одинаково - и всё равно не наезжают друг на друга и не вылезают.
"""
from collections import namedtuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPen

import render
from rect import Rect

TRACK_H = 5
KNOB_W = 9
SLIDER_H, SLIDER_LEAST = 32, 28
BUTTON_H, BUTTON_LEAST = 26, 23
TOGGLE_H, TOGGLE_LEAST = 26, 23
GAP, GAP_LEAST = 3, 1
PAD_LEAST = 8
BOTTOM = 3            # просвет под последним рядом

FILL = (58, 64, 78)
ACTIVE = (110, 190, 230)
KNOB = (206, 214, 230)
HOVER = (74, 82, 99)

DOWN, UP, MOVE = "down", "up", "move"
Mouse = namedtuple("Mouse", "kind pos button")


def press(pos, button=1):
    return Mouse(DOWN, pos, button)


def release(pos, button=1):
    return Mouse(UP, pos, button)


def move(pos):
    return Mouse(MOVE, pos, 0)


def _hovered(rect, mouse):
    return mouse is not None and rect.collidepoint(mouse)


def _text_centre_y(rect, f):
    return rect.top + (rect.height - render.line_height(f)) / 2.0


class Slider:
    height = SLIDER_H

    def __init__(self, rect, label, lo, hi, value, integer=False, fmt="{:.2f}"):
        self.rect = rect
        self.label = label
        self.lo, self.hi = float(lo), float(hi)
        self.integer = integer
        self.fmt = "{:.0f}" if integer else fmt
        self.value = value
        self.dragging = False

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        v = float(np.clip(v, self.lo, self.hi))
        self._value = int(round(v)) if self.integer else v

    @property
    def shown(self):
        return self.fmt.format(self.value)

    @property
    def bar(self):
        return Rect(self.rect.left, self.rect.bottom - TRACK_H - 4, self.rect.width, TRACK_H)

    def _from_x(self, x):
        bar = self.bar
        t = np.clip((x - bar.left) / max(bar.width, 1), 0.0, 1.0)
        self.value = self.lo + t * (self.hi - self.lo)

    def handle(self, event):
        if event.kind == DOWN and event.button == 1:
            grab = self.rect.inflate(0, 6)
            if grab.collidepoint(event.pos):
                self.dragging = True
                self._from_x(event.pos[0])
                return True
        elif event.kind == UP and event.button == 1:
            self.dragging = False
        elif event.kind == MOVE and self.dragging:
            self._from_x(event.pos[0])
            return True
        return False

    def draw(self, p, f, mouse=None):
        render.text(p, f, self.label, render.TEXT_DIM, self.rect.left, self.rect.top)
        shown = self.shown
        render.text(p, f, shown, render.TEXT, self.rect.right - render.text_width(f, shown), self.rect.top)

        bar = self.bar
        t = (self.value - self.lo) / max(self.hi - self.lo, 1e-9)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(render.qc(FILL))
        p.drawRoundedRect(bar.qrectf(), 2, 2)
        p.setBrush(render.qc(ACTIVE))
        p.drawRoundedRect(QRectF(bar.left, bar.top, bar.width * t, bar.height), 2, 2)
        knob = QRectF(0, 0, KNOB_W, TRACK_H + 8)
        knob.moveCenter(QPointF(bar.left + bar.width * t, bar.top + bar.height / 2.0))
        p.setBrush(render.qc(KNOB))
        p.drawRoundedRect(knob, 3, 3)


class Button:
    height = BUTTON_H

    def __init__(self, rect, label):
        self.rect = rect
        self.label = label
        self.pressed = False
        self.fired = False

    def handle(self, event):
        if event.kind == DOWN and event.button == 1:
            self.pressed = self.rect.collidepoint(event.pos)
        elif event.kind == UP and event.button == 1:
            hit = self.pressed and self.rect.collidepoint(event.pos)
            self.pressed = False
            if hit:
                self.fired = True
                return True
        return False

    def take(self):
        fired, self.fired = self.fired, False
        return fired

    def draw(self, p, f, mouse=None):
        colour = ACTIVE if self.pressed else HOVER if _hovered(self.rect, mouse) else FILL
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(render.qc(colour))
        p.drawRoundedRect(self.rect.qrectf(), 4, 4)
        x = self.rect.left + (self.rect.width - render.text_width(f, self.label)) / 2.0
        render.text(p, f, self.label, render.TEXT, x, _text_centre_y(self.rect, f))


class Toggle:
    height = TOGGLE_H

    def __init__(self, rect, label, options, index=0):
        self.rect = rect
        self.label = label
        self.options = list(options)
        self.index = index

    @property
    def value(self):
        return self.options[self.index]

    def handle(self, event):
        if event.kind == UP and event.button == 1 and self.rect.collidepoint(event.pos):
            self.index = (self.index + 1) % len(self.options)
            return True
        return False

    def draw(self, p, f, mouse=None):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(render.qc(HOVER if _hovered(self.rect, mouse) else FILL))
        p.drawRoundedRect(self.rect.qrectf(), 4, 4)
        y = _text_centre_y(self.rect, f)
        value = str(self.value)
        value_w = render.text_width(f, value)
        # Подпись уступает значению: если вместе не влезли (шрифт крупнее, чем задуман),
        # обрезается подпись, а не наезжает на значение.
        room = self.rect.width - 16 - value_w - render.text_width(f, " ")
        render.text(p, f, render.fit_text(f, self.label, room), render.TEXT_DIM, self.rect.left + 8, y)
        render.text(p, f, value, ACTIVE, self.rect.right - value_w - 8, y)


class Graph:
    def __init__(self, rect):
        self.rect = rect

    def draw(self, p, history):
        p.save()
        box = self.rect
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(render.qc(render.BG))
        p.drawRect(box.qrectf())
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(render.qc(render.MIDLINE), 1))
        p.drawRect(QRectF(box.left + 0.5, box.top + 0.5, box.width - 1, box.height - 1))
        if len(history) >= 2:
            best = np.array([s.best for s in history])
            mean = np.array([s.mean for s in history])
            top = max(float(best.max()) * 1.08, 1.0)
            xs = np.linspace(box.left + 1.5, box.right - 1.5, len(best))
            p.setClipRect(QRectF(box.left + 1, box.top + 1, box.width - 2, box.height - 2))
            for values, colour in ((mean, render.TEXT_DIM), (best, render.LEADER_RING)):
                ys = box.bottom - 1.5 - values / top * (box.height - 3)
                pen = QPen(render.qc(colour), 2)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.setPen(pen)
                p.drawPolyline(render.qpolygon(np.stack([xs, ys], axis=1)))
        p.restore()


class Panel:
    """Столбец виджетов. Строится сверху вниз (skip, graph, slider, toggle, buttons),
    `layout(rect)` раскладывает заново под другую высоту."""

    def __init__(self, rect, f, pad=14):
        self.rect = rect
        self.font = f
        self.pad = pad
        self.top_pad = pad
        self.squeeze = 1.0       # 1 - естественная раскладка, 0 - сжата до предела
        self._cache = self._cache_key = None
        self.widgets = {}
        self.items = []          # (естественная высота, наименьшая, [(ключ, доля ширины)] или None)

    @property
    def inner_width(self):
        return self.rect.width - 2 * self.pad

    # --- построение -------------------------------------------------------------------

    def skip(self, pixels, least=None):
        self.items.append((pixels, pixels if least is None else least, None))
        self.layout()

    def _add(self, key, widget, natural, least):
        self.widgets[key] = widget
        self.items.append((natural, least, [key]))
        self.layout()
        return widget

    def slider(self, key, label, lo, hi, value, integer=False, fmt="{:.2f}"):
        return self._add(key, Slider(Rect(0, 0, 0, 0), label, lo, hi, value, integer, fmt),
                         SLIDER_H, SLIDER_LEAST)

    def toggle(self, key, label, options, index=0):
        return self._add(key, Toggle(Rect(0, 0, 0, 0), label, options, index), TOGGLE_H, TOGGLE_LEAST)

    def graph(self, key, height, least=None):
        return self._add(key, Graph(Rect(0, 0, 0, 0)), height, height if least is None else least)

    def buttons(self, items, per_row=2):
        for start in range(0, len(items), per_row):
            row = items[start:start + per_row]
            for key, label in row:
                self.widgets[key] = Button(Rect(0, 0, 0, 0), label)
            self.items.append((BUTTON_H, BUTTON_LEAST, [key for key, _ in row]))
        self.layout()

    # --- раскладка --------------------------------------------------------------------

    def _heights(self, t):
        """Высоты элементов, промежуток и верхний отступ при сжатии t (1 - естественно)."""
        heights = [least + t * (natural - least) for natural, least, _ in self.items]
        gap = GAP_LEAST + t * (GAP - GAP_LEAST)
        top = PAD_LEAST + t * (self.pad - PAD_LEAST)
        return heights, gap, top

    def _total(self, t):
        heights, gap, top = self._heights(t)
        gaps = sum(1 for _, _, keys in self.items if keys is not None)
        return top + sum(heights) + gap * gaps - (gap if gaps else 0.0)

    def natural_height(self):
        return int(np.ceil(self._total(1.0))) + BOTTOM

    def least_height(self):
        return int(np.ceil(self._total(0.0))) + BOTTOM

    def layout(self, rect=None):
        if rect is not None:
            self.rect = rect
        room = self.rect.height - BOTTOM
        lo, hi = self._total(0.0), self._total(1.0)
        t = 1.0 if hi <= room else 0.0 if lo >= room or hi - lo < 1e-9 else (room - lo) / (hi - lo)
        self.squeeze = t
        heights, gap, top = self._heights(t)
        self.top_pad = int(round(top))
        y = float(self.rect.top) + top
        left, width = self.rect.left + self.pad, self.inner_width
        for (natural, least, keys), h in zip(self.items, heights):
            if keys is None:
                y += h
                continue
            y0, y1 = int(round(y)), int(round(y + h))
            if len(keys) == 1 and not isinstance(self.widgets[keys[0]], Button):
                self.widgets[keys[0]].rect = Rect(left, y0, width, y1 - y0)
            else:
                cell = (width - GAP * (len(keys) - 1)) // len(keys)
                for i, key in enumerate(keys):
                    self.widgets[key].rect = Rect(left + i * (cell + GAP), y0, cell, y1 - y0)
            y += h + gap

    # --- работа -----------------------------------------------------------------------

    def value(self, key):
        return self.widgets[key].value

    def clicked(self, key):
        return self.widgets[key].take()

    def handle(self, event):
        return any([w.handle(event) for w in self.widgets.values() if hasattr(w, "handle")])

    def signature(self, mouse=None):
        """Всё, от чего зависит вид виджетов: пока оно то же, картинка та же."""
        out = []
        for w in self.widgets.values():
            if isinstance(w, Graph):
                continue
            out.append((type(w).__name__, tuple(w.rect), w.label, getattr(w, "_value", None),
                        getattr(w, "index", None), tuple(getattr(w, "options", ())),
                        getattr(w, "pressed", None), _hovered(w.rect, mouse)))
        return tuple(out)

    def draw(self, p, mouse=None):
        """Виджеты рисуются в картинку и дальше копируются, пока ничего в них не сменилось:
        двадцать надписей и тридцать скруглённых прямоугольников - треть кадра, а меняются
        они только от мыши и клавиш."""
        from PySide6.QtGui import QPixmap
        device = p.device()
        dpr = device.devicePixelRatioF() if device is not None else 1.0
        key = (self.signature(mouse), tuple(self.rect), dpr, id(self.font))
        if key != self._cache_key:
            pix = QPixmap(max(1, int(self.rect.width * dpr)), max(1, int(self.rect.height * dpr)))
            pix.setDevicePixelRatio(dpr)
            pix.fill(Qt.GlobalColor.transparent)
            q = render.painter(pix)
            q.translate(-self.rect.left, -self.rect.top)
            for w in self.widgets.values():
                if not isinstance(w, Graph):
                    w.draw(q, self.font, mouse)
            q.end()
            self._cache, self._cache_key = pix, key
        p.drawPixmap(QPointF(self.rect.left, self.rect.top), self._cache)
