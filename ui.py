import numpy as np
import pygame

import render

TRACK_H = 5
KNOB_W = 9
SLIDER_H = 32
BUTTON_H = 26
TOGGLE_H = 26
GAP = 3

FILL = (58, 64, 78)
ACTIVE = (110, 190, 230)
KNOB = (206, 214, 230)
HOVER = (74, 82, 99)


def fit_text(font, text, width, tail="…"):
    if font is None or font.size(text)[0] <= width:
        return text
    trimmed = text
    while trimmed and font.size(trimmed + tail)[0] > width:
        trimmed = trimmed[:-1]
    return trimmed + tail


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
    def bar(self):
        return pygame.Rect(self.rect.left, self.rect.bottom - TRACK_H - 4,
                           self.rect.width, TRACK_H)

    def _from_x(self, x):
        bar = self.bar
        t = np.clip((x - bar.left) / max(bar.width, 1), 0.0, 1.0)
        self.value = self.lo + t * (self.hi - self.lo)

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            grab = self.rect.inflate(0, 6)
            if grab.collidepoint(event.pos):
                self.dragging = True
                self._from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._from_x(event.pos[0])
            return True
        return False

    def draw(self, surf, font):
        surf.blit(font.render(self.label, True, render.TEXT_DIM), (self.rect.left, self.rect.top))
        shown = font.render(self.fmt.format(self.value), True, render.TEXT)
        surf.blit(shown, (self.rect.right - shown.get_width(), self.rect.top))

        bar = self.bar
        pygame.draw.rect(surf, FILL, bar, border_radius=2)
        t = (self.value - self.lo) / max(self.hi - self.lo, 1e-9)
        done = pygame.Rect(bar.left, bar.top, int(bar.width * t), bar.height)
        pygame.draw.rect(surf, ACTIVE, done, border_radius=2)
        knob = pygame.Rect(0, 0, KNOB_W, TRACK_H + 8)
        knob.center = (bar.left + int(bar.width * t), bar.centery)
        pygame.draw.rect(surf, KNOB, knob, border_radius=3)


class Button:
    height = BUTTON_H

    def __init__(self, rect, label):
        self.rect = rect
        self.label = label
        self.pressed = False
        self.fired = False

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.pressed = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            hit = self.pressed and self.rect.collidepoint(event.pos)
            self.pressed = False
            if hit:
                self.fired = True
                return True
        return False

    def take(self):
        fired, self.fired = self.fired, False
        return fired

    def draw(self, surf, font):
        hovered = self.rect.collidepoint(pygame.mouse.get_pos())
        colour = ACTIVE if self.pressed else HOVER if hovered else FILL
        pygame.draw.rect(surf, colour, self.rect, border_radius=4)
        text = font.render(self.label, True, render.TEXT)
        surf.blit(text, text.get_rect(center=self.rect.center))


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
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.rect.collidepoint(event.pos):
            self.index = (self.index + 1) % len(self.options)
            return True
        return False

    def draw(self, surf, font):
        hovered = self.rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(surf, HOVER if hovered else FILL, self.rect, border_radius=4)
        surf.blit(font.render(self.label, True, render.TEXT_DIM),
                  (self.rect.left + 8, self.rect.top + 5))
        text = font.render(str(self.value), True, ACTIVE)
        surf.blit(text, (self.rect.right - text.get_width() - 8, self.rect.top + 5))


class Graph:
    def __init__(self, rect):
        self.rect = rect

    def draw(self, surf, history):
        pygame.draw.rect(surf, render.BG, self.rect)
        pygame.draw.rect(surf, render.MIDLINE, self.rect, 1)
        if len(history) < 2:
            return
        best = np.array([s.best for s in history])
        mean = np.array([s.mean for s in history])
        top = max(float(best.max()) * 1.08, 1.0)
        xs = np.linspace(self.rect.left + 1, self.rect.right - 1, len(best))
        for values, colour in ((mean, render.TEXT_DIM), (best, render.LEADER_RING)):
            pts = [(x, self.rect.bottom - 1 - v / top * (self.rect.height - 2))
                   for x, v in zip(xs, values)]
            pygame.draw.lines(surf, colour, False, pts, 2)


class Panel:
    def __init__(self, rect, font, pad=14):
        self.rect = rect
        self.font = font
        self.pad = pad
        self.widgets = {}
        self.cursor = rect.top + pad

    @property
    def inner_width(self):
        return self.rect.width - 2 * self.pad

    def skip(self, pixels):
        self.cursor += pixels

    def place(self, height):
        box = pygame.Rect(self.rect.left + self.pad, self.cursor, self.inner_width, height)
        self.cursor += height + GAP
        return box

    def slider(self, key, label, lo, hi, value, integer=False, fmt="{:.2f}"):
        self.widgets[key] = Slider(self.place(SLIDER_H), label, lo, hi, value, integer, fmt)
        return self.widgets[key]

    def toggle(self, key, label, options, index=0):
        self.widgets[key] = Toggle(self.place(TOGGLE_H), label, options, index)
        return self.widgets[key]

    def buttons(self, items, per_row=2):
        for start in range(0, len(items), per_row):
            row = items[start:start + per_row]
            box = self.place(BUTTON_H)
            width = (box.width - GAP * (len(row) - 1)) // len(row)
            for i, (key, label) in enumerate(row):
                cell = pygame.Rect(box.left + i * (width + GAP), box.top, width, BUTTON_H)
                self.widgets[key] = Button(cell, label)

    def graph(self, key, height):
        self.widgets[key] = Graph(self.place(height))
        return self.widgets[key]

    def value(self, key):
        return self.widgets[key].value

    def clicked(self, key):
        return self.widgets[key].take()

    def handle(self, event):
        return any([w.handle(event) for w in self.widgets.values() if hasattr(w, "handle")])

    def draw(self, surf):
        for w in self.widgets.values():
            if not isinstance(w, Graph):
                w.draw(surf, self.font)
