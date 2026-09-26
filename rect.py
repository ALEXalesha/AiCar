"""Прямоугольник в целых пикселях с правилами pygame.Rect (2.0.0).

До 2.0.0 игра была на pygame, и на его правилах написана вся логика попаданий мышью,
зажима окошка телеметрии в поле и раскладки панели:

- `right` и `bottom` - пиксель ЗА последним: `right = left + width`;
- точка на левом или верхнем краю внутри, на правом или нижнем - снаружи;
- прямоугольники, касающиеся краями, не пересекаются.

У QRect правый край - последний пиксель (`right() = left + width - 1`). Перевести всё на него -
это десятки мест, где легко ошибиться на пиксель, поэтому здесь свой класс с прежними
именами. Qt он не нужен; в Qt он переводится только при рисовании (`qrect`, `qrectf`).
"""


def _px(v):
    return int(round(v))


class Rect:
    __slots__ = ("left", "top", "width", "height")

    def __init__(self, left, top, width, height):
        self.left, self.top = _px(left), _px(top)
        self.width, self.height = _px(width), _px(height)

    # --- края и центр -------------------------------------------------------------------

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def centerx(self):
        return self.left + self.width // 2

    @property
    def centery(self):
        return self.top + self.height // 2

    @property
    def center(self):
        return self.centerx, self.centery

    @center.setter
    def center(self, point):
        self.left = _px(point[0]) - self.width // 2
        self.top = _px(point[1]) - self.height // 2

    @property
    def topleft(self):
        return self.left, self.top

    @topleft.setter
    def topleft(self, point):
        self.left, self.top = _px(point[0]), _px(point[1])

    @property
    def size(self):
        return self.width, self.height

    # --- проверки -----------------------------------------------------------------------

    def collidepoint(self, x, y=None):
        if y is None:
            x, y = x
        return self.left <= x < self.right and self.top <= y < self.bottom

    def contains(self, other):
        return (self.left <= other.left and self.top <= other.top
                and other.right <= self.right and other.bottom <= self.bottom)

    def colliderect(self, other):
        return (self.left < other.right and other.left < self.right
                and self.top < other.bottom and other.top < self.bottom)

    # --- новые прямоугольники -----------------------------------------------------------

    def copy(self):
        return Rect(self.left, self.top, self.width, self.height)

    def move(self, dx, dy):
        return Rect(self.left + dx, self.top + dy, self.width, self.height)

    def inflate(self, dx, dy):
        """Шире на dx и выше на dy, центр на месте (как у pygame: половина в каждую сторону)."""
        return Rect(self.left - dx // 2, self.top - dy // 2, self.width + dx, self.height + dy)

    def clamp_ip(self, area):
        """Сдвинуть внутрь area. Больше area по оси - встаёт по её центру."""
        if self.width >= area.width:
            self.left = area.left + (area.width - self.width) // 2
        else:
            self.left = min(max(self.left, area.left), area.right - self.width)
        if self.height >= area.height:
            self.top = area.top + (area.height - self.height) // 2
        else:
            self.top = min(max(self.top, area.top), area.bottom - self.height)

    # --- Qt -----------------------------------------------------------------------------

    def qrect(self):
        from PySide6.QtCore import QRect
        return QRect(self.left, self.top, self.width, self.height)

    def qrectf(self):
        from PySide6.QtCore import QRectF
        return QRectF(self.left, self.top, self.width, self.height)

    # --- как кортеж ---------------------------------------------------------------------

    def __iter__(self):
        return iter((self.left, self.top, self.width, self.height))

    def __eq__(self, other):
        return isinstance(other, Rect) and tuple(self) == tuple(other)

    __hash__ = None      # изменяемый, как у pygame: в словарь ключом не годится

    def __repr__(self):
        return f"Rect{tuple(self)}"
