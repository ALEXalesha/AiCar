"""График линии на QPainter (без QtCharts), по образцу Нейро-змейки: карточка с заголовком,
сеткой, делениями и подписями обеих осей, линией с заливкой и последним значением в углу.

Здесь можно добавить вторую, приглушённую линию (средний результат поколения рядом с лучшим) -
тогда в углу оба последних значения с подписями, они же легенда. Пока точек меньше двух,
карточка не прячется, а говорит словами, откуда возьмётся график (`empty_text`).
"""
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

import theme

HEIGHT = 210
LEFT, TOP, RIGHT = 58, 42, 16
BOTTOM = 46                      # подписи делений и название оси под графиком


def clean(points):
    """Только конечные пары чисел, по возрастанию x: мусор из файла не рисуется."""
    out = []
    for item in points if isinstance(points, (list, tuple)) else []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        x, y = item
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (x, y)):
            out.append((float(x), float(y)))
    return sorted(out)


def fmt(v, percent=False, unit=""):
    if percent:
        return f"{v * 100:.0f}%" if abs(v) >= 0.1 else f"{v * 100:.1f}%"
    if v != 0 and abs(v) >= 100000:
        text = f"{v:.3g}"
    elif abs(v - round(v)) < 1e-9 or abs(v) >= 100:
        text = f"{v:.0f}"
    else:
        text = f"{v:.1f}"
    return text + unit


def y_scale(lo, hi, most=5):
    """Круглая шкала по y: (низ, верх, деления) шагом 1, 2, 2.5, 5 на степень десяти.
    Неотрицательные данные, у которых разброс больше их высоты, начинаются с нуля."""
    if lo >= 0 and lo <= 0.3 * hi:
        lo = 0.0
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    raw = (hi - lo) / (most - 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw - 1e-12)
    start = math.floor(lo / step + 1e-9) * step
    end = math.ceil(hi / step - 1e-9) * step
    if end - start < 1e-12:
        end = start + step
    count = int(round((end - start) / step))
    return start, end, [start + i * step for i in range(count + 1)], step


def _digits(step):
    """Сколько знаков после точки нужно, чтобы деления шагом step читались точно (2.5 - один)."""
    digits = 0
    while digits < 6 and abs(step * 10 ** digits - round(step * 10 ** digits)) > 1e-6:
        digits += 1
    return digits


def tick_text(v, step, percent=False):
    if percent:
        return f"{v * 100:.{_digits(step * 100)}f}%"
    return f"{v:.{_digits(step)}f}"


def x_ticks(x0, x1, most=6):
    """Целые деления оси x шагом 1, 2, 5, 10, 20, 50...: не больше most штук."""
    span = max(x1 - x0, 1e-9)
    k = 0
    while True:
        step = (1, 2, 5)[k % 3] * 10 ** (k // 3)
        if span / step <= most - 1:
            break
        k += 1
    first = math.ceil(x0 / step) * step
    ticks = []
    t = first
    while t <= x1 + 1e-9:
        ticks.append(t)
        t += step
    return ticks or [x0]


class Chart(QWidget):
    def __init__(self, title, color, x_label="", percent=False, unit="", names=None,
                 empty_text="", height=HEIGHT, y_label="", parent=None):
        super().__init__(parent)
        self.title = title
        self.color = QColor(color)
        self.x_label = x_label
        self.y_label = y_label
        self.percent = percent
        self.unit = unit
        self.names = names                   # подписи двух линий: ("лучший", "средний")
        self.empty_text = empty_text
        self.points = []
        self.second = []
        self.setMinimumSize(280, 190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(height)

    def set_points(self, points, second=None):
        self.points = clean(points)
        self.second = clean(second or [])
        # Без данных карточка видна, только если ей есть что сказать словами.
        self.setVisible(self.has_data() or bool(self.empty_text))
        self.update()

    def has_data(self):
        return len(self.points) >= 2

    def last_text(self):
        return fmt(self.points[-1][1], self.percent, self.unit) if self.points else ""

    def corner(self):
        """Что написать в правом верхнем углу: последнее значение (или оба, с подписями)."""
        if not self.points:
            return []
        if self.names and len(self.second) >= 2:
            return [(f"{self.names[0]} ", theme.MUTED), (self.last_text(), self.color.name()),
                    (f"   {self.names[1]} ", theme.MUTED),
                    (fmt(self.second[-1][1], self.percent, self.unit), theme.TEXT)]
        return [(self.last_text(), self.color.name())]

    def fonts(self):
        title_font = QFont(self.font())
        title_font.setPointSizeF(10)
        title_font.setBold(True)
        small = QFont(self.font())
        small.setPointSizeF(8)
        return title_font, small

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        card = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.setBrush(QColor(theme.CARD))
        p.drawRoundedRect(card, 12, 12)

        title_font, small = self.fonts()
        head = QRectF(14, 8, self.width() - 28, 22)
        p.setFont(title_font)
        # Угол: последние значения, справа налево; заголовок - в оставшееся место.
        right = head.right()
        for text, colour in reversed(self.corner()):
            w = p.fontMetrics().horizontalAdvance(text)
            p.setPen(QColor(colour))
            p.drawText(QRectF(right - w, head.top(), w + 1, head.height()),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
            right -= w
        p.setPen(QColor(theme.TEXT))
        title = p.fontMetrics().elidedText(self.title, Qt.TextElideMode.ElideRight,
                                           int(max(40, right - head.left() - 12)))
        p.drawText(head, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)

        if not self.has_data():
            p.setFont(QFont(self.font()))
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(24, 40, self.width() - 48, self.height() - 56),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.empty_text)
            p.end()
            return

        series = [self.points] + ([self.second] if len(self.second) >= 2 else [])
        xs = [x for s in series for x, _ in s]
        ys = [y for s in series for _, y in s]
        x0, x1 = min(xs), max(xs)
        y0, y1, y_ticks, step = y_scale(min(ys), max(ys))
        if x1 - x0 < 1e-12:
            x1 = x0 + 1

        top = TOP + (14 if self.y_label else 0)
        plot = QRectF(LEFT, top, self.width() - LEFT - RIGHT, self.height() - top - BOTTOM)
        p.setFont(small)
        if self.y_label:            # название оси y - над её подписями, слева
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(8, top - 26, plot.width(), 14),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.y_label)
        for value in y_ticks:
            yy = plot.bottom() - (value - y0) / (y1 - y0) * plot.height()
            p.setPen(QPen(QColor(theme.LINE), 1))
            p.drawLine(QPointF(plot.left(), yy), QPointF(plot.right(), yy))
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(4, yy - 8, plot.left() - 10, 16),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       tick_text(value, step, self.percent))

        def at(x, y):
            return QPointF(plot.left() + (x - x0) / (x1 - x0) * plot.width(),
                           plot.bottom() - (y - y0) / (y1 - y0) * plot.height())

        # Ось x: деления с подписями и название оси под ними.
        axis = QPen(QColor(theme.MUTED), 1)
        for t in x_ticks(x0, x1, most=int(max(6, min(12, plot.width() // 70)))):
            x = at(t, y0).x()
            p.setPen(axis)
            p.drawLine(QPointF(x, plot.bottom()), QPointF(x, plot.bottom() + 4))
            p.drawText(QRectF(x - 30, plot.bottom() + 5, 60, 14), Qt.AlignmentFlag.AlignHCenter, f"{t:g}")
        if self.x_label:
            p.drawText(QRectF(plot.left(), plot.bottom() + 22, plot.width(), 16),
                       Qt.AlignmentFlag.AlignHCenter, self.x_label)

        def path(points):
            line = QPainterPath(at(*points[0]))
            for x, y in points[1:]:
                line.lineTo(at(x, y))
            return line

        if len(self.second) >= 2:
            pen = QPen(QColor(theme.MUTED), 1.5)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path(self.second))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.MUTED))
            p.drawEllipse(at(*self.second[-1]), 2.5, 2.5)

        line = path(self.points)
        fill = QPainterPath(line)
        fill.lineTo(at(self.points[-1][0], y0))
        fill.lineTo(at(self.points[0][0], y0))
        fill.closeSubpath()
        grad = QLinearGradient(plot.topLeft(), plot.bottomLeft())
        top = QColor(self.color)
        top.setAlpha(90)
        bottom_c = QColor(self.color)
        bottom_c.setAlpha(0)
        grad.setColorAt(0, top)
        grad.setColorAt(1, bottom_c)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawPath(fill)
        pen = QPen(self.color, 2)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(line)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.color)
        p.drawEllipse(at(*self.points[-1]), 3.5, 3.5)
        p.end()
