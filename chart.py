"""График линии на QPainter (без QtCharts): карточка с заголовком, сеткой, подписями осей,
линией с заливкой и последним значением в углу. По образцу Нейро-змейки; здесь можно
добавить вторую, приглушённую линию (средний результат рядом с лучшим).
"""
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

import theme

HEIGHT = 210


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


class Chart(QWidget):
    def __init__(self, title, color, x_label="", percent=False, unit="", parent=None):
        super().__init__(parent)
        self.title = title
        self.color = QColor(color)
        self.x_label = x_label
        self.percent = percent
        self.unit = unit
        self.points = []
        self.second = []
        self.setMinimumSize(280, 190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(HEIGHT)

    def set_points(self, points, second=None):
        self.points = clean(points)
        self.second = clean(second or [])
        self.setVisible(self.has_data())
        self.update()

    def has_data(self):
        return len(self.points) >= 2

    def last_text(self):
        return fmt(self.points[-1][1], self.percent, self.unit) if self.points else ""

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        card = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.setBrush(QColor(theme.CARD))
        p.drawRoundedRect(card, 12, 12)

        title_font = QFont(self.font())
        title_font.setPointSizeF(10)
        title_font.setBold(True)
        small = QFont(self.font())
        small.setPointSizeF(8)
        head = QRectF(14, 8, self.width() - 28, 22)
        p.setFont(title_font)
        if not self.has_data():
            p.setPen(QColor(theme.TEXT))
            p.drawText(head, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.title)
            p.end()
            return

        # Последнее значение справа в заголовке, заголовок - в оставшееся место.
        last = self.last_text()
        last_w = p.fontMetrics().horizontalAdvance(last) + 12
        p.setPen(self.color)
        p.drawText(head, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, last)
        p.setPen(QColor(theme.TEXT))
        title = p.fontMetrics().elidedText(self.title, Qt.TextElideMode.ElideRight, int(head.width() - last_w))
        p.drawText(head, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)

        series = [self.points] + ([self.second] if len(self.second) >= 2 else [])
        xs = [x for s in series for x, _ in s]
        ys = [y for s in series for _, y in s]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if y1 - y0 < 1e-12:
            y0, y1 = y0 - 0.5, y1 + 0.5
        pad = (y1 - y0) * 0.08
        y0, y1 = y0 - pad, y1 + pad
        if x1 - x0 < 1e-12:
            x1 = x0 + 1

        plot = QRectF(58, 40, self.width() - 74, self.height() - 72)
        p.setFont(small)
        for i in range(5):
            t = i / 4
            yy = plot.bottom() - t * plot.height()
            p.setPen(QPen(QColor(theme.LINE), 1))
            p.drawLine(QPointF(plot.left(), yy), QPointF(plot.right(), yy))
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(4, yy - 8, plot.left() - 10, 16),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       fmt(y0 + t * (y1 - y0), self.percent))
        p.setPen(QColor(theme.MUTED))
        bottom = QRectF(plot.left(), plot.bottom() + 6, plot.width(), 16)
        p.drawText(bottom, Qt.AlignmentFlag.AlignLeft, f"{x0:g}")
        p.drawText(bottom, Qt.AlignmentFlag.AlignRight, f"{x1:g}")
        if self.x_label:
            p.drawText(bottom, Qt.AlignmentFlag.AlignHCenter, self.x_label)

        def at(x, y):
            return QPointF(plot.left() + (x - x0) / (x1 - x0) * plot.width(),
                           plot.bottom() - (y - y0) / (y1 - y0) * plot.height())

        def path(points):
            line = QPainterPath(at(*points[0]))
            for x, y in points[1:]:
                line.lineTo(at(x, y))
            return line

        if len(self.second) >= 2:
            dim = QColor(theme.MUTED)
            pen = QPen(dim, 1.5)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path(self.second))

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
