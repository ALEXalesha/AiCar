"""Вид выпадающего списка: кнопка раскрытия не торчит за скруглённый угол.

Алексей 26.09.2026 на 2.0.0: «меню закруглённое, а кнопка раскрытия квадратная и из-за
этого торчит немного и выглядит некрасиво». Кнопка ::drop-down без своего стиля рисуется
квадратом стиля Fusion поверх скругления рамки.
"""
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

import theme
from screens import Combo

BG = QColor("#101318")


def render(qapp, popup=False):
    host = QWidget()
    host.setStyleSheet(theme.QSS)
    host.setAutoFillBackground(True)
    pal = host.palette()
    pal.setColor(host.backgroundRole(), BG)
    host.setPalette(pal)
    box = Combo(host)
    box.addItems(["обученная модель", "случайная модель"])
    box.move(10, 10)
    box.adjustSize()
    host.resize(box.width() + 20, box.height() + 20)
    host.show()
    qapp.processEvents()
    image = host.grab().toImage()
    return image, box.geometry()


def near(a, b, tol=12):
    return abs(a.red() - b.red()) + abs(a.green() - b.green()) + abs(a.blue() - b.blue()) <= tol


def test_drop_down_button_stays_inside_rounded_corners(qapp):
    image, g = render(qapp)
    # За дугой скругления (радиус 8 px) справа - фон страницы: квадрат кнопки туда не лезет.
    r = 8
    for cx, cy, sx, sy in ((g.right() - r + 1, g.top() + r, 1, -1), (g.right() - r + 1, g.bottom() - r + 1, 1, 1)):
        for dx in range(r):
            for dy in range(r):
                x, y = cx + sx * dx, cy + sy * dy
                if (dx + 0.5) ** 2 + (dy + 0.5) ** 2 > (r + 1.5) ** 2:
                    c = image.pixelColor(x, y)
                    assert near(c, BG), f"угол ({x - g.x()}; {y - g.y()}) выпадающего списка закрашен {c.name()}: кнопка торчит"


def test_drop_down_draws_no_shadow_of_its_own(qapp):
    image, g = render(qapp)
    # У квадрата кнопки Fusion своя тень темнее страницы; в поле списка её быть не должно.
    for x in range(g.x(), g.right() + 1):
        for y in range(g.y(), g.bottom() + 1):
            c = image.pixelColor(x, y)
            assert c.lightness() >= BG.lightness() - 2, f"({x - g.x()}; {y - g.y()}) темнее страницы: {c.name()}"


def test_drop_down_has_no_separate_square_fill(qapp):
    image, g = render(qapp)
    # Под стрелкой - тот же цвет поля, что и слева от неё: отдельного квадрата нет.
    y = g.center().y() - 6
    left = image.pixelColor(g.x() + g.width() // 2, y)
    right = image.pixelColor(g.right() - 4, y)
    assert near(left, right), f"поле {left.name()}, под кнопкой {right.name()}"


def test_drop_down_arrow_is_visible(qapp):
    image, g = render(qapp)
    # Стрелка - светлые точки в правых 24 px поля: без неё не понять, что это список.
    bright = [(x, y) for x in range(g.right() - 24, g.right())
              for y in range(g.y() + 3, g.bottom() - 3) if image.pixelColor(x, y).lightness() > 120]
    assert len(bright) >= 12, f"стрелки раскрытия нет: светлых точек {len(bright)}"
    ys = [y for _, y in bright]
    mid = (min(ys) + max(ys)) / 2
    assert abs(mid - g.center().y()) <= 2, f"стрелка не по середине высоты: {mid} против {g.center().y()}"
    # Треугольник остриём вниз: сверху строка светлых точек шире, чем снизу (прямоугольник не пройдёт).
    width = lambda yy: sum(1 for _, y in bright if y == yy)
    assert width(min(ys)) >= width(max(ys)) + 4, f"стрелка не треугольник: сверху {width(min(ys))}, снизу {width(max(ys))}"
