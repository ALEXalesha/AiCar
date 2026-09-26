"""Экраны окна кроме самой игры (2.0.0): меню и настройки, плюс общие кирпичики
(подписи, карточки, кнопки), которыми собран и экран статистики.

Вид - по образцу Нейро-змейки и крестиков-ноликов: название, строка о том, что это такое,
крупные кнопки, тёмная тема.
"""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QSlider, QVBoxLayout, QWidget)

import main
import render
import theme

MENU_TEXT_W = 640

ABOUT = ("Трассу и машинку здесь придумывают нейросети. Водители - тоже нейросети: "
         "пятьдесят машинок учатся проезжать трассу эволюцией, поколение за поколением. "
         "Обучение идёт прямо на глазах: смотрите, следите за любой машинкой, меняйте "
         "трассу и правила отбора. Статистика покажет, как от раунда к раунду растёт мастерство.")


def label(text="", name=None, wrap=False):
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


def section(text):
    return label(text.upper(), "section", wrap=True)


def card():
    f = QFrame()
    f.setObjectName("card")
    return f


def button(text, primary=False):
    b = QPushButton(text)
    if primary:
        b.setObjectName("primary")
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def centred(lb, width):
    """Подпись с переносом строк по центру, шириной не больше width.

    Не setFixedWidth с выравниванием по центру: так Qt спрашивает у подписи высоту для всей
    ширины экрана (QWidgetItem::heightForWidth не смотрит на её ширину), получает три строки
    вместо пяти и обрезает текст - это и нашёл тест в окне минимального размера. В ряду с
    пружинами по бокам подпись получает свою ширину, и высота считается для неё."""
    lb.setMaximumWidth(width)
    lb.setMinimumWidth(min(width, 420))
    lb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    row = QHBoxLayout()
    row.addStretch(1)
    row.addWidget(lb, 10)
    row.addStretch(1)
    return row


def header(title, back_text, back):
    """Заголовок экрана с кнопкой возврата справа."""
    row = QHBoxLayout()
    row.addWidget(label(title, "screenTitle"))
    row.addStretch(1)
    b = button(back_text)
    b.clicked.connect(back)
    row.addWidget(b)
    return row, b


# --- меню ------------------------------------------------------------------------------------

class Combo(QComboBox):
    """Выпадающий список со своей стрелкой. Кнопка раскрытия в QSS прозрачная (квадрат
    Fusion торчал за скругление поля), а нарисовать треугольник средствами QSS Qt не
    умеет - только картинкой; картинка в сборке - лишний файл, стрелка проще кодом."""

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.MUTED if not self.isEnabled() else "#9aa3b5"))
        cx, cy = self.width() - 16, self.height() / 2
        up = self.view().isVisible()
        d = -1 if up else 1
        p.drawPolygon([QPointF(cx - 5, cy - 3 * d), QPointF(cx + 5, cy - 3 * d), QPointF(cx, cy + 3 * d)])
        p.end()


class Logo(QWidget):
    """Картинка над названием: кусок трассы, три машинки, у лидера лучи датчиков и кольцо -
    нарисовано тем же кодом, что и сама игра."""

    W, H = 340, 110

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.W, self.H)
        self.cars = [render.icon_car(colour, seed) for colour, seed in
                     (((108, 226, 168), 11), ((236, 104, 96), 5), ((120, 170, 250), 3))]

    def paintEvent(self, event):
        from PySide6.QtGui import QPainterPath
        p = render.painter(self)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.CARD))
        frame = QPainterPath()
        frame.addRoundedRect(QRectF(0.5, 0.5, self.W - 1, self.H - 1), 16, 16)
        p.drawPath(frame)
        p.setClipPath(frame)                      # трасса и лучи - только внутри карточки

        # Трасса: дуга снизу вверх слева направо, лента и стены.
        t = np.linspace(0.0, 1.0, 60)
        centre = np.stack([30 + t * (self.W - 60), self.H * 0.72 - 38 * np.sin(t * np.pi * 0.9)], axis=1)
        tangent = np.gradient(centre, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        half = 17.0
        left, right = centre + normal * half, centre - normal * half
        p.setBrush(render.qc(render.ASPHALT))
        p.drawPolygon(render.qpolygon(np.vstack([left, right[::-1]])))
        for wall, colour, width in ((left, render.WALL, 2), (right, render.WALL, 2), (centre, render.MIDLINE, 1)):
            pen = QPen(render.qc(colour), width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(render.qpolygon(wall))

        spots = (0.2, 0.46, 0.74)                 # последняя - лидер (первая машинка)
        placed = []
        for veh, at in zip(self.cars[::-1], spots):
            i = int(at * (len(t) - 1))
            placed.append((veh, centre[i], float(np.arctan2(tangent[i, 1], tangent[i, 0]))))
        _, origin, angle = placed[-1]
        p.setPen(QPen(render.qc(render.RAY_LINE), 1))
        for ray in np.radians([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]):
            tip = origin + 40 * np.array([np.cos(angle + ray), np.sin(angle + ray)])
            p.drawLine(QPointF(*origin), QPointF(*tip))
        for veh, at, heading in placed:
            render.draw_car_badge(p, veh, at, 1.7, heading)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(render.qc(render.LEADER_RING), 2))
        p.drawEllipse(QPointF(*origin), 24, 24)
        p.end()


class MenuScreen(QWidget):
    play = Signal()
    stats = Signal()
    settings = Signal()
    quit = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("screen")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 28, 32, 22)
        outer.addStretch(3)

        self.logo = Logo()
        outer.addWidget(self.logo, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addSpacing(14)
        title = label("AI Car Racing", "title")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(title)
        self.subtitle = label(ABOUT, "subtitle", wrap=True)
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        outer.addLayout(centred(self.subtitle, MENU_TEXT_W))
        outer.addSpacing(24)

        column = QVBoxLayout()
        column.setSpacing(10)
        self.play_button = button("Играть", primary=True)
        self.stats_button = button("Статистика")
        self.settings_button = button("Настройки")
        self.quit_button = button("Выход")
        for b, sig in ((self.play_button, self.play), (self.stats_button, self.stats),
                       (self.settings_button, self.settings), (self.quit_button, self.quit)):
            b.setFixedWidth(300)
            b.clicked.connect(sig)
            column.addWidget(b, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addLayout(column)
        outer.addStretch(4)

        self.info = label("", "muted", wrap=True)
        self.info.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        outer.addLayout(centred(self.info, MENU_TEXT_W + 120))

    def set_info(self, game):
        """Строка внизу: с какими настройками начнётся игра и сколько уже сыграно."""
        ui = game.ui
        rounds = game.totals.get("rounds", 0)
        state = "идёт обучение" if game.started else "игра ещё не начата"
        self.info.setText(f"уровень {ui.value('level')}   ·   {ui.value('generator')}   ·   "
                          f"популяция {ui.value('pop_size')}   ·   раундов сыграно {rounds}   ·   {state}")
        self.play_button.setText("Продолжить" if game.started else "Играть")


# --- настройки -------------------------------------------------------------------------------

class SettingsScreen(QWidget):
    """То же, что ползунки и переключатели панели, только крупно и с пояснениями; меняет
    панель игры сразу, а окно запоминает всё в settings.json (prefs.py)."""

    back = Signal()
    changed = Signal(str)

    def __init__(self, game, parent=None):
        super().__init__(parent)
        self.setObjectName("screen")
        self.game = game
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 22, 28, 22)
        row, self.back_button = header("Настройки", "В меню", self.back.emit)
        outer.addLayout(row)
        outer.addSpacing(10)

        form = card()
        form.setMaximumWidth(820)
        grid = QGridLayout(form)
        grid.setContentsMargins(24, 20, 24, 20)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(14)
        grid.setColumnStretch(1, 1)
        ui = game.ui.widgets
        self.sliders, self.values, self.combos = {}, {}, {}

        rows = (
            ("volume", "Громкость", None, "мотор ведущей машинки, удар и финиш; 0 - тишина"),
            ("level", "Уровень", ui["level"].options,
             "ширина трассы и крутизна поворотов разом; ползунки ширины и сложности на панели "
             "уточняют его"),
            ("generator", "Генератор трассы", ui["generator"].options,
             "обученная модель строит трассу в десятки раз быстрее эволюции CPPN"),
            ("pop_size", "Популяция", None, "сколько машинок едет в каждом поколении"),
            ("generations", "Поколений", None,
             "предел обучения на раунд: если никто не доехал, раунд кончается на нём"),
            ("speed", "Скорость показа", ui["speed"].options,
             "то же, что клавиши 1-4 в игре; «без отрисовки» считает поколение за кадр"),
            ("replay", "Повтор заезда", ui["replay"].options, "что показать, когда обучение закончилось"),
        )
        for n, (key, name, options, hint) in enumerate(rows):
            if options is None:
                widget = QSlider(Qt.Orientation.Horizontal)
                lo, hi = ui[key].lo, ui[key].hi
                scale = 100 if key == "volume" else 1
                widget.setRange(int(round(lo * scale)), int(round(hi * scale)))
                widget.valueChanged.connect(lambda v, k=key, s=scale: self._slide(k, v / s))
                self.sliders[key] = (widget, scale)
                value = label("", "value")
                value.setMinimumWidth(70)
                self.values[key] = value
            else:
                widget = Combo()
                widget.addItems([str(o) for o in options])
                widget.currentIndexChanged.connect(lambda i, k=key: self._choose(k, i))
                self.combos[key] = widget
                value = None
            self._row(grid, n, name, widget, value, hint)
        outer.addWidget(form)
        outer.addStretch(1)

    @staticmethod
    def _row(grid, row, name, widget, value, hint):
        grid.addWidget(label(name, "value"), row, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(4)
        line = QHBoxLayout()
        line.addWidget(widget, 1 if isinstance(widget, QSlider) else 0)
        if value is not None:
            line.addWidget(value)
        else:
            widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            line.addStretch(1)
        col.addLayout(line)
        if hint:
            col.addWidget(label(hint, "muted", wrap=True))
        grid.addLayout(col, row, 1)

    def load(self):
        """Показать то, что сейчас на панели игры, ничего не меняя."""
        ui = self.game.ui.widgets
        for key, (widget, scale) in self.sliders.items():
            widget.blockSignals(True)
            widget.setValue(int(round(ui[key].value * scale)))
            widget.blockSignals(False)
        for key, widget in self.combos.items():
            widget.blockSignals(True)
            widget.setCurrentIndex(ui[key].index)
            widget.blockSignals(False)
        self._labels()

    def _labels(self):
        ui = self.game.ui.widgets
        for key, value in self.values.items():
            value.setText(f"{ui[key].value * 100:.0f}%" if key == "volume" else ui[key].shown)

    def _slide(self, key, value):
        self.game.ui.widgets[key].value = value
        self._labels()
        self.changed.emit(key)

    def _choose(self, key, index):
        self.game.ui.widgets[key].index = index
        if key == "level":
            self.game.apply_level()           # ширина и сложность уровня - сразу на панели
        self._labels()
        self.changed.emit(key)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.back.emit()
            return
        super().keyPressEvent(event)
