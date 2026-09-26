"""Окно игры на Qt (2.0.0; до того - цикл pygame в main.Game.run).

`MainWindow` - стопка экранов: меню (с него окно открывается), игра, статистика, настройки;
место окна и настройки между запусками, звук по таймеру. `GameView` - экран игры: крутит
`main.Game` таймером, рисует её кадр в paintEvent, отдаёт ей мышь и клавиши, показывает
всплывающие сообщения.

Цикл - QTimer с точным ходом (16 мс) и счётом кадров по часам: логика идёт ровно
60 кадров в секунду, как у pygame с clock.tick(60), а не 62.5 от целых миллисекунд.
Отставание больше двух кадров не догоняется: после паузы в пару секунд (строилась трасса)
игра не пытается наверстать их разом.
"""
from collections import deque

from PySide6.QtCore import QElapsedTimer, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow, QStackedWidget, QWidget

import config as cfg
import main
import prefs
import render
import theme
import window_state
from screens import MenuScreen, SettingsScreen
from stats_screen import StatsScreen

FRAME_MS = 1000.0 / cfg.FPS
MAX_CATCH_UP = 2
AUDIO_MS = 15
TOAST_TOP, TOAST_SIDE, TOAST_MAX_W = 16, 24, 560
BUTTONS = {Qt.MouseButton.LeftButton: 1, Qt.MouseButton.MiddleButton: 2, Qt.MouseButton.RightButton: 3}


def make_icon():
    vehicle = render.icon_car()
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(QPixmap.fromImage(render.icon_image(vehicle, size)))
    return icon


def key_code(event):
    """Код клавиши для Game.key. Буквы и цифры - по физической клавише (код Windows равен
    Qt.Key для латиницы), чтобы R, S, L работали и на русской раскладке."""
    vk = event.nativeVirtualKey()
    if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
        return vk
    return int(event.key())


class GameView(QWidget):
    stats_wanted = Signal()
    menu_wanted = Signal()

    def __init__(self, game, parent=None):
        super().__init__(parent)
        self.game = game
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMinimumSize(main.MIN_W, main.MIN_H)
        game.on_splash = self.show_splash

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(int(FRAME_MS))
        self.timer.timeout.connect(self.tick)
        self.clock = QElapsedTimer()
        self.frames = 0
        self.paint_ms = deque(maxlen=240)

        # Сообщение (say) - поверх поля, целиком и с переносом строк. В pygame оно
        # подменяло строку итогов в панели и обрезалось многоточием.
        self.toast = QLabel(self)
        self.toast.setObjectName("toast")
        self.toast.setWordWrap(True)
        self.toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.toast.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.toast.hide()

    # --- цикл ---------------------------------------------------------------------------

    def start(self):
        self.clock.start()
        self.frames = 0
        self.timer.start()

    def stop(self):
        self.timer.stop()
        self.game.audio.stop()

    def is_running(self):
        return self.timer.isActive()

    def tick(self):
        due = int(self.clock.elapsed() / FRAME_MS) - self.frames
        if due <= 0:
            return
        if due > MAX_CATCH_UP:
            self.frames += due - MAX_CATCH_UP
            due = MAX_CATCH_UP
        if not self.game.speed:          # «без отрисовки»: кадр - целое поколение, одного хватит
            due = 1
        for _ in range(due):
            self.game.frame()
            self.frames += 1
            if self.game.want_stats or self.game.want_menu:
                break
        self.after_input()

    def after_input(self):
        """После кадра, щелчка или клавиши: сообщение, перерисовка, переход на другой экран."""
        self.sync_toast()
        self.update()
        if self.game.want_stats:
            self.game.want_stats = False
            self.stats_wanted.emit()
        if self.game.want_menu:
            self.game.want_menu = False
            self.menu_wanted.emit()

    def show_splash(self):
        """Трасса строится пару секунд, и всё это время цикл стоит: заставку рисуем сразу."""
        if self.isVisible():
            self.repaint()

    # --- сообщение ----------------------------------------------------------------------

    def sync_toast(self):
        g = self.game
        if g.message_left <= 0 or not g.message:
            self.toast.hide()
            return
        if self.toast.text() != g.message:
            self.toast.setText(g.message)
        self.place_toast()
        self.toast.show()

    def place_toast(self):
        view = self.game.view
        pad = self.toast.contentsMargins().left() + self.toast.contentsMargins().right() + 32
        natural = self.toast.fontMetrics().horizontalAdvance(self.toast.text()) + pad
        width = int(max(120, min(natural, TOAST_MAX_W, view.width - 2 * TOAST_SIDE)))
        self.toast.setFixedWidth(width)
        self.toast.resize(width, self.toast.heightForWidth(width))
        self.toast.move(view.left + (view.width - width) // 2, view.top + TOAST_TOP)

    # --- Qt -----------------------------------------------------------------------------

    def paintEvent(self, event):
        started = QElapsedTimer()
        started.start()
        p = render.painter(self)
        self.game.paint(p)
        p.end()
        self.paint_ms.append(started.nsecsElapsed() / 1e6)

    def resizeEvent(self, event):
        self.game.resize(self.width(), self.height())
        if self.toast.isVisible():
            self.place_toast()
        super().resizeEvent(event)

    @staticmethod
    def _pos(event):
        at = event.position()
        return int(at.x()), int(at.y())

    def mousePressEvent(self, event):
        self.game.mouse_down(self._pos(event), BUTTONS.get(event.button(), 0))
        self.after_input()

    def mouseDoubleClickEvent(self, event):
        # Второй щелчок двойного Qt присылает сюда, а не в mousePressEvent: без этого
        # быстрые нажатия на кнопку панели терялись бы через одно.
        self.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.game.mouse_move(self._pos(event))
        self.update()

    def mouseReleaseEvent(self, event):
        self.game.mouse_up(self._pos(event), BUTTONS.get(event.button(), 0))
        self.after_input()

    def leaveEvent(self, event):
        self.game.mouse = None
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        if event.isAutoRepeat():         # зажатый пробел не должен мигать паузой
            return
        if not self.game.started:
            return
        self.game.key(key_code(event))
        self.after_input()


class MainWindow(QMainWindow):
    def __init__(self, game, window_path=None, settings_path=None):
        super().__init__()
        self.game = game
        self.settings_path = settings_path
        if settings_path is not None:
            prefs.apply(game, prefs.load(settings_path))
        self.setWindowTitle("AI Car Racing")
        self.setWindowIcon(make_icon())
        self.setStyleSheet(theme.QSS)

        self.stack = QStackedWidget()
        self.menu = MenuScreen()
        self.view = GameView(game)
        self.stats_screen = StatsScreen()
        self.settings_screen = SettingsScreen(game)
        for page in (self.menu, self.view, self.stats_screen, self.settings_screen):
            self.stack.addWidget(page)
        self.setCentralWidget(self.stack)
        self.setMinimumSize(main.MIN_W, main.MIN_H)
        self.resize(cfg.WINDOW_W, cfg.WINDOW_H)
        self.stats_from = self.menu          # куда вернуться со статистики

        self.menu.play.connect(self.play)
        self.menu.stats.connect(lambda: self.show_stats(self.menu))
        self.menu.settings.connect(self.show_settings)
        self.menu.quit.connect(self.close)
        self.view.stats_wanted.connect(lambda: self.show_stats(self.view))
        self.view.menu_wanted.connect(self.show_menu)
        self.stats_screen.back.connect(self.back_from_stats)
        self.settings_screen.back.connect(self.show_menu)

        # Звук дописывается в устройство своим таймером: и когда игра стоит (пауза, меню,
        # статистика), иначе затухание мотора не доиграло бы до конца.
        self.audio_timer = QTimer(self)
        self.audio_timer.setInterval(AUDIO_MS)
        self.audio_timer.timeout.connect(game.audio.pump)
        self.audio_timer.start()

        self.remember = None
        if window_path is not None:
            self.remember = window_state.Remember(
                self, window_path, {"width": cfg.WINDOW_W, "height": cfg.WINDOW_H,
                                    "minWidth": main.MIN_W, "minHeight": main.MIN_H})
        self.show_menu()

    def show_window(self):
        if self.remember is not None:
            self.remember.show()
        else:
            self.show()

    def current(self):
        return self.stack.currentWidget()

    # --- экраны -------------------------------------------------------------------------

    def show_menu(self):
        self.view.stop()
        self.save_settings()
        self.menu.set_info(self.game)
        self.stack.setCurrentWidget(self.menu)
        self.menu.play_button.setFocus()

    def play(self):
        """«Играть»: первый раз - строится трасса (заставка видна), потом - та же игра дальше."""
        self.stack.setCurrentWidget(self.view)
        self.view.setFocus()
        if not self.game.started:
            self.game.new_round(new_car=True)
        self.view.start()

    def show_stats(self, came_from=None):
        self.view.stop()
        self.stats_from = came_from or self.menu
        g = self.game
        self.stats_screen.back_button.setText("К игре" if self.stats_from is self.view else "В меню")
        self.stats_screen.refresh(g.totals, g.history if g.started else None,
                                  live=g.started and g.state == main.TRAINING)
        self.stack.setCurrentWidget(self.stats_screen)
        self.stats_screen.setFocus()

    def back_from_stats(self):
        if self.stats_from is self.view:
            self.play()
        else:
            self.show_menu()

    def show_settings(self):
        self.view.stop()
        self.settings_screen.load()
        self.stack.setCurrentWidget(self.settings_screen)
        self.settings_screen.setFocus()

    def save_settings(self):
        if self.settings_path is not None:
            prefs.save(self.settings_path, prefs.collect(self.game))

    # --- окно ---------------------------------------------------------------------------

    def moveEvent(self, event):
        if self.remember is not None:
            self.remember.track()
        super().moveEvent(event)

    def resizeEvent(self, event):
        if self.remember is not None:
            self.remember.track()
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.view.stop()
        self.audio_timer.stop()
        self.game.audio.close()
        self.save_settings()
        if self.remember is not None:
            self.remember.save()
        super().closeEvent(event)
