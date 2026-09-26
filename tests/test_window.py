"""Окно игры на Qt (window.py, 2.0.0): цикл, мышь, клавиши, экраны, размеры, подписи."""
import functools
import json

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

import config as cfg
import main
import render
import window
from qt_app import qapp

QWIDGETSIZE_MAX = 16777215


@functools.lru_cache(maxsize=1)
def shared_game():
    qapp()
    return main.Game(seed=0)


@pytest.fixture
def win(qapp):
    g = shared_game()
    g.watched = main.WATCH_LEADER
    g.paused = False
    g.want_stats = g.want_menu = False
    g.hud_grab = None
    g.message_left = 0
    w = window.MainWindow(g)
    w.show()
    w.play()
    qapp.processEvents()
    yield w
    w.close()
    g.resize(cfg.WINDOW_W, cfg.WINDOW_H)


def view_point(w, x, y):
    return QPoint(int(x), int(y))


def test_the_window_paints_the_game(win):
    image = win.view.grab().toImage()
    assert not image.isNull()
    g = win.game
    panel = image.pixelColor(g.panel_rect.left + 3, g.height // 2)
    assert (panel.red(), panel.green(), panel.blue()) == render.PANEL_BG


def test_the_timer_drives_the_game(win):
    g = win.game
    g.ui.widgets["speed"].index = 0
    steps = g.race.steps
    QTest.qWait(200)
    assert win.view.is_running()
    assert g.race.steps > steps or g.state != main.TRAINING


def test_the_timer_keeps_sixty_frames_a_second(win, monkeypatch):
    """Счёт кадров по часам: 16-мс таймер дал бы 62.5 кадра, а надо 60."""
    v = win.view
    v.timer.stop()
    elapsed = {"ms": 0.0}
    monkeypatch.setattr(v.clock, "elapsed", lambda: elapsed["ms"])
    frames = []
    monkeypatch.setattr(v.game, "frame", lambda: frames.append(1))
    v.frames = 0
    for tick in range(1, 626):
        elapsed["ms"] = tick * 16.0
        v.tick()
    assert len(frames) == int(625 * 16.0 / window.FRAME_MS)       # 10 секунд - 600 кадров


def test_a_long_stall_is_not_caught_up_at_once(win, monkeypatch):
    v = win.view
    v.timer.stop()
    monkeypatch.setattr(v.clock, "elapsed", lambda: 2000.0)       # две секунды строилась трасса
    frames = []
    monkeypatch.setattr(v.game, "frame", lambda: frames.append(1))
    v.frames = 0
    v.tick()
    assert len(frames) == window.MAX_CATCH_UP


def test_clicking_a_car_with_the_mouse_watches_it(win):
    g = win.game
    step = len(g.track.center) // g.race.n
    g.race.pos[:] = g.track.center[:g.race.n * step:step]
    shown = g.telemetry()["index"]
    for i in range(g.race.n):
        x, y = g.camera.to_screen(g.race.pos[i])[0]
        if i != shown and g.race.nearest_to(g.camera.to_world((int(x), int(y)))) == i \
                and not g.hud.collidepoint(int(x), int(y)):
            break
    QTest.mouseClick(win.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     view_point(win, x, y))
    assert g.watched == i


def test_keys_reach_the_game(win):
    g = win.game
    QTest.keyClick(win.view, Qt.Key.Key_N)
    assert g.watched == main.WATCH_NONE
    QTest.keyClick(win.view, Qt.Key.Key_L)
    assert g.watched == main.WATCH_LEADER
    QTest.keyClick(win.view, Qt.Key.Key_3)
    assert g.ui.widgets["speed"].index == 2
    QTest.keyClick(win.view, Qt.Key.Key_Space)
    assert g.paused
    QTest.keyClick(win.view, Qt.Key.Key_Space)
    assert not g.paused


def test_letters_work_on_the_russian_layout():
    """На русской раскладке R даёт «К», но код физической клавиши у Windows тот же."""
    event = QKeyEvent(QEvent.Type.KeyPress, 0x041A, Qt.KeyboardModifier.NoModifier, 19, 0x52, 0, "к")
    assert window.key_code(event) == main.KEY_ROUND
    plain = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    assert window.key_code(plain) == main.KEY_MENU


def test_a_held_key_does_not_repeat(win):
    g = win.game
    repeat = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, "", True)
    QApplication.sendEvent(win.view, repeat)
    assert not g.paused


def test_escape_in_the_game_goes_to_the_menu(win):
    QTest.keyClick(win.view, Qt.Key.Key_Escape)
    assert win.current() is win.menu and win.isVisible()
    assert not win.view.is_running(), "в меню игра стоит"
    assert win.menu.play_button.text() == "Продолжить"
    QTest.mouseClick(win.menu.play_button, Qt.MouseButton.LeftButton)
    assert win.current() is win.view and win.view.is_running()


def test_the_menu_button_on_the_panel_goes_to_the_menu(win):
    box = win.game.ui.widgets["menu"].rect
    QTest.mouseClick(win.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     view_point(win, *box.center))
    QTest.qWait(60)
    assert win.current() is win.menu


def test_a_double_click_on_a_button_counts_twice(win):
    g = win.game
    box = g.ui.widgets["pause"].rect
    QTest.mouseDClick(win.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      view_point(win, *box.center))
    g.apply_buttons()
    QTest.mouseClick(win.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     view_point(win, *box.center))
    g.apply_buttons()
    assert g.paused        # три щелчка: пауза, снята, пауза
    g.paused = False


def test_hovering_a_button_is_seen_by_the_panel(win):
    g = win.game
    box = g.ui.widgets["save"].rect
    QTest.mouseMove(win.view, view_point(win, *box.center))
    assert g.mouse == box.center


def test_the_stats_button_opens_the_stats_screen_and_back(win):
    g = win.game
    box = g.ui.widgets["stats"].rect
    QTest.mouseClick(win.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     view_point(win, *box.center))
    QTest.qWait(60)
    assert win.stack.currentWidget() is win.stats_screen
    assert not win.view.is_running(), "игра должна стоять, пока открыта статистика"
    assert win.stats_screen.back_button.text() == "К игре"
    QTest.mouseClick(win.stats_screen.back_button, Qt.MouseButton.LeftButton)
    assert win.stack.currentWidget() is win.view and win.view.is_running()
    win.show_stats(win.view)
    QTest.keyClick(win.stats_screen, Qt.Key.Key_Escape)
    assert win.stack.currentWidget() is win.view


def test_the_window_fits_a_1024x768_screen(win):
    # экран 1024x768 минус панель задач и заголовок окна
    assert win.minimumWidth() <= 1000 and win.minimumHeight() <= 690
    assert win.minimumSizeHint().width() <= 1000 and win.minimumSizeHint().height() <= 690
    win.show_stats()
    assert win.minimumSizeHint().width() <= 1000 and win.minimumSizeHint().height() <= 690


def test_resizing_the_window_resizes_the_game(win, qapp):
    g = win.game
    win.resize(1500, 900)
    qapp.processEvents()
    assert (g.width, g.height) == (win.view.width(), win.view.height())
    assert g.panel_rect.right == win.view.width()
    image = win.view.grab().toImage()
    assert (image.width(), image.height()) == (win.view.width(), win.view.height())


def test_at_the_smallest_size_every_panel_widget_is_visible(win, qapp):
    g = win.game
    win.resize(win.minimumSize())
    qapp.processEvents()
    assert (win.view.width(), win.view.height()) == (main.MIN_W, main.MIN_H)
    for key, w in g.ui.widgets.items():
        assert g.panel_rect.contains(w.rect), key


def test_the_message_is_shown_whole_above_the_field(win):
    g = win.game
    g.paused = True
    g.say("мозг загружен, обучение начато заново")
    win.view.after_input()
    toast = win.view.toast
    assert toast.isVisible() and toast.text() == g.message
    assert toast.heightForWidth(toast.width()) <= toast.height()
    assert g.view.contains(render.Rect(toast.x(), toast.y(), toast.width(), toast.height()))
    for _ in range(main.MESSAGE_FRAMES):
        g.frame()
    win.view.after_input()
    assert not toast.isVisible()


def test_the_splash_is_painted_before_a_new_track_is_built(win, monkeypatch):
    painted = []
    monkeypatch.setattr(win.view, "repaint", lambda: painted.append(win.game.splash_text))
    monkeypatch.setattr(main.track, "evolve_track", lambda rng, w, d: win.game.track)
    QTest.keyClick(win.view, Qt.Key.Key_T)
    assert painted == ["генерация трассы..."]
    assert win.game.splash_text == ""


def test_the_window_place_is_saved_on_close_and_restored(qapp, tmp_path):
    from window_state import screen_areas
    path = tmp_path / "window.json"
    g = shared_game()
    w = window.MainWindow(g, window_path=path)
    w.show_window()
    # экран offscreen маленький (800x600): окно минимального размера и место, где оно
    # влезает целиком, - иначе правило места законно его подвинет
    area = screen_areas()[0]
    width, height = w.minimumWidth(), w.minimumHeight()
    x = area["x"] + max(0, min(12, area["width"] - width))
    y = area["y"] + max(0, min(20, area["height"] - height))
    w.resize(width, height)
    w.move(x, y)
    qapp.processEvents()
    w.close()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert (saved["width"], saved["height"]) == (width, height)
    w2 = window.MainWindow(g, window_path=path)
    assert (w2.x(), w2.y(), w2.width(), w2.height()) == (saved["x"], saved["y"], width, height)
    w2.close()


# --- подписи с переносом строк (закон из крестиков-ноликов) ------------------------------
# Нашёл Алексей 26.09.2026 в TicTacToeAi: подсказке дали высоту ровно в одну строку, а на
# настоящем экране фраза чуть шире, чем в offscreen, - перенеслась и не влезла. Шрифт на 10%
# крупнее изображает это «чуть шире».

def wrapped_labels(widget):
    return [lb for lb in widget.findChildren(QLabel) if lb.wordWrap() and lb.isVisible() and lb.text()]


def test_wrapping_labels_are_never_capped_in_height(win):
    g = win.game
    g.paused = True
    g.say("сохранения нет или оно не подходит")
    win.view.after_input()
    for show in (win.play, win.show_stats, win.show_menu, win.show_settings):
        show()
        QTest.qWait(10)
        labels = wrapped_labels(win)
        assert labels, "нечего проверять"
        capped = [lb.text() for lb in labels if lb.maximumHeight() < QWIDGETSIZE_MAX]
        assert capped == []


LONGEST_MESSAGES = ["мозг загружен, обучение начато заново", "сохранения нет или оно не подходит",
                    "уровень обычный, трасса 40 px", "мозги сохранены: 120 шт"]


@pytest.mark.parametrize("scale", [1.0, 1.1])
def test_the_message_fits_in_the_smallest_window(win, qapp, scale):
    g = win.game
    g.paused = True
    win.resize(win.minimumSize())
    toast = win.view.toast
    toast.setStyleSheet(f"font-size: {round(15 * scale)}px;")
    for text in LONGEST_MESSAGES:
        g.say(text)
        win.view.after_input()
        qapp.processEvents()
        assert toast.heightForWidth(toast.width()) <= toast.height(), text
        assert toast.x() >= 0 and toast.x() + toast.width() <= g.view.right
    toast.setStyleSheet("")


def enlarge(screen, scale):
    for lb in screen.findChildren(QLabel):
        px = lb.font().pixelSize() if lb.font().pixelSize() > 0 else round(lb.font().pointSizeF() * 96 / 72)
        lb.setStyleSheet(f"font-size: {round(px * scale)}px;")


def text_problems(screen):
    """Подписи с переносом, которым не хватило высоты, и подписи без переноса, которым -
    ширины; и подписи, вылезшие за экран (их не видно, хоть они и целы)."""
    clipped = [(lb.text()[:40], lb.height(), lb.heightForWidth(lb.width())) for lb in wrapped_labels(screen)
               if lb.heightForWidth(lb.width()) > lb.height()]
    narrow = [lb.text()[:40] for lb in screen.findChildren(QLabel)
              if lb.isVisible() and not lb.wordWrap() and lb.text()
              and lb.fontMetrics().horizontalAdvance(lb.text().split("\n")[0]) > lb.width() + 1]
    return clipped + narrow


@pytest.mark.parametrize("page", ["stats", "menu", "settings"])
@pytest.mark.parametrize("scale", [1.0, 1.1])
def test_screen_text_fits_in_the_smallest_window(win, qapp, scale, page):
    show, screen = {"stats": (win.show_stats, win.stats_screen), "menu": (win.show_menu, win.menu),
                    "settings": (win.show_settings, win.settings_screen)}[page]
    show()
    enlarge(screen, scale)
    win.resize(win.minimumSize())
    QTest.qWait(30)
    problems = text_problems(screen)
    for lb in screen.findChildren(QLabel):
        lb.setStyleSheet("")
    assert problems == []


# --- меню --------------------------------------------------------------------------------------

@pytest.fixture
def fresh_window(qapp, tmp_path):
    w = window.MainWindow(main.Game(seed=0, start=False), settings_path=tmp_path / "settings.json")
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def test_the_window_opens_on_the_menu(fresh_window):
    w = fresh_window
    assert w.current() is w.menu
    assert w.menu.play_button.text() == "Играть"
    assert [b.text() for b in (w.menu.play_button, w.menu.stats_button, w.menu.settings_button,
                               w.menu.quit_button)] == ["Играть", "Статистика", "Настройки", "Выход"]
    assert not w.view.is_running() and not w.game.started


def test_every_menu_button_opens_its_screen_and_leads_back(fresh_window, monkeypatch):
    w = fresh_window
    monkeypatch.setattr(main.track, "evolve_track", lambda rng, width, difficulty: shared_game().track)
    click = lambda b: QTest.mouseClick(b, Qt.MouseButton.LeftButton)  # noqa: E731

    click(w.menu.stats_button)
    assert w.current() is w.stats_screen and w.stats_screen.back_button.text() == "В меню"
    click(w.stats_screen.back_button)
    assert w.current() is w.menu

    click(w.menu.settings_button)
    assert w.current() is w.settings_screen
    click(w.settings_screen.back_button)
    assert w.current() is w.menu
    w.show_settings()
    QTest.keyClick(w.settings_screen, Qt.Key.Key_Escape)
    assert w.current() is w.menu

    click(w.menu.play_button)
    assert w.current() is w.view and w.game.started and w.view.is_running()
    QTest.keyClick(w.view, Qt.Key.Key_Escape)
    assert w.current() is w.menu

    click(w.menu.quit_button)
    assert not w.isVisible()


def test_the_menu_says_what_the_game_is(fresh_window):
    text = fresh_window.menu.subtitle.text()
    for words in ("нейросети", "эволюцией", "поколение за поколением", "на глазах", "Статистика"):
        assert words in text
    assert 3 <= text.count(".") + text.count(":") <= 8


@pytest.mark.parametrize("scale", [1.0, 1.1])
def test_the_menu_text_is_fully_visible_in_the_smallest_window(fresh_window, qapp, scale):
    w = fresh_window
    enlarge(w.menu, scale)
    w.resize(w.minimumSize())
    QTest.qWait(30)
    about = w.menu.subtitle
    assert about.maximumHeight() == QWIDGETSIZE_MAX
    assert about.heightForWidth(about.width()) <= about.height()
    top_left = about.mapTo(w.menu, QPoint(0, 0))
    assert top_left.y() >= 0 and top_left.y() + about.height() <= w.menu.height()
    assert top_left.x() >= 0 and top_left.x() + about.width() <= w.menu.width()
    lines = about.height() / about.fontMetrics().lineSpacing()
    assert 3 <= round(lines) <= 6, lines
    for b in (w.menu.play_button, w.menu.quit_button):
        at = b.mapTo(w.menu, QPoint(0, 0))
        assert at.y() + b.height() <= w.menu.height()
    for lb in w.menu.findChildren(QLabel):
        lb.setStyleSheet("")


def test_settings_change_the_game_and_survive_a_restart(fresh_window, qapp, tmp_path):
    w = fresh_window
    s = w.settings_screen
    w.show_settings()
    s.sliders["pop_size"][0].setValue(80)
    s.sliders["volume"][0].setValue(15)
    s.combos["level"].setCurrentIndex(0)                  # лёгкий
    s.combos["speed"].setCurrentIndex(2)
    g = w.game
    assert g.ui.value("pop_size") == 80 and abs(g.ui.value("volume") - 0.15) < 1e-9
    assert g.ui.value("level") == "лёгкий" and g.ui.value("width") == main.LEVELS[0][1]
    assert g.ui.value("speed") == "x20"
    assert s.values["pop_size"].text() == "80" and s.values["volume"].text() == "15%"
    QTest.mouseClick(s.back_button, Qt.MouseButton.LeftButton)
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert saved["pop_size"] == 80 and saved["level"] == "лёгкий" and saved["speed"] == "x20"

    again = window.MainWindow(main.Game(seed=0, start=False), settings_path=tmp_path / "settings.json")
    assert again.game.ui.value("pop_size") == 80 and again.game.ui.value("level") == "лёгкий"
    again.show_settings()
    assert again.settings_screen.combos["speed"].currentIndex() == 2
    again.close()


def test_the_settings_screen_shows_what_the_panel_has(fresh_window):
    w = fresh_window
    g = w.game
    g.ui.widgets["generations"].value = 33
    g.ui.widgets["replay"].index = 2
    w.show_settings()
    assert w.settings_screen.sliders["generations"][0].value() == 33
    assert w.settings_screen.combos["replay"].currentText() == main.REPLAY_OFF
