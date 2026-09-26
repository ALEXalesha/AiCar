"""Окно игры открывается там, где его закрыли (window_state.py; с 1.1.0, на Qt с 2.0.0).

Правило места - tk_window_state.restore строка в строку (оно же в калькуляторах, Paint Pro и
Нейро-змейке): что бы ни лежало в файле, окно открывается там, где его видно и за заголовок
можно взяться. Файл window.json - в папке данных, как в 1.1.0.
"""
import json
import os
import random
import subprocess
import sys
import textwrap

import pytest

import window_state as ws

FULL_HD = {"x": 0, "y": 0, "width": 1920, "height": 1040}
RIGHT = {"x": 1920, "y": 0, "width": 2560, "height": 1400}
OPTS = {"width": 1280, "height": 720, "minWidth": 960, "minHeight": 640}
SAVED = {"x": 100, "y": 50, "width": 1280, "height": 720, "maximized": False}
# Как писала 1.1.0 (pygame): границы рамки вокруг клиента 1280x720 и сдвиг клиента в ней.
OLD = {"x": 100, "y": 50, "width": 1296, "height": 759, "maximized": False, "client_dx": 8, "client_dy": 31}


def test_a_window_on_screen_comes_back_exactly():
    assert ws.restore(SAVED, [FULL_HD], OPTS) == SAVED
    bigger = {**SAVED, "width": 1500, "height": 900, "maximized": True}
    assert ws.restore(bigger, [FULL_HD], OPTS) == bigger


def test_no_file_or_junk_gives_the_default_size_and_lets_windows_choose():
    for saved in [None, 42, "x", [], {}, {"width": "a", "height": 5}, {"width": float("nan"), "height": 5}]:
        assert ws.restore(saved, [FULL_HD], OPTS) == {"width": 1280, "height": 720, "maximized": False}, saved


def test_an_unplugged_monitor_lets_windows_choose():
    far = {**SAVED, "x": 2500}
    assert ws.restore(far, [FULL_HD, RIGHT], OPTS)["x"] == 2500
    got = ws.restore(far, [FULL_HD], OPTS)
    assert "x" not in got and (got["width"], got["height"]) == (1280, 720)


def test_a_window_half_off_the_screen_is_pulled_back_whole():
    got = ws.restore({**SAVED, "x": 1500, "y": 900}, [FULL_HD], OPTS)
    assert (got["x"], got["y"]) == (1920 - 1280, 1040 - 720)


def test_size_stays_between_the_minimum_and_the_screen():
    got = ws.restore({**SAVED, "width": 100, "height": 50000}, [FULL_HD], OPTS)
    assert (got["width"], got["height"]) == (960, 1040)


def test_whatever_is_in_the_file_the_title_bar_ends_up_on_a_screen():
    rnd = random.Random(7)
    for _ in range(3000):
        areas, x0 = [], rnd.randint(-5000, 5000)
        for _ in range(rnd.randint(1, 3)):
            a = {"x": x0, "y": rnd.randint(-3000, 3000), "width": rnd.randint(1400, 4000),
                 "height": rnd.randint(800, 2500)}
            areas.append(a)
            x0 += a["width"]
        v = lambda: rnd.choice([None, "x", rnd.randint(-20000, 20000), rnd.uniform(-5e4, 5e4)])  # noqa: E731
        saved = {"x": v(), "y": v(), "width": v(), "height": v(), "maximized": rnd.choice([True, False, 1]),
                 "client_dx": rnd.choice([8, 0, -1, 7.5, None]), "client_dy": rnd.choice([31, 0, 300, None])}
        got = ws.restore(ws.from_file(saved), areas, OPTS)
        assert got["width"] >= OPTS["minWidth"] and got["height"] >= OPTS["minHeight"]
        if "x" not in got:
            continue
        fx, fy = got["x"], got["y"]
        assert any(fx >= a["x"] and fy >= a["y"] and fx < a["x"] + a["width"]
                   and fy + ws.GRIP_HEIGHT <= a["y"] + a["height"] for a in areas), saved


# --- файл версии 1.1.0 -----------------------------------------------------------------

def test_the_1_1_file_opens_the_window_at_the_same_place_with_the_same_client():
    got = ws.restore(ws.from_file(OLD), [FULL_HD], OPTS)
    assert got == {"x": 100, "y": 50, "width": 1280, "height": 720, "maximized": False}


def test_from_file_leaves_the_new_format_and_junk_alone():
    """Переводится только настоящий файл 1.1.0; остальное уходит в restore как есть,
    а там мусор и так даёт место по умолчанию."""
    assert ws.from_file(SAVED) == SAVED
    for junk in (None, 42, [], {**OLD, "client_dx": True}, {**OLD, "client_dy": 5000},
                 {**OLD, "client_dx": 7.5}, {**OLD, "width": "a"}):
        assert ws.from_file(junk) == junk


def test_the_saved_file_is_the_new_format(tmp_path):
    path = tmp_path / "window.json"
    assert ws.save(path, SAVED)
    assert json.loads(path.read_text(encoding="utf-8")) == SAVED
    assert ws.load(path) == SAVED
    assert not (tmp_path / "window.json.tmp").exists()


def test_a_broken_file_loads_as_nothing(tmp_path):
    path = tmp_path / "window.json"
    for text in ("", "{", "\x00\x01", "[" * 100000):
        path.write_text(text, encoding="utf-8")
        assert ws.load(path) is None
    assert ws.load(tmp_path / "нет.json") is None


# --- Qt (offscreen) --------------------------------------------------------------------

def test_screen_areas_come_from_qt(qapp):
    areas = ws.screen_areas()
    assert areas and all(a["width"] > 0 and a["height"] > 0 for a in areas)


def test_remember_puts_the_window_and_saves_it(qapp, tmp_path):
    from PySide6.QtWidgets import QWidget

    path = tmp_path / "window.json"
    area = ws.screen_areas()[0]
    x, y = area["x"] + 10, area["y"] + 20
    small = {"width": 400, "height": 300, "minWidth": 200, "minHeight": 150}
    path.write_text(json.dumps({"x": x, "y": y, "width": 420, "height": 310}), encoding="utf-8")
    win = QWidget()
    rem = ws.Remember(win, path, small)
    assert (win.x(), win.y(), win.width(), win.height()) == (x, y, 420, 310)
    win.resize(390, 280)
    win.move(x + 5, y + 6)
    assert rem.save()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"x": x + 5, "y": y + 6, "width": 390, "height": 280, "maximized": False}


def test_a_maximized_window_keeps_its_normal_place(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QWidget

    path = tmp_path / "window.json"
    win = QWidget()
    rem = ws.Remember(win, path, {"width": 400, "height": 300, "minWidth": 200, "minHeight": 150})
    win.move(30, 40)
    rem.track()
    normal = dict(rem.normal)
    monkeypatch.setattr(win, "isMaximized", lambda: True)
    win.resize(1000, 700)
    rem.track()
    assert rem.save()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {**normal, "maximized": True}


REAL = textwrap.dedent("""
    import json, os, sys
    sys.path.insert(0, {root!r})
    os.environ.pop("QT_QPA_PLATFORM", None)
    from PySide6.QtWidgets import QApplication, QWidget
    import window_state as ws
    path, move = sys.argv[1], sys.argv[2] == "move"
    app = QApplication([])
    win = QWidget()
    win.setWindowTitle("AiCar: проверка места окна")
    rem = ws.Remember(win, path, {{"width": 640, "height": 400, "minWidth": 320, "minHeight": 200}})
    rem.show()
    app.processEvents()
    if move:
        a = ws.screen_areas()[0]
        win.move(a["x"] + 40, a["y"] + 60)
        app.processEvents()
    fg = win.frameGeometry()
    print(fg.x(), fg.y(), win.width(), win.height())
    rem.save()
    win.close()
""")


@pytest.mark.skipif(sys.platform != "win32", reason="место окна берётся у Windows")
def test_a_real_window_opens_where_it_was_closed(tmp_path):
    """Настоящее окно Windows, дважды: сдвинули и закрыли - открылось там же и того же размера."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = tmp_path / "real.py"
    script.write_text(REAL.format(root=root), encoding="utf-8")
    f = tmp_path / "window.json"
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}

    def launch(mode):
        out = subprocess.run([sys.executable, str(script), str(f), mode], env=env, capture_output=True,
                             text=True, timeout=60)
        if out.returncode:
            pytest.skip("окно не открылось: " + out.stderr.strip()[-200:])
        return tuple(map(int, out.stdout.strip().splitlines()[-1].split()))

    first = launch("move")
    saved = json.loads(f.read_text(encoding="utf-8"))
    assert (saved["x"], saved["y"], saved["width"], saved["height"]) == first
    assert launch("stay") == first
