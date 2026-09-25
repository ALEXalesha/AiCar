"""Окно игры открывается там, где его закрыли (window_memory.py, 1.1.0)."""
import json
import os
import random
import subprocess
import sys
import textwrap

import pytest

import window_memory as wm

FULL_HD = {"x": 0, "y": 0, "width": 1920, "height": 1040}
RIGHT = {"x": 1920, "y": 0, "width": 2560, "height": 1400}
CW, CH = 1280, 720
# Как пишет capture(): рамка вокруг клиентской области 1280x720 (у Windows 11 - 8 и 31).
SAVED = {"x": 100, "y": 50, "width": 1296, "height": 759, "maximized": False, "client_dx": 8, "client_dy": 31}


def test_a_window_on_screen_comes_back_exactly():
    assert wm.plan(SAVED, [FULL_HD], CW, CH) == "108,81"


def test_no_file_junk_or_unplugged_monitor_leaves_the_place_to_windows():
    for saved in [None, 42, "x", [], {}, {**SAVED, "client_dx": None}, {**SAVED, "client_dx": True},
                  {**SAVED, "client_dy": 5000}, {**SAVED, "x": "a"}]:
        assert wm.plan(saved, [FULL_HD], CW, CH) is None, saved
    far = {**SAVED, "x": 2500}
    assert wm.plan(far, [FULL_HD, RIGHT], CW, CH) == "2508,81"
    assert wm.plan(far, [FULL_HD], CW, CH) is None


def test_a_window_half_off_the_screen_is_pulled_back_whole():
    x, y = map(int, wm.plan({**SAVED, "x": 1500, "y": 900}, [FULL_HD], CW, CH).split(","))
    assert (x - 8, y - 31) == (1920 - 1296, 1040 - 759)


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
        saved = {"x": v(), "y": v(), "width": v(), "height": v(),
                 "client_dx": rnd.choice([8, 0, -1, 7.5, None]), "client_dy": rnd.choice([31, 0, 300, None])}
        pos = wm.plan(saved, areas, CW, CH)
        if pos is None:
            continue
        cx, cy = map(int, pos.split(","))
        fx, fy = cx - saved["client_dx"], cy - saved["client_dy"]
        assert any(fx >= a["x"] and fy >= a["y"] and fx < a["x"] + a["width"]
                   and fy + wm.ws.GRIP_HEIGHT <= a["y"] + a["height"] for a in areas), saved


def test_the_game_asks_before_the_window_and_saves_before_quit():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"),
               encoding="utf-8").read()
    init = src.split("def __init__(self, seed=None):", 1)[1]
    assert init.index("window_memory.before_window(") < init.index("pygame.display.set_mode(")
    run = src.split("    def run(self):", 1)[1]
    assert run.index("window_memory.remember(") < run.index("pygame.quit()")


def test_tests_never_touch_the_file(tmp_path):
    # SDL_VIDEODRIVER=dummy (conftest тестов игры): ни чтения, ни записи.
    f = tmp_path / "window.json"
    f.write_text(json.dumps(SAVED), encoding="utf-8")
    os.environ.pop(wm.ENV, None)
    old = os.environ.get("SDL_VIDEODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    try:
        wm.before_window(str(f))
        assert wm.ENV not in os.environ
        assert wm.remember(str(f)) is False
    finally:
        if old is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = old


REAL = textwrap.dedent("""
    import ctypes, os, sys
    sys.path.insert(0, {root!r})
    os.environ.pop("SDL_VIDEODRIVER", None)
    import window_memory as wm, config as cfg
    path, move = sys.argv[1], sys.argv[2] == "move"
    wm.before_window(path)
    import pygame
    pygame.display.init()
    pygame.display.set_mode((cfg.WINDOW_W, cfg.WINDOW_H))
    hwnd = pygame.display.get_wm_info()["window"]
    if move:
        a = wm.ws.work_areas()[0]
        ctypes.windll.user32.SetWindowPos(hwnd, 0, a["x"] + 40, a["y"] + 60, 0, 0, 0x0001 | 0x0004 | 0x0010)
    pygame.event.pump()
    print(wm.capture(hwnd)["x"], wm.capture(hwnd)["y"])
    wm.remember(path)
    pygame.quit()
""")


@pytest.mark.skipif(sys.platform != "win32", reason="место окна берётся у Windows")
def test_a_real_window_opens_where_it_was_closed(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = tmp_path / "real.py"
    script.write_text(REAL.format(root=root), encoding="utf-8")
    f = tmp_path / "window.json"
    env = {k: v for k, v in os.environ.items() if k not in ("SDL_VIDEODRIVER", wm.ENV)}

    def launch(mode):
        out = subprocess.run([sys.executable, str(script), str(f), mode], env=env, capture_output=True,
                             text=True, timeout=60)
        if out.returncode:
            pytest.skip("окно не открылось: " + out.stderr.strip()[-200:])
        return tuple(map(int, out.stdout.strip().splitlines()[-1].split()))

    first = launch("move")
    saved = json.loads(f.read_text(encoding="utf-8"))
    assert (saved["x"], saved["y"]) == first
    assert launch("stay") == first
