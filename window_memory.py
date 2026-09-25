"""Окно игры открывается там, где его закрыли (1.1.0).

Размер у окна постоянный (1280x720), так что запоминается только место. SDL ставит окно
по клиентской области, а заголовок, за который окно берут, - над ней; поэтому в файл идут
границы всего окна (GetWindowRect) и сдвиг клиентской области внутри них. При запуске
место выбирается тем же правилом, что в калькуляторах, Paint Pro и Tkinter-окнах
(tk_window_state.restore), и отдаётся SDL через SDL_VIDEO_WINDOW_POS до создания окна -
окно сразу появляется на месте, а не прыгает туда после показа.

Всё это только на Windows и только с настоящим окном: в тестах (SDL_VIDEODRIVER=dummy)
окна у Windows нет, и файл не читается и не пишется.
"""
import os
import sys

import tk_window_state as ws

ENV = "SDL_VIDEO_WINDOW_POS"


def plan(saved, areas, client_w, client_h):
    """Куда поставить клиентскую область: строка 'x,y' для SDL_VIDEO_WINDOW_POS или None
    (нет файла, мусор, того монитора нет) - тогда место выбирает Windows, как раньше."""
    if not isinstance(saved, dict):
        return None
    dx, dy = saved.get("client_dx"), saved.get("client_dy")
    if not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 200 for v in (dx, dy)):
        return None
    # Размер окна - от нынешнего клиента и сохранённых рамок, не из файла целиком.
    w, h = client_w + 2 * dx, client_h + dy + dx
    placed = ws.restore({**saved, "width": w, "height": h, "maximized": False}, areas,
                        {"width": w, "height": h, "minWidth": w, "minHeight": h})
    if "x" not in placed:
        return None
    return f"{placed['x'] + dx},{placed['y'] + dy}"


def before_window(path):
    """Звать до pygame.display.set_mode()."""
    if sys.platform != "win32" or os.environ.get("SDL_VIDEODRIVER") == "dummy":
        return
    import config as cfg
    pos = plan(ws.load(path), ws.work_areas(), cfg.WINDOW_W, cfg.WINDOW_H)
    if pos:
        os.environ[ENV] = pos


def _rects(hwnd):
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    frame = wintypes.RECT()
    origin = wintypes.POINT(0, 0)
    if not user32.GetWindowRect(hwnd, ctypes.byref(frame)) or not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    return frame, origin


def capture(hwnd):
    """Границы окна и сдвиг клиентской области в них; None - не вышло."""
    got = _rects(hwnd)
    if got is None:
        return None
    frame, origin = got
    return {"x": frame.left, "y": frame.top, "width": frame.right - frame.left,
            "height": frame.bottom - frame.top, "maximized": False,
            "client_dx": origin.x - frame.left, "client_dy": origin.y - frame.top}


def remember(path):
    """Звать перед pygame.quit(), пока окно ещё есть."""
    if sys.platform != "win32" or os.environ.get("SDL_VIDEODRIVER") == "dummy":
        return False
    import pygame
    hwnd = pygame.display.get_wm_info().get("window") if pygame.display.get_init() else None
    state = capture(hwnd) if hwnd else None
    return bool(state) and ws.save(path, state)
