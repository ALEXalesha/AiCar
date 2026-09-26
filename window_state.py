"""Место и размер окна между запусками (window.json в папке данных; с 1.1.0, на Qt с 2.0.0).

Правило выбора места - `restore`, строка в строку то же, что было в `tk_window_state.restore`
(1.1.0) и что стоит в калькуляторах, Paint Pro и Нейро-змейке: что бы ни лежало в файле -
окно на отключённом мониторе, заголовок за экраном, размер больше экрана или меньше минимума,
мусор, обрезанный файл, - окно открывается там, где его видно и за заголовок можно взяться.
Если того монитора нет или заголовок за экраном - место выбирает Windows.

В файле: x, y - левый верхний угол рамки окна (`pos()` у Qt), width и height - внутренняя
часть (`size()`), maximized. Ровно то, что принимают move() и resize().

Файл 1.1.0 (pygame) устроен иначе: width и height там - вся рамка, и есть сдвиг клиентской
области в ней (client_dx, client_dy). `from_file` переводит его в нынешний вид, чтобы окно
после обновления открылось там же и того же размера, что закрылось в 1.1.0.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

# Сколько окна должно остаться на экране, чтобы за него можно было взяться: полоса
# заголовка высотой 38 и хотя бы 80 по ширине.
GRIP_HEIGHT = 38
GRIP_WIDTH = 80
OLD_FRAME_MAX = 200     # сдвиг клиента в рамке у 1.1.0: больше не бывает


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _round(v) -> int:
    """Как Math.round в JS: половина - вверх. round() Python округлял бы 2.5 до 2."""
    return math.floor(v + 0.5)


def _overlap(a, b):
    w = min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])
    h = min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"])
    return (w, h) if w > 0 and h > 0 else None


def restore(saved, areas, opts) -> dict:
    """Где и какого размера открыть окно.

    saved - что лежало в файле (что угодно); areas - рабочие области экранов
    {x, y, width, height}, основной первым; opts - {width, height, minWidth, minHeight}.
    Без x и y в ответе - место выбирает система."""
    by_default = {"width": opts["width"], "height": opts["height"], "maximized": False}
    screens = [a for a in (areas if isinstance(areas, list) else [])
               if isinstance(a, dict) and all(_finite(a.get(k)) for k in ("x", "y", "width", "height"))
               and a["width"] > 0 and a["height"] > 0]
    if not isinstance(saved, dict) or not _finite(saved.get("width")) or not _finite(saved.get("height")):
        return by_default

    maximized = saved.get("maximized") is True
    big_w = max([opts["width"]] + [a["width"] for a in screens])
    big_h = max([opts["height"]] + [a["height"] for a in screens])
    width = _round(min(max(saved["width"], opts["minWidth"]), max(big_w, opts["minWidth"])))
    height = _round(min(max(saved["height"], opts["minHeight"]), max(big_h, opts["minHeight"])))

    if not _finite(saved.get("x")) or not _finite(saved.get("y")):
        return {"width": width, "height": height, "maximized": maximized}
    x, y = _round(saved["x"]), _round(saved["y"])

    grip = {"x": x, "y": y, "width": width, "height": GRIP_HEIGHT}
    best, best_area = None, 0
    for a in screens:
        o = _overlap(grip, a)
        if o and o[0] * o[1] > best_area:
            best, best_area = a, o[0] * o[1]
    if best is None or best_area < GRIP_WIDTH * GRIP_HEIGHT / 2:
        return {"width": width, "height": height, "maximized": maximized}

    width = min(width, max(best["width"], opts["minWidth"]))
    height = min(height, max(best["height"], opts["minHeight"]))
    nx = max(best["x"], min(x, best["x"] + best["width"] - width))
    ny = max(best["y"], min(y, best["y"] + best["height"] - height))
    return {"x": nx, "y": ny, "width": width, "height": height, "maximized": maximized}


def from_file(saved):
    """Файл 1.1.0 (рамка и client_dx/client_dy) - в нынешний вид; всё прочее - как есть."""
    if not isinstance(saved, dict) or "client_dx" not in saved:
        return saved
    dx, dy = saved.get("client_dx"), saved.get("client_dy")
    if not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= OLD_FRAME_MAX for v in (dx, dy)):
        return saved
    if not _finite(saved.get("width")) or not _finite(saved.get("height")):
        return saved
    out = {k: v for k, v in saved.items() if k not in ("client_dx", "client_dy")}
    # Рамка Windows: по dx слева, справа и снизу, сверху - заголовок dy.
    out["width"] = saved["width"] - 2 * dx
    out["height"] = saved["height"] - dy - dx
    return out


def screen_areas() -> list:
    """Рабочие области экранов (без панели задач) от Qt, основной первым."""
    from PySide6.QtGui import QGuiApplication

    primary = QGuiApplication.primaryScreen()
    screens = [primary] + [s for s in QGuiApplication.screens() if s is not primary]
    out = []
    for s in screens:
        if s is None:
            continue
        g = s.availableGeometry()
        out.append({"x": g.x(), "y": g.y(), "width": g.width(), "height": g.height()})
    return out


def load(path):
    """Прочитать файл; нет файла или в нём мусор - None."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return None


def save(path, state: dict) -> bool:
    """Записать через временный файл: убитый посреди записи процесс оставит прежний файл."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        return False


class Remember:
    """Ставит окно по файлу и помнит обычные границы (у развёрнутого - те, к которым оно
    вернётся). Окно зовёт track() из moveEvent/resizeEvent и save() при закрытии."""

    def __init__(self, window, path, opts: dict):
        self.window = window
        self.path = Path(path)
        self.normal = None
        placed = restore(from_file(load(self.path)), screen_areas(), opts)
        window.resize(placed["width"], placed["height"])
        if "x" in placed:
            window.move(placed["x"], placed["y"])
        self.maximize = placed["maximized"]     # развернуть при первом показе
        self.track()

    def show(self):
        """Показать окно так, как его закрыли."""
        if self.maximize:
            self.window.showMaximized()
        else:
            self.window.show()

    def track(self):
        w = self.window
        if not (w.isMaximized() or w.isMinimized() or w.isFullScreen()):
            self.normal = {"x": w.x(), "y": w.y(), "width": w.width(), "height": w.height()}

    def save(self) -> bool:
        self.track()
        if self.normal is None:
            return False
        return save(self.path, {**self.normal, "maximized": self.window.isMaximized()})
