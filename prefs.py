"""Настройки между запусками (settings.json в папке данных, с 2.0.0).

До 2.0.0 ползунки и переключатели панели каждый запуск начинались с умолчаний. Теперь окно
пишет их при закрытии и при уходе с экрана настроек, а при запуске ставит обратно. Хранятся
значения ползунков и подписи вариантов переключателей (не номера: порядок вариантов может
поменяться, а «сложный» останется «сложным»). Мусор, чужие ключи и значения вне пределов
не ломают запуск: непонятное пропускается, числа зажимаются пределами ползунков.
"""
import json
import math
import os
from pathlib import Path

import config as cfg
import main

SLIDERS = ("pop_size", "generations", "mut_sigma", "elite_frac", "width", "difficulty", "volume")
TOGGLES = ("level", "generator", "speed", "brain", "replay")

DEFAULTS = {
    "pop_size": cfg.POP_SIZE, "generations": cfg.MAX_GENERATIONS, "mut_sigma": cfg.MUT_SIGMA,
    "elite_frac": cfg.ELITE_FRAC, "width": int(cfg.TRACK_WIDTH), "difficulty": cfg.DIFFICULTY,
    "volume": cfg.VOLUME, "level": main.LEVEL_NAMES[main.DEFAULT_LEVEL], "generator": main.GEN_CPPN,
    "speed": main.SPEED_NAMES[0], "brain": main.BRAIN_NAMES[0], "replay": main.REPLAY_NAMES[0],
}


def collect(game):
    """Что сейчас на панели игры."""
    widgets = game.ui.widgets
    out = {key: widgets[key].value for key in SLIDERS}
    out.update({key: widgets[key].value for key in TOGGLES})
    return out


def apply(game, saved):
    """Поставить сохранённое на панель. Уровень ставится вместе с «уже применён», чтобы
    apply_level не затёр сохранённые ширину и сложность ширинами уровня."""
    if not isinstance(saved, dict):
        return
    widgets = game.ui.widgets
    for key in TOGGLES:
        value = saved.get(key)
        if isinstance(value, str) and value in widgets[key].options:
            widgets[key].index = widgets[key].options.index(value)
            if key == "level":
                game.level_shown = widgets[key].index
    for key in SLIDERS:
        value = saved.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            widgets[key].value = value


def load(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


def save(path, data):
    """Через временный файл: убитый посреди записи процесс оставит прежний файл."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        return False
