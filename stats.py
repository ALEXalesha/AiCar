import json
import os
import tempfile

import numpy as np

import brain
import config as cfg

# log и last_curve копятся с 2.0.0 для экрана статистики. В stats.json 1.1.0 их нет -
# такой файл читается как раньше, а они берутся пустыми.
EMPTY = {"rounds": 0, "finished": 0, "generations": 0, "best_time": None, "gens_to_finish": [],
         "log": [], "last_curve": {"best": [], "mean": []}}
LOG_LIMIT = 300


def _empty():
    """Свежая копия EMPTY: списки в ней свои, общий EMPTY никто не испортит."""
    return {**EMPTY, "gens_to_finish": [], "log": [], "last_curve": {"best": [], "mean": []}}


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _clean(totals):
    """Что пришло из файла - к ожидаемым типам. Мусорный ключ - как в пустом файле."""
    fresh = _empty()
    for key in ("rounds", "finished", "generations"):
        if not (isinstance(totals[key], int) and not isinstance(totals[key], bool)):
            totals[key] = fresh[key]
    if totals["best_time"] is not None and not _number(totals["best_time"]):
        totals["best_time"] = None
    if not isinstance(totals["gens_to_finish"], list):
        totals["gens_to_finish"] = []
    log = totals["log"]
    totals["log"] = [e for e in log if isinstance(e, dict)][-LOG_LIMIT:] if isinstance(log, list) else []
    curve = totals["last_curve"]
    if not (isinstance(curve, dict) and all(isinstance(curve.get(k), list) for k in ("best", "mean"))):
        totals["last_curve"] = fresh["last_curve"]
    return totals


def _ensure_dir(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def use_sandbox(prefix):
    """Увести запись мозгов и счётчиков во временную папку.

    Для всего, что играет настоящий раунд не ради игрока: самопроверка, проверка
    инвариантов, сборка кадров для README. Конец раунда пишет статистику, и без этого
    каждый такой прогон добавлял игроку раунд, которого он не играл.
    """
    folder = os.path.join(tempfile.mkdtemp(prefix=prefix), "saves")
    os.makedirs(folder, exist_ok=True)
    cfg.SAVE_DIR = folder
    cfg.BRAIN_FILE = os.path.join(folder, "brains.npz")
    cfg.STATS_FILE = os.path.join(folder, "stats.json")
    return folder


def save_brains(genomes, fitness, path=None):
    path = cfg.BRAIN_FILE if path is None else path
    _ensure_dir(path)
    np.savez(path, genomes=np.asarray(genomes, dtype=float), fitness=np.asarray(fitness, dtype=float))
    return path


def load_brains(path=None, expect=None):
    path = cfg.BRAIN_FILE if path is None else path
    expect = brain.genome_size() if expect is None else expect
    try:
        with np.load(path) as data:
            genomes, fitness = data["genomes"], data["fitness"]
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None
    if genomes.ndim != 2 or genomes.shape[1] != expect or len(fitness) != len(genomes):
        return None
    return genomes, fitness


def best_brain(path=None, expect=None):
    loaded = load_brains(path, expect)
    if loaded is None:
        return None
    genomes, fitness = loaded
    return genomes[int(np.argmax(fitness))].copy()


def load_totals(path=None):
    path = cfg.STATS_FILE if path is None else path
    try:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return _empty()
    if not isinstance(saved, dict):
        return _empty()
    totals = _empty()
    totals.update({k: v for k, v in saved.items() if k in EMPTY})
    return _clean(totals)


def save_totals(totals, path=None):
    path = cfg.STATS_FILE if path is None else path
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(totals, f, ensure_ascii=False, indent=2)
    return path


def record_round(totals, history, info=None):
    """Итоги после раунда. Без info - ровно как в 1.1.0. С info (что было в раунде: уровень,
    генератор, трасса, машинка, доля трассы у лучшего - main.Game.round_info) ещё запись
    в log и кривая обучения раунда в last_curve - для экрана статистики."""
    totals = dict(totals)
    if not history:
        return totals

    last = history[-1]
    totals["rounds"] += 1
    totals["generations"] += len(history)
    if last.finished:
        totals["finished"] += 1
        totals["gens_to_finish"] = totals["gens_to_finish"] + [len(history)]
        if totals["best_time"] is None or last.best_time < totals["best_time"]:
            totals["best_time"] = last.best_time
    if info is not None:
        entry = dict(info)
        entry.update(n=totals["rounds"], gens=len(history), finished=bool(last.finished),
                     time=round(float(last.best_time), 2) if last.finished else None)
        totals["log"] = (list(totals.get("log", [])) + [entry])[-LOG_LIMIT:]
        totals["last_curve"] = {"best": [round(float(s.best), 1) for s in history],
                                "mean": [round(float(s.mean), 1) for s in history]}
    return totals


def average_gens_to_finish(totals):
    gens = totals["gens_to_finish"]
    return sum(gens) / len(gens) if gens else None


def summary(totals):
    """Итоги одной строкой. Слитно, для тестов и консоли."""
    left, right = summary_parts(totals)
    return left if not right else f"{left}  {right}"


def summary_parts(totals, short=False):
    """Итоги двумя половинами: счётчики влево, рекорд вправо.

    Панель рисует их по разным краям, поэтому длина каждой считается отдельно.
    `short` даёт запасной, укороченный вид подписи рекорда: при четырёхзначных
    счётчиках полное слово уже не влезает, а само число обрезать нельзя - оно и
    есть то, ради чего строка существует.
    """
    if not totals["rounds"]:
        return "всего: раундов нет", ""
    left = f"всего {totals['rounds']}  доехало {totals['finished']}"
    if totals["best_time"] is None:
        return left, ""
    label = "рек" if short else "рекорд"
    return left, f"{label} {totals['best_time']:.1f}с"
