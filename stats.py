import json
import os

import numpy as np

import brain
import config as cfg

EMPTY = {"rounds": 0, "finished": 0, "generations": 0, "best_time": None, "gens_to_finish": []}


def _ensure_dir(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


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
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return dict(EMPTY)
    if not isinstance(saved, dict):
        return dict(EMPTY)
    totals = dict(EMPTY)
    totals.update({k: v for k, v in saved.items() if k in EMPTY})
    return totals


def save_totals(totals, path=None):
    path = cfg.STATS_FILE if path is None else path
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(totals, f, ensure_ascii=False, indent=2)
    return path


def record_round(totals, history):
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
