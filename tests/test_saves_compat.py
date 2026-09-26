"""Сохранения версии 1.1.0 (pygame) читаются версией 2.0.0 (Qt).

Файлы в tests/data записаны кодом 1.1.0 (git archive тега 1.1.0, три настоящих раунда на
лёгкой трассе): мозги - stats.save_brains, итоги - stats.save_totals, место окна -
tk_window_state.save в том виде, в каком его писал window_memory.capture.
"""
import json
import os
import shutil

import numpy as np

import config as cfg
import stats
import window_state as ws

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BRAINS = os.path.join(DATA, "brains_1.1.0.npz")
TOTALS = os.path.join(DATA, "stats_1.1.0.json")
WINDOW = os.path.join(DATA, "window_1.1.0.json")


def test_brains_of_1_1_load_and_the_best_is_the_same():
    genomes, fitness = stats.load_brains(BRAINS)
    assert genomes.shape == (30, 112) and len(fitness) == 30
    assert int(np.argmax(fitness)) == 23 and fitness.max() == 5066.0
    assert np.array_equal(stats.best_brain(BRAINS), genomes[23])


def test_the_load_button_takes_the_1_1_brain(qapp):
    import main
    shutil.copy(BRAINS, cfg.BRAIN_FILE)
    g = main.Game(seed=0)
    g.load_brains()
    genomes, fitness = stats.load_brains(BRAINS)
    assert np.array_equal(g.best_brain, genomes[23])
    assert g.message == "мозг загружен, обучение начато заново"
    assert g.state == main.TRAINING and g.race.brains is g.brains


def test_totals_of_1_1_load_as_they_were():
    raw = json.load(open(TOTALS, encoding="utf-8"))
    loaded = stats.load_totals(TOTALS)
    assert {k: loaded[k] for k in raw} == raw
    assert stats.summary(loaded) == "всего 3  доехало 3  рекорд 7.7с"
    assert loaded["log"] == []


def test_a_new_round_extends_the_1_1_totals_and_keeps_them_readable(tmp_path):
    class Gen:
        finished, best_time, best, mean = 1, 6.5, 4800.0, 900.0

    path = str(tmp_path / "stats.json")
    shutil.copy(TOTALS, path)
    info = {"level": "лёгкий", "generator": "cppn", "pop": 30, "progress": 1.0,
            "track": {"length": 2100.0}, "car": {"speed": 190.0}}
    totals = stats.record_round(stats.load_totals(path), [Gen()] * 4, info)
    stats.save_totals(totals, path)
    again = stats.load_totals(path)
    assert (again["rounds"], again["finished"], again["generations"]) == (4, 4, 35)
    assert again["gens_to_finish"] == [6, 5, 20, 4] and again["best_time"] == 6.5
    assert len(again["log"]) == 1 and again["log"][0]["n"] == 4


def test_the_stats_screen_shows_the_1_1_numbers(qapp):
    import stats_screen
    s = stats_screen.StatsScreen()
    s.refresh(stats.load_totals(TOTALS))
    assert [s.values[k].text() for k in ("rounds", "finished", "generations", "best_time")] == \
        ["3", "3", "31", "7.7 с"]
    assert s.chart_gens.points == [(1.0, 6.0), (2.0, 5.0), (3.0, 20.0)]
    assert s.levels_note.isVisibleTo(s) and "(3)" in s.levels_note.text()


def test_the_window_of_1_1_opens_where_it_was(qapp):
    saved = ws.load(WINDOW)
    assert "client_dx" in saved
    full_hd = {"x": 0, "y": 0, "width": 1920, "height": 1040}
    opts = {"width": 1280, "height": 720, "minWidth": 960, "minHeight": 640}
    assert ws.restore(ws.from_file(saved), [full_hd], opts) == \
        {"x": 212, "y": 96, "width": 1280, "height": 720, "maximized": False}


def test_the_main_window_reads_the_1_1_window_file(qapp, tmp_path):
    import main
    import window
    path = tmp_path / "window.json"
    shutil.copy(WINDOW, path)
    w = window.MainWindow(main.Game(seed=0, start=False), window_path=path)
    want = ws.restore(ws.from_file(ws.load(path)), ws.screen_areas(),
                      {"width": cfg.WINDOW_W, "height": cfg.WINDOW_H, "minWidth": main.MIN_W,
                       "minHeight": main.MIN_H})
    assert (w.width(), w.height()) == (want["width"], want["height"])
    if "x" in want:
        assert (w.x(), w.y()) == (want["x"], want["y"])
    w.close()
    assert "client_dx" not in json.loads(path.read_text(encoding="utf-8"))
