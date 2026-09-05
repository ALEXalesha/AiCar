import json

import numpy as np

import brain
import stats


class Gen:
    def __init__(self, finished=0, best_time=0.0):
        self.finished = finished
        self.best_time = best_time


def brains(n=5, size=None):
    size = brain.genome_size() if size is None else size
    return np.random.default_rng(0).normal(0, 1, (n, size))


def test_save_and_load_returns_the_same_array(tmp_path):
    path = str(tmp_path / "brains.npz")
    g = brains()
    stats.save_brains(g, np.arange(5.0), path)
    loaded, fitness = stats.load_brains(path)
    assert np.allclose(loaded, g)
    assert np.allclose(fitness, np.arange(5.0))


def test_save_creates_missing_folders(tmp_path):
    path = str(tmp_path / "deep" / "nested" / "brains.npz")
    stats.save_brains(brains(), np.zeros(5), path)
    assert stats.load_brains(path) is not None


def test_missing_file_loads_as_nothing(tmp_path):
    assert stats.load_brains(str(tmp_path / "absent.npz")) is None


def test_a_genome_of_the_wrong_length_is_rejected(tmp_path):
    path = str(tmp_path / "brains.npz")
    stats.save_brains(brains(size=brain.genome_size() + 7), np.zeros(5), path)
    assert stats.load_brains(path) is None


def test_a_corrupt_file_is_rejected(tmp_path):
    path = tmp_path / "brains.npz"
    path.write_text("это не npz", encoding="utf-8")
    assert stats.load_brains(str(path)) is None


def test_best_brain_picks_the_highest_fitness(tmp_path):
    path = str(tmp_path / "brains.npz")
    g = brains()
    stats.save_brains(g, np.array([1.0, 9.0, 3.0, 2.0, 0.0]), path)
    assert np.allclose(stats.best_brain(path), g[1])


def test_best_brain_of_a_missing_file_is_nothing(tmp_path):
    assert stats.best_brain(str(tmp_path / "absent.npz")) is None


def test_missing_totals_load_as_empty(tmp_path):
    assert stats.load_totals(str(tmp_path / "absent.json")) == stats.EMPTY


def test_corrupt_totals_load_as_empty(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text("{ это не json", encoding="utf-8")
    assert stats.load_totals(str(path)) == stats.EMPTY


def test_totals_of_the_wrong_shape_load_as_empty(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert stats.load_totals(str(path)) == stats.EMPTY


def test_unknown_keys_are_dropped_on_load(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"rounds": 4, "мусор": 1}), encoding="utf-8")
    loaded = stats.load_totals(str(path))
    assert loaded["rounds"] == 4
    assert "мусор" not in loaded


def test_totals_survive_a_save_and_load(tmp_path):
    path = str(tmp_path / "stats.json")
    totals = stats.record_round(stats.load_totals(path), [Gen(1, 12.5)])
    stats.save_totals(totals, path)
    assert stats.load_totals(path) == totals


def test_counters_add_up_across_rounds():
    totals = dict(stats.EMPTY)
    totals = stats.record_round(totals, [Gen(), Gen(), Gen(1, 20.0)])
    totals = stats.record_round(totals, [Gen(), Gen()])
    assert totals["rounds"] == 2
    assert totals["finished"] == 1
    assert totals["generations"] == 5


def test_best_time_keeps_the_fastest():
    totals = dict(stats.EMPTY)
    totals = stats.record_round(totals, [Gen(1, 20.0)])
    totals = stats.record_round(totals, [Gen(1, 14.5)])
    totals = stats.record_round(totals, [Gen(1, 18.0)])
    assert totals["best_time"] == 14.5


def test_an_empty_round_changes_nothing():
    totals = dict(stats.EMPTY)
    assert stats.record_round(totals, []) == totals


def test_record_round_does_not_mutate_the_original():
    totals = dict(stats.EMPTY)
    stats.record_round(totals, [Gen(1, 9.0)])
    assert totals == stats.EMPTY


def test_average_generations_to_finish():
    totals = dict(stats.EMPTY)
    totals = stats.record_round(totals, [Gen(), Gen(1, 5.0)])
    totals = stats.record_round(totals, [Gen(), Gen(), Gen(), Gen(1, 6.0)])
    assert stats.average_gens_to_finish(totals) == 3.0


def test_average_is_nothing_without_a_single_finish():
    assert stats.average_gens_to_finish(dict(stats.EMPTY)) is None


def test_summary_of_an_empty_history():
    assert "раундов нет" in stats.summary(dict(stats.EMPTY))


def test_summary_reports_the_record():
    totals = stats.record_round(dict(stats.EMPTY), [Gen(1, 11.25)])
    text = stats.summary(totals)
    assert "всего 1" in text and "11.2" in text


def test_summary_fits_the_panel():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    import config as cfg

    pygame.init()
    font = pygame.font.SysFont("consolas", 15)
    totals = dict(stats.EMPTY, rounds=999, finished=999, best_time=123.45)
    assert font.size(stats.summary(totals))[0] <= cfg.PANEL_W - 28
