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


def test_summary_is_trimmed_to_the_panel_width(qapp):
    """Ширина - в пикселях шрифта панели (QFontMetrics), а не в числе знаков."""
    import config as cfg
    import render

    font = render.font(15)
    width = cfg.PANEL_W - 28
    for rounds in (1, 999, 123456):
        totals = dict(stats.EMPTY, rounds=rounds, finished=rounds, best_time=98765.4)
        assert render.text_width(font, render.fit_text(font, stats.summary(totals), width)) <= width


# --- 2.0.0: подробности о раундах для экрана статистики ----------------------------------

class Gen2(Gen):
    def __init__(self, finished=0, best_time=0.0, best=100.0, mean=40.0):
        super().__init__(finished, best_time)
        self.best, self.mean = best, mean


INFO = {"level": "сложный", "generator": "cppn", "pop": 50, "progress": 0.5,
        "track": {"length": 2000.0, "radius": 30.0, "interest": 55.0, "width": 34.0, "build_s": 1.9},
        "car": {"speed": 200.0, "steer": 2.9, "mass": 1.7}}


def test_without_info_a_round_is_recorded_exactly_as_in_1_1():
    totals = stats.record_round(stats.load_totals("нет такого файла"), [Gen2(), Gen2(1, 9.5)])
    assert totals["log"] == [] and totals["last_curve"] == {"best": [], "mean": []}
    assert (totals["rounds"], totals["finished"], totals["generations"]) == (1, 1, 2)


def test_with_info_the_round_goes_to_the_log():
    history = [Gen2(best=10.0, mean=2.0), Gen2(best=50.0, mean=9.0), Gen2(1, 12.345, 4700.0, 300.0)]
    totals = stats.record_round(dict(stats.EMPTY), history, INFO)
    entry = totals["log"][-1]
    assert entry["n"] == 1 and entry["gens"] == 3 and entry["finished"] is True
    assert entry["time"] == 12.35 and entry["level"] == "сложный" and entry["track"]["interest"] == 55.0
    assert totals["last_curve"] == {"best": [10.0, 50.0, 4700.0], "mean": [2.0, 9.0, 300.0]}


def test_a_round_without_a_finish_has_no_time():
    totals = stats.record_round(dict(stats.EMPTY), [Gen2(0, 0.0)], INFO)
    assert totals["log"][-1]["finished"] is False and totals["log"][-1]["time"] is None


def test_the_log_keeps_only_the_last_rounds():
    totals = dict(stats.EMPTY)
    for _ in range(stats.LOG_LIMIT + 7):
        totals = stats.record_round(totals, [Gen2()], INFO)
    assert len(totals["log"]) == stats.LOG_LIMIT
    assert totals["log"][0]["n"] == 8 and totals["log"][-1]["n"] == stats.LOG_LIMIT + 7


def test_recording_does_not_touch_the_shared_empty_totals():
    before = json.dumps(stats.EMPTY, sort_keys=True)
    stats.record_round(stats.load_totals("нет такого файла"), [Gen2(1, 5.0)], INFO)
    stats.record_round(dict(stats.EMPTY), [Gen2(1, 5.0)], INFO)
    assert json.dumps(stats.EMPTY, sort_keys=True) == before


def test_the_log_survives_a_save_and_load(tmp_path):
    path = str(tmp_path / "stats.json")
    totals = stats.record_round(dict(stats.EMPTY), [Gen2(), Gen2(1, 8.0)], INFO)
    stats.save_totals(totals, path)
    assert stats.load_totals(path) == totals


def test_a_1_1_file_loads_with_an_empty_log(tmp_path):
    path = tmp_path / "stats.json"
    old = {"rounds": 12, "finished": 9, "generations": 170, "best_time": 8.4,
           "gens_to_finish": [5, 12, 30, 7, 9, 21, 16, 3, 18]}
    path.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = stats.load_totals(str(path))
    assert {k: loaded[k] for k in old} == old
    assert loaded["log"] == [] and loaded["last_curve"] == {"best": [], "mean": []}
    assert stats.summary(loaded) == stats.summary(dict(stats.EMPTY, **old))


def test_junk_in_the_new_keys_is_dropped(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"rounds": 3, "log": "мусор", "last_curve": [1, 2]}), encoding="utf-8")
    loaded = stats.load_totals(str(path))
    assert loaded["rounds"] == 3 and loaded["log"] == [] and loaded["last_curve"] == {"best": [], "mean": []}
    path.write_text(json.dumps({"rounds": "x", "log": [1, {"n": 1}, None], "best_time": "y"}), encoding="utf-8")
    loaded = stats.load_totals(str(path))
    assert loaded["rounds"] == 0 and loaded["log"] == [{"n": 1}] and loaded["best_time"] is None
