"""Экран статистики (stats_screen.py, 2.0.0): числа с файла, старый и пустой stats.json."""
import json

import pytest
from PySide6.QtWidgets import QLabel

import stats
import stats_screen as ss
import theme

OLD = {"rounds": 12, "finished": 9, "generations": 170, "best_time": 8.4,
       "gens_to_finish": [5, 12, 30, 7, 9, 21, 16, 3, 18]}


def entry(n, level="сложный", generator="cppn", finished=True, gens=10, time=9.5, progress=1.0,
          speed=200.0, steer=2.8, mass=1.6, interest=55.0, length=2000.0, build=2.0):
    return {"n": n, "level": level, "generator": generator, "finished": finished, "gens": gens,
            "time": time if finished else None, "progress": progress, "pop": 50,
            "track": {"length": length, "radius": 30.0, "interest": interest, "width": 34.0, "build_s": build},
            "car": {"speed": speed, "steer": steer, "mass": mass}}


def with_log():
    log = [entry(1, gens=12, time=10.0), entry(2, finished=False, gens=60, progress=0.4, speed=180.0),
           entry(3, level="лёгкий", generator="model", gens=4, time=7.25, interest=61.0, build=0.05),
           entry(4, gens=20, time=9.0), entry(5, level="свой", finished=False, progress=0.7, gens=60)]
    totals = dict(stats.EMPTY, rounds=5, finished=3, generations=156, best_time=7.25,
                  gens_to_finish=[12, 4, 20], log=log,
                  last_curve={"best": [100.0, 300.0, 5000.0], "mean": [20.0, 80.0, 900.0]})
    return totals


@pytest.fixture
def screen(qapp):
    s = ss.StatsScreen()
    s.setStyleSheet(theme.QSS)
    s.resize(1100, 800)
    s.show()
    yield s
    s.close()


def load_file(tmp_path, data):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats.load_totals(str(path))


def cards(s):
    return {k: v.text() for k, v in s.values.items()}


def test_an_empty_file_shows_zeros_and_no_charts(screen, tmp_path):
    screen.refresh(stats.load_totals(str(tmp_path / "нет.json")))
    assert cards(screen) == {"rounds": "0", "finished": "0", "generations": "0", "best_time": "-"}
    # главный график не пропадает, а говорит, откуда возьмётся; остальные без данных скрыты
    assert screen.visible_charts() == 1 and screen.chart_training.isVisible()
    assert not screen.chart_training.has_data() and "первый раунд" in screen.chart_training.empty_text
    assert "ещё нет" in screen.bench_label.text()
    assert not screen.levels_note.isVisible()
    assert not screen.grab().isNull()


def test_a_1_1_file_shows_its_numbers_and_what_it_cannot(screen, tmp_path):
    screen.refresh(load_file(tmp_path, OLD))
    assert cards(screen) == {"rounds": "12", "finished": "9", "generations": "170", "best_time": "8.4 с"}
    assert screen.notes["finished"].text() == "75% раундов"
    assert screen.notes["generations"].text() == f"до финиша в среднем {sum(OLD['gens_to_finish']) / 9:.1f}"
    # поколений до финиша - единственный график, данные для которого были и в 1.1.0;
    # главный виден с подписью: кривой обучения в файле 1.1.0 нет
    assert not screen.chart_gens.isHidden() and screen.visible_charts() == 2
    assert not screen.chart_training.has_data()
    assert screen.chart_gens.points[-1] == (9.0, 18.0)
    assert screen.levels_note.isVisible() and "(12)" in screen.levels_note.text()
    assert screen.level_cells["сложный", "rounds"].text() == "0"
    assert not screen.grab().isNull()


def test_the_numbers_come_from_the_file(screen, tmp_path):
    totals = load_file(tmp_path, with_log())
    screen.refresh(totals)
    assert cards(screen) == {"rounds": "5", "finished": "3", "generations": "156", "best_time": "7.2 с"}
    cell = {key: screen.level_cells["сложный", key].text() for key in ("rounds", "finished", "gens", "time")}
    assert cell == {"rounds": "3", "finished": "2", "gens": "16.0", "time": "9.0 с"}
    assert screen.level_cells["лёгкий", "time"].text() == "7.2 с"
    assert screen.level_cells["свой", "rounds"].text() == "1"
    assert screen.level_cells["адский", "rounds"].text() == "0" and screen.level_cells["адский", "gens"].text() == "-"
    assert screen.gen_cells["cppn", "rounds"].text() == "4" and screen.gen_cells["model", "rounds"].text() == "1"
    assert screen.gen_cells["model", "interest"].text() == "61.0"
    assert screen.gen_cells["model", "build"].text() == "50 мс"
    assert not screen.levels_note.isVisible()
    assert screen.visible_charts() == 4


def test_the_curves_come_from_the_log(tmp_path):
    points = ss.curves(load_file(tmp_path, with_log()))
    assert points["training"] == [(0, 100.0), (1, 300.0), (2, 5000.0)]
    assert points["time"] == [(1, 10.0), (3, 7.25), (4, 9.0)]
    assert points["progress"][1] == (2, 0.4)
    assert points["gens"] == [(1, 12), (2, 4), (3, 20)]


def test_the_live_round_wins_over_the_saved_curve(tmp_path):
    class S:
        def __init__(self, best, mean):
            self.best, self.mean = best, mean
    points = ss.curves(load_file(tmp_path, with_log()), [S(1.0, 0.5), S(2.0, 1.0)])
    assert points["training"] == [(0, 1.0), (1, 2.0)] and points["training_mean"][1] == (1, 1.0)


def test_the_benchmark_turns_green_only_when_not_worse(screen):
    good = dict(stats.EMPTY, rounds=6, finished=6,
                log=[entry(i, gens=g) for i, g in enumerate((10, 12, 14, 15, 16, 20), 1)])
    screen.refresh(good)
    info = ss.compare(good)
    assert info["share_ok"] and info["gens_ok"] and info["median_gens"] == 14.5
    assert screen.bench_label.text().count(theme.GOOD) == 2

    bad = dict(stats.EMPTY, rounds=4, finished=2,
               log=[entry(1, gens=30), entry(2, gens=40), entry(3, finished=False), entry(4, finished=False)])
    screen.refresh(bad)
    info = ss.compare(bad)
    assert not info["share_ok"] and not info["gens_ok"]
    assert screen.bench_label.text().count(theme.WARN) == 2


def test_cars_that_finish_are_compared_with_those_that_do_not(tmp_path):
    info = ss.cars(load_file(tmp_path, with_log()))
    assert info["finished"]["rounds"] == 3 and info["crashed"]["rounds"] == 2
    assert info["finished"]["speed"] == 200.0 and info["crashed"]["speed"] == 190.0


def test_junk_in_the_log_is_skipped_not_shown(screen):
    junk = dict(stats.EMPTY, rounds=3, finished=1,
                log=[{"n": "x", "level": 5, "finished": "да", "car": [], "track": None},
                     {"n": 2, "finished": True, "gens": "много", "time": float("nan")}, entry(3)])
    screen.refresh(junk)
    assert screen.level_cells["сложный", "rounds"].text() == "1"
    assert screen.level_cells["свой", "rounds"].text() == "2"
    assert not screen.grab().isNull()


# --- главный график: обучение водителей по поколениям ---------------------------------------

@pytest.fixture(scope="module")
def real_stats_file(tmp_path_factory):
    """stats.json, записанный настоящей игрой: два раунда до конца на быстрой скорости."""
    import config as cfg
    import main
    from qt_app import qapp
    qapp()
    folder = tmp_path_factory.mktemp("real")
    saved = cfg.STATS_FILE
    cfg.STATS_FILE = str(folder / "stats.json")
    try:
        g = main.Game(seed=3)
        g.ui.widgets["speed"].index = 3
        g.ui.widgets["generations"].value = 7
        for _ in range(2):
            for _ in range(20):
                g.frame()
                if g.state != main.TRAINING:
                    break
            if g.rounds < 2:
                g.new_round(new_track=False)
    finally:
        cfg.STATS_FILE = saved
    return folder / "stats.json"


def test_the_chart_is_drawn_from_a_real_stats_file(screen, real_stats_file):
    raw = json.loads(real_stats_file.read_text(encoding="utf-8"))
    screen.refresh(stats.load_totals(str(real_stats_file)))
    chart = screen.chart_training
    best, mean = raw["last_curve"]["best"], raw["last_curve"]["mean"]
    assert len(best) >= 2 and chart.has_data() and chart.isVisible()
    assert len(chart.points) == len(best) == raw["log"][-1]["gens"]
    assert len(chart.second) == len(mean)
    assert chart.points[-1][1] == best[-1] and chart.second[-1][1] == mean[-1]
    assert chart.corner()[1][0] == ss.chart_fmt(best[-1])            # последнее - в углу
    assert chart.title.endswith(f"раунд {raw['log'][-1]['n']}")
    image = chart.grab().toImage()
    gold = [(x, y) for x in range(0, image.width(), 3) for y in range(40, image.height() - 46, 3)
            if image.pixelColor(x, y).name() == theme.GOLD]
    assert gold, "линии лучшего результата на графике нет"


def test_an_empty_file_draws_a_caption_instead_of_an_empty_chart(screen, tmp_path):
    screen.refresh(stats.load_totals(str(tmp_path / "нет.json")))
    chart = screen.chart_training
    assert chart.isVisible() and not chart.has_data() and chart.points == []
    with_caption = chart.grab().toImage()
    text, chart.empty_text = chart.empty_text, ""
    without = chart.grab().toImage()
    chart.empty_text = text
    assert with_caption != without, "подписи вместо графика не видно"


def test_the_live_round_is_shown_while_training(screen, real_stats_file):
    class S:
        def __init__(self, best, mean):
            self.best, self.mean = best, mean
    screen.refresh(stats.load_totals(str(real_stats_file)), [S(10, 1), S(20, 2), S(35, 4)], 9, live=True)
    assert screen.chart_training.title.endswith("раунд 9, идёт")
    assert screen.chart_training.points[-1] == (2.0, 35.0)


def test_the_chart_is_readable_in_the_smallest_window(qapp, real_stats_file):
    import main
    import window
    w = window.MainWindow(main.Game(seed=0, start=False))
    w.show()
    w.resize(w.minimumSize())
    w.stats_screen.refresh(stats.load_totals(str(real_stats_file)))
    w.stack.setCurrentWidget(w.stats_screen)
    qapp.processEvents()
    chart = w.stats_screen.chart_training
    title_font, small = chart.fonts()
    from PySide6.QtGui import QFontMetrics
    corner = sum(QFontMetrics(title_font).horizontalAdvance(t) for t, _ in chart.corner())
    title = QFontMetrics(title_font).horizontalAdvance(chart.title)
    w.close()
    assert chart.width() >= 600, chart.width()
    assert title + corner + 40 <= chart.width(), "заголовок и последние значения не влезают рядом"


def test_all_texts_on_the_screen_are_russian_and_filled(screen, tmp_path):
    screen.refresh(load_file(tmp_path, with_log()))
    empty = [lb.objectName() for lb in screen.findChildren(QLabel) if lb.isVisible() and not lb.text()]
    assert empty == []
