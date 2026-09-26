"""Настройки между запусками (prefs.py, 2.0.0): ползунки и переключатели панели в settings.json."""
import functools
import json

import main
import prefs
from qt_app import qapp


@functools.lru_cache(maxsize=1)
def game():
    qapp()
    return main.Game(seed=0, start=False)


def fresh():
    g = game()
    prefs.apply(g, prefs.DEFAULTS)
    return g


def test_the_defaults_are_what_a_new_game_starts_with(qapp):
    assert prefs.collect(main.Game(seed=0, start=False)) == prefs.DEFAULTS


def test_a_missing_or_broken_file_gives_nothing(tmp_path):
    assert prefs.load(tmp_path / "нет.json") == {}
    for text in ("", "{", "[1, 2]", "42", "\x00"):
        (tmp_path / "s.json").write_text(text, encoding="utf-8")
        assert prefs.load(tmp_path / "s.json") == {}


def test_collect_takes_every_setting_of_the_panel():
    got = prefs.collect(fresh())
    assert set(got) == set(prefs.SLIDERS) | set(prefs.TOGGLES)
    assert got["pop_size"] == 50 and got["level"] == "сложный" and got["speed"] == "x1"


def test_apply_then_collect_is_the_same():
    g = fresh()
    wanted = dict(prefs.collect(g), pop_size=80, volume=0.15, speed="x20", replay="по кругу",
                  level="лёгкий", width=46, difficulty=0.4)
    prefs.apply(g, wanted)
    assert prefs.collect(g) == wanted


def test_a_saved_custom_width_survives_the_level_switch():
    """Ширину сдвинули руками при уровне «обычный»: после запуска apply_level не должен
    вернуть ширину уровня - иначе настройка не пережила перезапуск."""
    g = fresh()
    prefs.apply(g, dict(prefs.collect(g), level="обычный", width=63, difficulty=0.2))
    g.apply_level()
    assert g.ui.value("width") == 63 and abs(g.ui.value("difficulty") - 0.2) < 1e-9
    assert g.ui.value("level") == "обычный"


def test_junk_values_are_ignored_and_ranges_are_kept():
    g = fresh()
    before = prefs.collect(g)
    prefs.apply(g, {"pop_size": "много", "volume": 7.0, "level": "невозможный", "speed": 3,
                    "mut_sigma": float("nan"), "лишнее": 1, "generations": True, "width": -5})
    got = prefs.collect(g)
    assert got["volume"] == 1.0 and got["width"] == 25          # зажаты в пределы ползунков
    for key in ("pop_size", "level", "speed", "mut_sigma", "generations"):
        assert got[key] == before[key], key


def test_apply_ignores_anything_that_is_not_a_dict():
    g = fresh()
    before = prefs.collect(g)
    for junk in (None, [], "x", 5):
        prefs.apply(g, junk)
    assert prefs.collect(g) == before


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    data = prefs.collect(fresh())
    assert prefs.save(path, data)
    assert prefs.load(path) == data
    assert json.loads(path.read_text(encoding="utf-8")) == data
    assert not (tmp_path / "settings.json.tmp").exists()
