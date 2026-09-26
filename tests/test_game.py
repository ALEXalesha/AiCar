import functools
import os
import tempfile

import numpy as np

import main
import render
from qt_app import qapp


@functools.lru_cache(maxsize=1)
def game():
    qapp()
    return main.Game(seed=0)


def fresh():
    g = game()
    g.watched = main.WATCH_LEADER
    g.hud = main.render.hud_rect(g.view)
    g.hud_grab = None
    g.race.pos[:] = g.track.start_pos
    return g


def spread(g):
    """Развести машинки по трассе.

    На старте все 50 стоят в одной точке, и клик по координатам любой из них
    выбирает нулевую. Для проверок выбора машинки мышью это бесполезно.
    """
    step = len(g.track.center) // g.race.n
    g.race.pos[:] = g.track.center[:g.race.n * step:step]
    return g


def test_watching_nobody_hides_the_telemetry():
    g = fresh()
    g.watched = main.WATCH_NONE
    assert g.telemetry() is None


def test_watching_the_leader_picks_the_best_car():
    g = fresh()
    assert g.telemetry()["index"] == g.race.leader


def test_watching_a_chosen_car_keeps_it():
    g = fresh()
    g.watched = 7
    assert g.telemetry()["index"] == 7


def test_telemetry_reports_what_the_network_sees():
    g = fresh()
    watch = g.telemetry()
    assert len(watch["rays"]) == main.cfg.N_RAYS
    assert np.all(watch["rays"] >= 0.0) and np.all(watch["rays"] <= 1.0)
    assert -1.0 <= watch["steer"] <= 1.0
    assert -1.0 <= watch["throttle"] <= 1.0


def click_car(g, index):
    point = g.camera.to_screen(g.race.pos[index])[0]
    g.click_field((int(point[0]), int(point[1])))


def a_pickable_car(g):
    """Машинка, по которой клик попадёт именно в неё, и не та, что показана."""
    shown = g.telemetry()["index"]
    for i in range(g.race.n):
        point = g.camera.to_screen(g.race.pos[i])[0]
        if i != shown and g.race.nearest_to(g.camera.to_world(point)) == i:
            return i
    raise AssertionError("не нашлось машинки, по которой можно кликнуть")


def test_clicking_a_car_starts_watching_it():
    g = spread(fresh())
    target = a_pickable_car(g)
    click_car(g, target)
    assert g.watched == target
    assert g.telemetry()["index"] == target


def test_clicking_the_watched_car_again_stops_watching():
    g = spread(fresh())
    target = a_pickable_car(g)
    click_car(g, target)
    click_car(g, target)
    assert g.watched == main.WATCH_NONE
    assert g.telemetry() is None


def test_clicking_a_car_after_that_brings_the_panel_back():
    g = spread(fresh())
    target = a_pickable_car(g)
    click_car(g, target)
    click_car(g, target)
    click_car(g, target)
    assert g.telemetry() is not None


def test_the_cross_closes_the_telemetry():
    g = fresh()
    g.click_field(main.render.hud_close_rect(g.hud).center)
    assert g.watched == main.WATCH_NONE
    assert g.telemetry() is None
    assert g.hud_grab is None


def test_the_cross_is_not_the_drag_handle():
    g = fresh()
    cross = main.render.hud_close_rect(g.hud)
    g.click_field((cross.left - 20, cross.centery))
    assert g.hud_grab is not None
    assert g.watched != main.WATCH_NONE


def test_clicking_empty_asphalt_stops_watching():
    g = fresh()
    far = g.camera.to_screen(g.track.center[len(g.track.center) // 2])[0]
    g.click_field((int(far[0]), int(far[1])))
    assert g.watched == main.WATCH_NONE


def test_clicking_the_panel_changes_nothing():
    g = fresh()
    before = g.watched
    g.click_field((g.panel_rect.centerx, g.panel_rect.centery))
    assert g.watched == before


def test_clicking_the_telemetry_grabs_it_instead_of_a_car():
    g = fresh()
    before = g.watched
    g.click_field(g.hud.center)
    assert g.hud_grab is not None
    assert g.watched == before


def test_dragging_moves_the_telemetry():
    g = fresh()
    g.click_field(g.hud.center)
    g.drag_hud((500, 300))
    assert g.hud.collidepoint(500, 300)


def test_the_telemetry_cannot_leave_the_field():
    g = fresh()
    g.click_field(g.hud.center)
    for target in ((-4000, -4000), (9000, 9000)):
        g.drag_hud(target)
        assert g.view.contains(g.hud)


def test_dragging_without_a_grab_does_nothing():
    g = fresh()
    before = g.hud.topleft
    g.drag_hud((500, 300))
    assert g.hud.topleft == before


def test_replay_toggle_has_three_modes():
    assert game().ui.widgets["replay"].options == list(main.REPLAY_NAMES)


def test_panel_widgets_fit_the_window():
    g = game()
    for widget in g.ui.widgets.values():
        assert g.panel_rect.contains(widget.rect)


def test_keys_switch_the_watch_mode():
    g = fresh()
    g.key(main.KEY_NOBODY)
    assert g.watched == main.WATCH_NONE
    g.key(main.KEY_LEADER)
    assert g.watched == main.WATCH_LEADER


def test_levels_go_from_wide_to_narrow():
    widths = [width for _, width, _ in main.LEVELS]
    assert widths == sorted(widths, reverse=True)


def test_switching_the_level_moves_the_sliders():
    g = game()
    for index, (name, width, difficulty) in enumerate(main.LEVELS):
        g.ui.widgets["level"].index = index
        g.apply_level()
        assert g.ui.value("level") == name
        assert g.ui.value("width") == width
        assert abs(g.ui.value("difficulty") - difficulty) < 1e-9


def test_the_level_is_applied_only_when_it_changes():
    g = game()
    g.ui.widgets["level"].index = 0
    g.apply_level()
    g.ui.widgets["width"].value = 71
    g.apply_level()
    assert g.ui.value("width") == 71


def test_every_level_width_is_inside_the_slider_range():
    g = game()
    bar = g.ui.widgets["width"]
    for _, width, _ in main.LEVELS:
        assert bar.lo <= width <= bar.hi


def field_ink_on_the_panel(g):
    g.camera.fit(g.track.lo, g.track.hi)
    edge = [g.camera.to_world((g.view.right - k, y)) for k in (2, 12) for y in (120, 600)]
    g.race.pos[:len(edge)] = np.array(edge)
    return render.ink(g.snapshot(g.draw_field), render.BG)[g.panel_rect.left:, :]


def test_the_field_never_paints_on_the_side_panel():
    assert not field_ink_on_the_panel(fresh()).any()


def test_without_the_clip_the_field_would_reach_the_panel():
    """Показывает, что предыдущий тест не проходит сам собой."""
    g = fresh()
    g.camera.fit(g.track.lo, g.track.hi)
    edge = [g.camera.to_world((g.view.right - k, y)) for k in (2, 12) for y in (120, 600)]
    g.race.pos[:len(edge)] = np.array(edge)
    img = g.snapshot(lambda p: render.draw_cars(p, g.camera, g.race, g.car, None))
    spill = render.ink(img, render.BG)[g.panel_rect.left:, :]
    assert spill.any()


def test_the_race_always_drives_the_current_brains():
    g = fresh()
    assert g.race.brains is g.brains


def test_the_trace_is_reset_for_every_showcase():
    g = fresh()
    g.best_brain = g.brains[0].copy()
    g.start_showcase()
    first = len(g.path)
    for _ in range(20):
        g.advance()
    grown = len(g.path)
    g.start_showcase()
    assert first == 1 and grown > first and len(g.path) == 1


def test_saving_writes_the_fitness_of_the_trained_generation():
    import stats

    g = fresh()
    g.last_fitness = np.arange(len(g.brains), dtype=float)
    g.start_showcase()
    assert g.race.n == 1 and len(g.brains) > 1

    path = os.path.join(tempfile.mkdtemp(prefix="aicar-test-"), "brains.npz")
    stats.save_brains(g.brains, g.last_fitness, path)
    genomes, fitness = stats.load_brains(path)
    assert np.allclose(genomes, g.brains)
    assert np.allclose(fitness, g.last_fitness)


def test_selftest_leaves_the_player_saves_alone(monkeypatch, tmp_path):
    """--selftest играет настоящий раунд до конца, а конец раунда пишет статистику.

    Без песочницы каждая самопроверка добавляла игроку раунд «доехал за 4 поколения»:
    в настоящем stats.json такие записи и нашлись, по одной на запуск.
    """
    player = tmp_path / "игрок"
    monkeypatch.setattr(main.cfg, "SAVE_DIR", str(player))
    monkeypatch.setattr(main.cfg, "STATS_FILE", str(player / "stats.json"))
    monkeypatch.setattr(main.cfg, "BRAIN_FILE", str(player / "brains.npz"))

    lines = main.selftest(str(tmp_path / "отчёт.txt"))

    assert lines[-1] == "OK"
    assert any("состояние done" in line for line in lines), "раунд не доигран - проверка ничего не доказала"
    assert any("шрифт:" in line and "НЕ" not in line for line in lines), lines
    assert not player.exists(), sorted(p.name for p in player.iterdir())


# --- 2.0.0: окно на Qt ------------------------------------------------------------------

def test_the_mouse_is_dispatched_like_the_pygame_loop():
    """Нажатие - щелчок по полю и панели; движение с хватом - тянет окошко, а не ползунки;
    отпускание - бросает окошко."""
    g = fresh()
    g.mouse_down(g.hud.center)
    assert g.hud_grab is not None
    g.mouse_move((500, 300))
    assert g.hud.collidepoint(500, 300)
    g.mouse_up((500, 300))
    assert g.hud_grab is None
    before = g.hud.topleft
    g.mouse_move((100, 100))
    assert g.hud.topleft == before and g.mouse == (100, 100)


def test_panel_buttons_work_through_the_mouse():
    g = fresh()
    g.paused = False
    box = g.ui.widgets["pause"].rect
    g.mouse_down(box.center)
    g.mouse_up(box.center)
    g.apply_buttons()
    assert g.paused
    g.paused = False


def test_the_stats_button_asks_the_window_for_the_stats_screen():
    g = fresh()
    box = g.ui.widgets["stats"].rect
    g.mouse_down(box.center)
    g.mouse_up(box.center)
    g.frame()
    assert g.want_stats
    g.want_stats = False


def test_a_message_lives_for_its_frames():
    g = fresh()
    g.paused = True
    g.say("проверка")
    for _ in range(main.MESSAGE_FRAMES - 1):
        g.frame()
    assert g.message_left == 1
    g.frame()
    assert g.message_left == 0
    g.paused = False


def test_resizing_splits_the_window_between_field_and_panel():
    g = fresh()
    try:
        for w, h in ((main.MIN_W, main.MIN_H), (1920, 1017), (1280, 720)):
            g.resize(w, h)
            assert g.view.right == g.panel_rect.left and g.panel_rect.right == w
            assert g.view.height == g.panel_rect.height == h
            assert g.camera.rect is g.view
            for widget in g.ui.widgets.values():
                assert g.panel_rect.contains(widget.rect), (w, h, widget.rect)
            assert g.view.contains(g.hud)
    finally:
        g.resize(main.cfg.WINDOW_W, main.cfg.WINDOW_H)


def test_the_telemetry_stays_in_the_corner_until_it_is_moved():
    g = fresh()
    g.hud_moved = False
    try:
        g.resize(1600, 900)
        assert g.hud == main.render.hud_rect(g.view)
        g.click_field(g.hud.center)
        g.drag_hud((700, 300))
        g.hud_grab = None
        moved = g.hud.topleft
        g.resize(1500, 850)
        assert g.hud.topleft == moved
    finally:
        g.hud_moved = False
        g.resize(main.cfg.WINDOW_W, main.cfg.WINDOW_H)


def test_the_whole_frame_paints_the_field_and_the_panel():
    g = fresh()
    img = g.snapshot()
    assert (img.width(), img.height()) == (g.width, g.height)
    pixels = render.pixels(img)
    assert (pixels[g.panel_rect.left + 2, g.height // 2] == render.PANEL_BG).all()
    assert render.ink(img, render.BG)[:g.view.right].any()


def test_before_the_first_round_only_the_splash_is_drawn(qapp):
    g = main.Game(seed=1, start=False)
    assert not g.started
    img = g.snapshot()
    assert render.ink(img, render.BG)[:g.view.right].any(), "заставки не видно"
    assert not render.ink(img, render.BG)[g.view.right:].any()


def test_the_level_name_follows_the_sliders():
    g = fresh()
    saved = g.ui.value("width"), g.ui.value("difficulty")
    try:
        for name, width, difficulty in main.LEVELS:
            g.ui.widgets["width"].value = width
            g.ui.widgets["difficulty"].value = difficulty
            assert g.level_name() == name
        g.ui.widgets["width"].value = 55
        assert g.level_name() == main.LEVEL_CUSTOM
    finally:
        g.ui.widgets["width"].value, g.ui.widgets["difficulty"].value = saved


def test_a_finished_round_records_what_the_stats_screen_needs(tmp_path, monkeypatch):
    import stats
    monkeypatch.setattr(main.cfg, "STATS_FILE", str(tmp_path / "stats.json"))
    g = main.Game(seed=5)
    g.ui.widgets["speed"].index = 3
    g.ui.widgets["generations"].value = 2
    for _ in range(10):
        g.frame()
        if g.state != main.TRAINING:
            break
    assert g.state != main.TRAINING
    saved = stats.load_totals()
    assert saved["rounds"] == 1 and len(saved["log"]) == 1
    entry = saved["log"][0]
    assert entry["level"] == "сложный" and entry["generator"] in ("cppn", "model")
    assert entry["gens"] == len(g.history) and entry["n"] == 1
    assert 0.0 <= entry["progress"] <= 1.0
    assert entry["track"]["length"] == round(float(g.track.length), 1)
    assert entry["car"]["speed"] == round(float(g.car.max_speed), 1)
    assert len(saved["last_curve"]["best"]) == len(g.history)
