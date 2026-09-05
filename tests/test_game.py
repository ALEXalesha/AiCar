import functools
import os
import tempfile

import numpy as np
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import main


@functools.lru_cache(maxsize=1)
def game():
    return main.Game(seed=0)


def fresh():
    g = game()
    g.watched = main.WATCH_LEADER
    g.hud = main.render.hud_rect(g.view)
    g.hud_grab = None
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


def test_clicking_a_car_starts_watching_it():
    g = fresh()
    target = 11
    point = g.camera.to_screen(g.race.pos[target])[0]
    g.click_field((int(point[0]), int(point[1])))
    assert g.race.watch(g.watched) in range(g.race.n)
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
    g.key(pygame.K_n)
    assert g.watched == main.WATCH_NONE
    g.key(pygame.K_l)
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
    g.screen.fill(main.render.BG)
    g.draw_field()
    surface = pygame.surfarray.array3d(g.screen)
    return np.any(surface != np.array(main.render.BG), axis=2)[g.panel_rect.left:, :]


def test_the_field_never_paints_on_the_side_panel():
    assert not field_ink_on_the_panel(fresh()).any()


def test_without_the_clip_the_field_would_reach_the_panel():
    """Показывает, что предыдущий тест не проходит сам собой."""
    g = fresh()
    g.camera.fit(g.track.lo, g.track.hi)
    edge = [g.camera.to_world((g.view.right - k, y)) for k in (2, 12) for y in (120, 600)]
    g.race.pos[:len(edge)] = np.array(edge)
    g.screen.fill(main.render.BG)
    main.render.draw_cars(g.screen, g.camera, g.race, g.car, None)
    surface = pygame.surfarray.array3d(g.screen)
    spill = np.any(surface != np.array(main.render.BG), axis=2)[g.panel_rect.left:, :]
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
