import pygame
import pytest

import ui


def press(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button)


def release(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=button)


def move(pos):
    return pygame.event.Event(pygame.MOUSEMOTION, pos=pos)


def slider(lo=0.0, hi=100.0, value=50.0, **kw):
    return ui.Slider(pygame.Rect(100, 200, 200, ui.SLIDER_H), "тест", lo, hi, value, **kw)


def test_slider_starts_at_the_given_value():
    assert slider().value == 50.0


def test_slider_clamps_on_both_ends():
    s = slider()
    s.value = -10.0
    assert s.value == 0.0
    s.value = 999.0
    assert s.value == 100.0


def test_integer_slider_rounds():
    s = slider(integer=True)
    s.value = 12.7
    assert s.value == 13 and isinstance(s.value, int)


def test_clicking_the_bar_sets_the_value():
    s = slider()
    s.handle(press((s.bar.left, s.bar.centery)))
    assert s.value == 0.0
    s.handle(press((s.bar.right - 1, s.bar.centery)))
    assert s.value > 99.0


def test_clicking_the_middle_gives_the_middle_value():
    s = slider()
    s.handle(press((s.bar.centerx, s.bar.centery)))
    assert abs(s.value - 50.0) < 1.0


def test_dragging_only_works_after_grabbing():
    s = slider()
    assert not s.handle(move((s.bar.left, s.bar.centery)))
    assert s.value == 50.0
    s.handle(press((s.bar.centerx, s.bar.centery)))
    assert s.handle(move((s.bar.left, s.bar.centery)))
    assert s.value == 0.0


def test_releasing_stops_the_drag():
    s = slider()
    s.handle(press((s.bar.centerx, s.bar.centery)))
    s.handle(release((s.bar.centerx, s.bar.centery)))
    s.handle(move((s.bar.left, s.bar.centery)))
    assert abs(s.value - 50.0) < 1.0


def test_clicking_outside_does_not_grab():
    s = slider()
    assert not s.handle(press((0, 0)))
    assert s.value == 50.0


def test_button_fires_only_on_release_inside():
    b = ui.Button(pygame.Rect(0, 0, 100, ui.BUTTON_H), "жми")
    b.handle(press((50, 10)))
    assert b.handle(release((50, 10)))
    assert b.take()


def test_button_does_not_fire_when_released_outside():
    b = ui.Button(pygame.Rect(0, 0, 100, ui.BUTTON_H), "жми")
    b.handle(press((50, 10)))
    assert not b.handle(release((500, 500)))
    assert not b.take()


def test_button_does_not_fire_without_a_press():
    b = ui.Button(pygame.Rect(0, 0, 100, ui.BUTTON_H), "жми")
    assert not b.handle(release((50, 10)))
    assert not b.take()


def test_take_consumes_the_click():
    b = ui.Button(pygame.Rect(0, 0, 100, ui.BUTTON_H), "жми")
    b.handle(press((50, 10)))
    b.handle(release((50, 10)))
    assert b.take()
    assert not b.take()


def test_toggle_cycles_through_options():
    t = ui.Toggle(pygame.Rect(0, 0, 100, ui.TOGGLE_H), "скорость", ["x1", "x5", "x20"])
    assert t.value == "x1"
    t.handle(release((50, 10)))
    assert t.value == "x5"
    t.handle(release((50, 10)))
    t.handle(release((50, 10)))
    assert t.value == "x1"


def test_toggle_ignores_clicks_outside():
    t = ui.Toggle(pygame.Rect(0, 0, 100, ui.TOGGLE_H), "скорость", ["x1", "x5"])
    assert not t.handle(release((500, 500)))
    assert t.value == "x1"


def test_panel_stacks_widgets_without_overlap():
    p = ui.Panel(pygame.Rect(0, 0, 300, 720), None)
    p.slider("a", "a", 0, 1, 0.5)
    p.toggle("b", "b", ["один", "два"])
    p.buttons([("c", "c"), ("d", "d")])
    boxes = [w.rect for w in p.widgets.values()]
    for i, box in enumerate(boxes):
        for other in boxes[i + 1:]:
            assert not box.colliderect(other)


def test_panel_keeps_widgets_inside_its_width():
    p = ui.Panel(pygame.Rect(20, 0, 300, 720), None)
    p.slider("a", "a", 0, 1, 0.5)
    p.buttons([("b", "b"), ("c", "c"), ("d", "d")], per_row=3)
    for w in p.widgets.values():
        assert w.rect.left >= 20 + p.pad
        assert w.rect.right <= 320 - p.pad


def test_panel_reads_values_and_clicks():
    p = ui.Panel(pygame.Rect(0, 0, 300, 720), None)
    p.slider("size", "размер", 10, 90, 40, integer=True)
    p.buttons([("go", "старт")])
    assert p.value("size") == 40
    box = p.widgets["go"].rect
    p.handle(press(box.center))
    p.handle(release(box.center))
    assert p.clicked("go")
    assert not p.clicked("go")


def test_panel_reports_whether_anything_changed():
    p = ui.Panel(pygame.Rect(0, 0, 300, 720), None)
    p.slider("size", "размер", 10, 90, 40, integer=True)
    assert not p.handle(press((900, 900)))
    assert p.handle(press(p.widgets["size"].bar.center))


def test_graph_survives_an_empty_history():
    pygame.init()
    surf = pygame.Surface((200, 100))
    ui.Graph(pygame.Rect(0, 0, 200, 100)).draw(surf, [])


def test_graph_survives_a_single_generation():
    pygame.init()
    surf = pygame.Surface((200, 100))
    ui.Graph(pygame.Rect(0, 0, 200, 100)).draw(surf, [_Stat(1.0, 1.0)])


def test_graph_handles_all_zero_history():
    pygame.init()
    surf = pygame.Surface((200, 100))
    ui.Graph(pygame.Rect(0, 0, 200, 100)).draw(surf, [_Stat(0.0, 0.0), _Stat(0.0, 0.0)])


class _Stat:
    def __init__(self, best, mean):
        self.best = best
        self.mean = mean
