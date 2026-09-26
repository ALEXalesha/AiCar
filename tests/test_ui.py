import pytest

import render
import ui
from rect import Rect
from ui import move, press, release


def slider(lo=0.0, hi=100.0, value=50.0, **kw):
    return ui.Slider(Rect(100, 200, 200, ui.SLIDER_H), "тест", lo, hi, value, **kw)


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


def test_the_right_button_does_not_grab():
    s = slider()
    assert not s.handle(press((s.bar.left, s.bar.centery), button=3))
    assert s.value == 50.0


def test_button_fires_only_on_release_inside():
    b = ui.Button(Rect(0, 0, 100, ui.BUTTON_H), "жми")
    b.handle(press((50, 10)))
    assert b.handle(release((50, 10)))
    assert b.take()


def test_button_does_not_fire_when_released_outside():
    b = ui.Button(Rect(0, 0, 100, ui.BUTTON_H), "жми")
    b.handle(press((50, 10)))
    assert not b.handle(release((500, 500)))
    assert not b.take()


def test_button_does_not_fire_without_a_press():
    b = ui.Button(Rect(0, 0, 100, ui.BUTTON_H), "жми")
    assert not b.handle(release((50, 10)))
    assert not b.take()


def test_take_consumes_the_click():
    b = ui.Button(Rect(0, 0, 100, ui.BUTTON_H), "жми")
    b.handle(press((50, 10)))
    b.handle(release((50, 10)))
    assert b.take()
    assert not b.take()


def test_toggle_cycles_through_options():
    t = ui.Toggle(Rect(0, 0, 100, ui.TOGGLE_H), "скорость", ["x1", "x5", "x20"])
    assert t.value == "x1"
    t.handle(release((50, 10)))
    assert t.value == "x5"
    t.handle(release((50, 10)))
    t.handle(release((50, 10)))
    assert t.value == "x1"


def test_toggle_ignores_clicks_outside():
    t = ui.Toggle(Rect(0, 0, 100, ui.TOGGLE_H), "скорость", ["x1", "x5"])
    assert not t.handle(release((500, 500)))
    assert t.value == "x1"


def test_panel_stacks_widgets_without_overlap():
    p = ui.Panel(Rect(0, 0, 300, 720), None)
    p.slider("a", "a", 0, 1, 0.5)
    p.toggle("b", "b", ["один", "два"])
    p.buttons([("c", "c"), ("d", "d")])
    boxes = [w.rect for w in p.widgets.values()]
    for i, box in enumerate(boxes):
        for other in boxes[i + 1:]:
            assert not box.colliderect(other)


def test_panel_keeps_widgets_inside_its_width():
    p = ui.Panel(Rect(20, 0, 300, 720), None)
    p.slider("a", "a", 0, 1, 0.5)
    p.buttons([("b", "b"), ("c", "c"), ("d", "d")], per_row=3)
    for w in p.widgets.values():
        assert w.rect.left >= 20 + p.pad
        assert w.rect.right <= 320 - p.pad


def test_panel_reads_values_and_clicks():
    p = ui.Panel(Rect(0, 0, 300, 720), None)
    p.slider("size", "размер", 10, 90, 40, integer=True)
    p.buttons([("go", "старт")])
    assert p.value("size") == 40
    box = p.widgets["go"].rect
    p.handle(press(box.center))
    p.handle(release(box.center))
    assert p.clicked("go")
    assert not p.clicked("go")


def test_panel_reports_whether_anything_changed():
    p = ui.Panel(Rect(0, 0, 300, 720), None)
    p.slider("size", "размер", 10, 90, 40, integer=True)
    assert not p.handle(press((900, 900)))
    assert p.handle(press(p.widgets["size"].bar.center))


# --- раскладка по высоте (2.0.0: окно меняет размер) ------------------------------------

def game_like_panel(height):
    """Та же раскладка, что у игры (main.Game.build_panel), без самой игры."""
    import main
    p = ui.Panel(Rect(980, 0, 300, height), None)
    p.skip(main.STATS_H, main.STATS_LEAST)
    p.graph("graph", main.GRAPH_H, main.GRAPH_LEAST)
    for i in range(7):
        p.slider(f"s{i}", "п", 0, 1, 0.5)
    for i in range(5):
        p.toggle(f"t{i}", "т", ["раз", "два"])
    p.buttons([("a", "a"), ("b", "b")])
    p.buttons([("c", "c"), ("d", "d")])
    p.buttons([("e", "e"), ("f", "f")])
    p.buttons([("g", "g")], per_row=1)
    return p


def test_a_tall_panel_keeps_the_natural_sizes():
    p = game_like_panel(900)
    assert p.squeeze == 1.0 and p.top_pad == p.pad
    assert p.widgets["s0"].rect.height == ui.SLIDER_H
    assert p.widgets["t0"].rect.height == ui.TOGGLE_H
    assert p.widgets["a"].rect.height == ui.BUTTON_H
    assert p.widgets["graph"].rect.top == p.pad + 172


@pytest.mark.parametrize("height", [640, 655, 690, 720, 749, 800])
def test_at_any_window_height_nothing_overlaps_or_leaves_the_panel(height):
    import main
    p = game_like_panel(height)
    assert height >= p.least_height() or height < main.MIN_H
    boxes = [w.rect for w in p.widgets.values()]
    for i, box in enumerate(boxes):
        assert p.rect.contains(box), (height, box)
        assert box.height > 0
        for other in boxes[i + 1:]:
            assert not box.colliderect(other), (height, box, other)


def test_the_minimum_window_height_fits_the_squeezed_panel():
    import main
    assert game_like_panel(main.MIN_H).least_height() <= main.MIN_H


def test_a_shorter_panel_squeezes_everything_evenly():
    tall, short = game_like_panel(749), game_like_panel(660)
    assert short.squeeze < tall.squeeze
    for key in ("s3", "t2", "e", "graph"):
        assert short.widgets[key].rect.height <= tall.widgets[key].rect.height
    assert short.widgets["s0"].rect.height >= ui.SLIDER_LEAST


# --- подписи влезают, и со шрифтом крупнее на 10% ----------------------------------------
# Закон из крестиков-ноликов: на настоящем экране текст чуть шире, чем в offscreen, - шрифт
# на 10% крупнее изображает это «чуть шире». Здесь его меряет QFontMetrics, а не число знаков.

@pytest.mark.parametrize("scale", [1.0, 1.1])
def test_every_label_and_value_fits_its_widget(qapp, scale):
    import main
    game = main.Game(seed=0, start=False)
    f = render.font(round(15 * scale))
    space = render.text_width(f, " ")          # между подписью и значением - хотя бы пробел
    for key, w in game.ui.widgets.items():
        if isinstance(w, ui.Slider):
            widest = max(render.text_width(f, w.fmt.format(v)) for v in (w.lo, w.hi))
            assert render.text_width(f, w.label) + space + widest <= w.rect.width, key
        elif isinstance(w, ui.Toggle):
            for option in w.options:
                need = 8 + render.text_width(f, w.label) + space + render.text_width(f, str(option)) + 8
                assert need <= w.rect.width, (key, option)
        elif isinstance(w, ui.Button):
            assert render.text_width(f, w.label) + 2 * space <= w.rect.width, key


def test_a_toggle_label_gives_way_to_its_value(qapp):
    """Шрифт ещё крупнее - подпись обрезается многоточием, а не наезжает на значение."""
    big = render.font(22)
    t = ui.Toggle(Rect(0, 0, 272, 26), "скорость показа", ["без отрисовки"])
    img = render.canvas(272, 26, (0, 0, 0))
    p = render.painter(img)
    t.draw(p, big)
    p.end()
    value_left = 272 - 8 - render.text_width(big, "без отрисовки")
    dim = (render.pixels(img)[:int(value_left) - 2] == render.TEXT_DIM).all(axis=2)
    active = (render.pixels(img)[:int(value_left) - 2] == ui.ACTIVE).all(axis=2)
    assert dim.any() and not active.any()


def test_labels_are_centred_in_squeezed_rows(qapp):
    f = render.font(15)
    for h in (ui.TOGGLE_LEAST, ui.TOGGLE_H):
        top = ui._text_centre_y(Rect(0, 100, 50, h), f)
        assert 100 <= top and top + render.line_height(f) <= 100 + h + 1


def test_graph_survives_an_empty_history(qapp):
    img = render.canvas(200, 100)
    p = render.painter(img)
    ui.Graph(Rect(0, 0, 200, 100)).draw(p, [])
    p.end()


def test_graph_survives_a_single_generation(qapp):
    img = render.canvas(200, 100)
    p = render.painter(img)
    ui.Graph(Rect(0, 0, 200, 100)).draw(p, [_Stat(1.0, 1.0)])
    p.end()


def test_graph_handles_all_zero_history(qapp):
    img = render.canvas(200, 100)
    p = render.painter(img)
    ui.Graph(Rect(0, 0, 200, 100)).draw(p, [_Stat(0.0, 0.0), _Stat(0.0, 0.0)])
    p.end()


def test_graph_lines_stay_inside_the_box(qapp):
    img = render.canvas(300, 200, (255, 0, 255))
    p = render.painter(img)
    box = Rect(50, 40, 200, 60)
    ui.Graph(box).draw(p, [_Stat(v, v / 2) for v in (0.0, 10.0, 5.0, 1e6, 3.0)])
    p.end()
    painted = render.ink(img, (255, 0, 255))
    painted[box.left:box.right, box.top:box.bottom] = False
    assert not painted.any()


def test_the_widget_layer_is_redrawn_only_when_it_changes(qapp):
    p = ui.Panel(Rect(0, 0, 300, 400), render.font(15))
    p.slider("size", "размер", 10, 90, 40, integer=True)
    img = render.canvas(300, 400)
    q = render.painter(img)
    p.draw(q)
    first = p._cache
    p.draw(q)
    assert p._cache is first
    p.widgets["size"].value = 60
    p.draw(q)
    assert p._cache is not first
    second = p._cache
    p.draw(q, mouse=p.widgets["size"].rect.center)      # у ползунка подсветки нет, но ключ другой
    q.end()
    assert p._cache is not second


class _Stat:
    def __init__(self, best, mean):
        self.best = best
        self.mean = mean
