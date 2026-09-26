import numpy as np

import render
from rect import Rect


def view(w=980, h=720):
    return Rect(0, 0, w, h)


def cam(lo=(-300.0, -200.0), hi=(300.0, 200.0), rect=None):
    return render.Camera(rect or view(), np.array(lo), np.array(hi))


def test_centre_of_the_world_lands_in_the_centre_of_the_view():
    c = cam()
    x, y = c.to_screen(np.array([0.0, 0.0]))[0]
    assert (x, y) == (c.rect.centerx, c.rect.centery)


def test_the_whole_track_fits_inside_the_view():
    c = cam()
    corners = np.array([[-300.0, -200.0], [300.0, -200.0], [300.0, 200.0], [-300.0, 200.0]])
    pts = c.to_screen(corners)
    assert np.all(pts[:, 0] >= c.rect.left) and np.all(pts[:, 0] <= c.rect.right)
    assert np.all(pts[:, 1] >= c.rect.top) and np.all(pts[:, 1] <= c.rect.bottom)


def test_margin_is_respected():
    c = cam()
    pts = c.to_screen(np.array([[-300.0, 0.0], [300.0, 0.0]]))
    assert pts[0][0] >= c.rect.left + c.margin - 1e-6
    assert pts[1][0] <= c.rect.right - c.margin + 1e-6


def test_screen_y_grows_downwards():
    c = cam()
    low, high = c.to_screen(np.array([[0.0, -100.0], [0.0, 100.0]]))
    assert high[1] < low[1]


def test_scale_keeps_the_aspect_ratio():
    c = cam()
    a = c.to_screen(np.array([[0.0, 0.0], [100.0, 0.0]]))
    b = c.to_screen(np.array([[0.0, 0.0], [0.0, 100.0]]))
    assert abs(abs(a[1][0] - a[0][0]) - abs(b[1][1] - b[0][1])) < 1e-9


def test_round_trip_through_screen_and_back():
    c = cam()
    for p in np.array([[0.0, 0.0], [123.0, -45.0], [-299.0, 199.0]]):
        assert np.allclose(c.to_world(c.to_screen(p)[0]), p)


def test_a_tall_view_is_limited_by_width():
    c = cam(rect=view(400, 2000))
    assert abs(c.scale - (400 - 2 * c.margin) / 600.0) < 1e-9


def test_a_wide_view_is_limited_by_height():
    c = cam(rect=view(4000, 300))
    assert abs(c.scale - (300 - 2 * c.margin) / 400.0) < 1e-9


def test_degenerate_bounds_do_not_divide_by_zero():
    c = cam(lo=(5.0, 5.0), hi=(5.0, 5.0))
    assert np.isfinite(c.scale) and c.scale > 0.0


def test_follow_recentres_and_rescales():
    c = cam()
    c.follow(np.array([100.0, -50.0]), 3.0)
    assert c.scale == 3.0
    x, y = c.to_screen(np.array([100.0, -50.0]))[0]
    assert (x, y) == (c.rect.centerx, c.rect.centery)


def test_the_qt_matrix_is_the_same_as_to_screen(qapp):
    """Трассу QPainter ставит матрицей камеры, щелчки мышью считает to_world: разойдись они,
    машинка была бы видна не там, где по ней попадает клик."""
    from PySide6.QtCore import QPointF
    c = cam(rect=Rect(37, 11, 700, 500))
    t = c.transform()
    for p in np.array([[0.0, 0.0], [123.0, -45.0], [-299.0, 199.0]]):
        q = t.map(QPointF(*p))
        assert np.allclose([q.x(), q.y()], c.to_screen(p)[0])


def test_every_car_matrix_is_the_same_as_car_polygon(qapp):
    import car
    from PySide6.QtCore import QPointF
    veh = car.random_car(np.random.default_rng(3))
    c = cam()
    pos = np.array([[10.0, 20.0], [-150.0, 80.0]])
    angle = np.array([0.3, -2.4])
    for t, p0, a in zip(render.car_transforms(c, pos, angle), pos, angle):
        want = c.to_screen(render.car_polygon(veh.stacked, p0, a))
        got = [t.map(QPointF(*pt)) for pt in veh.stacked]
        assert np.allclose([[q.x(), q.y()] for q in got], want)


def test_car_polygon_rotates_and_moves():
    shape = np.array([[10.0, 0.0], [-5.0, 4.0], [-5.0, -4.0]])
    nose = render.car_polygon(shape, np.array([100.0, 200.0]), np.pi / 2.0)[0]
    assert np.allclose(nose, [100.0, 210.0], atol=1e-9)


def test_car_polygon_keeps_its_size():
    shape = np.array([[10.0, 0.0], [-5.0, 4.0], [-5.0, -4.0]])
    moved = render.car_polygon(shape, np.array([7.0, -3.0]), 1.234)
    before = np.linalg.norm(shape[0] - shape[1])
    after = np.linalg.norm(moved[0] - moved[1])
    assert abs(before - after) < 1e-9


def test_numpy_points_reach_the_polygon_unchanged(qapp):
    pts = np.random.default_rng(0).uniform(-500, 500, (1800, 2))
    poly = render.qpolygon(pts)
    assert poly.size() == 1800
    back = np.array([[poly.at(i).x(), poly.at(i).y()] for i in range(0, 1800, 97)])
    assert np.array_equal(back, pts[::97])
    assert render.qpolygon(np.zeros((0, 2))).size() == 0


def test_the_track_layer_is_redrawn_only_when_something_changed(qapp):
    import track
    trk = track.Track(track.circle_centerline(), 40.0)
    c = render.Camera(view(), trk.lo, trk.hi)
    layer = render.TrackLayer()
    img = render.canvas(980, 720)
    p = render.painter(img)
    layer.draw(p, c, trk)
    first = layer.pixmap
    layer.draw(p, c, trk)
    assert layer.pixmap is first
    c.rect = view(600, 400)
    c.fit(trk.lo, trk.hi)
    layer.draw(p, c, trk)
    p.end()
    assert layer.pixmap is not first


def test_fit_text_leaves_short_lines_and_trims_long_ones(qapp):
    f = render.font(15)
    assert render.fit_text(f, "коротко", 400) == "коротко"
    long = "всего 123456  доехало 123456  рекорд 98765.4с"
    cut = render.fit_text(f, long, 120)
    assert cut.endswith("…") and render.text_width(f, cut) <= 120
    # и обрезано не больше нужного: ещё один знак уже не влез бы
    assert render.text_width(f, long[:len(cut)] + "…") > 120


def telemetry_box(watch, colour=(255, 0, 255)):
    import car

    box = Rect(30, 30, render.HUD_W, render.HUD_H)
    img = render.canvas(box.width + 60, box.height + 60, colour)
    p = render.painter(img)
    render.draw_telemetry(p, box, render.font(15), car.random_car(np.random.default_rng(0)), watch)
    p.end()

    painted = render.ink(img, colour)
    outside = np.ones(painted.shape, dtype=bool)
    outside[box.left:box.right, box.top:box.bottom] = False
    return painted & outside


def watching(**over):
    base = dict(index=13, alive=True, finished=False,
                rays=np.full(7, 0.5), speed=208.0, steer=0.3, throttle=0.6, cp="2/44")
    base.update(over)
    return base


def test_the_telemetry_stays_inside_its_box(qapp):
    assert not telemetry_box(watching()).any()


def test_the_telemetry_stays_inside_for_the_longest_values(qapp):
    longest = watching(index=999999, alive=False, finished=False,
                       speed=999999.0, cp="999999/999999")
    assert not telemetry_box(longest).any()


def bottom_line_right_edge(speed):
    """Правый край нижней строки.

    Ищем ровно цвет TEXT: фон окошка и рамка другого цвета, заголовок красится
    цветом машинки, подписи - TEXT_DIM. Значит единственное, что найдётся - сама
    строка "скорость ... чекпоинт ...".
    """
    import car

    box = Rect(30, 30, render.HUD_W, render.HUD_H)
    img = render.canvas(box.width + 60, box.height + 60, render.BG)
    p = render.painter(img)
    render.draw_telemetry(p, box, render.font(15), car.random_car(np.random.default_rng(0)),
                          watching(speed=speed))
    p.end()

    text = np.all(render.pixels(img) == np.array(render.TEXT), axis=2)
    assert text.any(), "нижняя строка не нашлась"
    return int(np.argwhere(text.any(axis=1)).max())


def test_the_checkpoint_does_not_move_when_the_speed_changes(qapp):
    edges = {bottom_line_right_edge(speed) for speed in (0.0, 208.0, 99999.0)}
    assert len(edges) == 1


def test_the_car_badge_fits_beside_the_panel_text():
    import car
    import config as cfg
    import main

    longest = car.HALF_SIZE_MAX * np.sqrt(car.ASPECT_MAX)
    right = cfg.WINDOW_W - main.BADGE_INSET + longest * main.BADGE_SCALE
    assert right < cfg.WINDOW_W


def test_the_icon_is_a_car_on_a_rounded_square(qapp):
    img = render.icon_image(render.icon_car(), 64)
    assert img.width() == 64 and img.hasAlphaChannel()
    corner, middle = img.pixelColor(0, 0), img.pixelColor(32, 32)
    assert corner.alpha() == 0 and middle.alpha() == 255
    assert (middle.red(), middle.green(), middle.blue()) != render.PANEL_BG
