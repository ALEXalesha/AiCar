import numpy as np
import pygame

import render


def view(w=980, h=720):
    return pygame.Rect(0, 0, w, h)


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
