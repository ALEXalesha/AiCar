"""Прямоугольник с правилами pygame (rect.py, 2.0.0).

На этих правилах написана вся логика попаданий мышью и раскладки панели: правый край -
пиксель за последним, попадание по левому и верхнему краю есть, по правому и нижнему нет.
У QRect правый край - последний пиксель, поэтому свой класс, а не QRect.
"""
import pytest

from rect import Rect


def test_right_and_bottom_are_one_past_the_last_pixel():
    r = Rect(10, 20, 30, 40)
    assert (r.left, r.top, r.width, r.height) == (10, 20, 30, 40)
    assert (r.right, r.bottom, r.centerx, r.centery) == (40, 60, 25, 40)
    assert r.center == (25, 40) and r.topleft == (10, 20) and r.size == (30, 40)


def test_collidepoint_is_half_open():
    r = Rect(0, 0, 10, 10)
    assert r.collidepoint((0, 0)) and r.collidepoint(9, 9)
    assert not r.collidepoint((10, 5)) and not r.collidepoint((5, 10))
    assert not r.collidepoint((-1, 5))


def test_collidepoint_takes_float_points():
    assert Rect(0, 0, 10, 10).collidepoint((9.5, 0.2))
    assert not Rect(0, 0, 10, 10).collidepoint((10.0, 0.2))


def test_contains_allows_touching_edges():
    outer = Rect(0, 0, 100, 100)
    assert outer.contains(Rect(0, 0, 100, 100))
    assert outer.contains(Rect(90, 90, 10, 10))
    assert not outer.contains(Rect(91, 90, 10, 10))
    assert not outer.contains(Rect(-1, 0, 10, 10))


def test_touching_edges_do_not_collide():
    a = Rect(0, 0, 10, 10)
    assert not a.colliderect(Rect(10, 0, 10, 10))
    assert not a.colliderect(Rect(0, 10, 10, 10))
    assert a.colliderect(Rect(9, 9, 10, 10))


def test_clamp_keeps_the_box_inside():
    view, box = Rect(0, 0, 980, 720), Rect(-500, 900, 252, 134)
    box.clamp_ip(view)
    assert view.contains(box) and box.topleft == (0, 720 - 134)
    box.topleft = (5000, -40)
    box.clamp_ip(view)
    assert box.topleft == (980 - 252, 0)


def test_clamp_a_box_bigger_than_the_area_centres_it():
    box = Rect(0, 0, 200, 50)
    box.clamp_ip(Rect(0, 0, 100, 100))
    assert box.centerx == 50 and box.top == 0


def test_inflate_grows_around_the_centre():
    r = Rect(100, 200, 200, 32).inflate(0, 6)
    assert (r.left, r.top, r.width, r.height) == (100, 197, 200, 38)


def test_move_returns_a_new_rect():
    r = Rect(1, 2, 3, 4)
    moved = r.move(10, 20)
    assert moved.topleft == (11, 22) and r.topleft == (1, 2)


def test_setting_topleft_and_center_moves_the_rect():
    r = Rect(0, 0, 10, 20)
    r.topleft = (5, 6)
    assert (r.left, r.top, r.right, r.bottom) == (5, 6, 15, 26)
    r.center = (100, 100)
    assert r.topleft == (95, 90)


def test_rect_unpacks_and_compares_like_a_tuple():
    r = Rect(1, 2, 3, 4)
    assert tuple(r) == (1, 2, 3, 4)
    assert r == Rect(1, 2, 3, 4) and r != Rect(1, 2, 3, 5)
    assert r.copy() == r and r.copy() is not r


def test_coordinates_are_whole_pixels():
    r = Rect(1.7, 2.2, 3.9, 4.5)
    assert all(isinstance(v, int) for v in tuple(r))


def test_qt_conversions_keep_the_pygame_edges():
    pytest.importorskip("PySide6")
    r = Rect(10, 20, 30, 40)
    q = r.qrect()
    assert (q.left(), q.top(), q.width(), q.height()) == (10, 20, 30, 40)
    f = r.qrectf()
    assert (f.left(), f.right(), f.bottom()) == (10.0, 40.0, 60.0)
