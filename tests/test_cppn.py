import numpy as np
import pytest

import cppn


def test_genome_size_matches_layers():
    assert cppn.genome_size((2, 8, 1)) == 2 * 8 + 8 + 8 * 1 + 1


def test_genome_size_track_layers():
    assert cppn.genome_size((2, 8, 8, 8, 1)) == 177


def test_genome_size_car_layers():
    assert cppn.genome_size((2, 6, 6, 1)) == 67


def test_forward_output_shape():
    layers = (2, 6, 6, 1)
    g = np.zeros(cppn.genome_size(layers))
    assert cppn.forward(g, layers, np.zeros((10, 2))).shape == (10, 1)


def test_forward_output_is_bounded():
    layers = (2, 8, 8, 8, 1)
    rng = np.random.default_rng(0)
    g = rng.normal(0, 3.0, cppn.genome_size(layers))
    out = cppn.forward(g, layers, rng.normal(0, 1.0, (200, 2)))
    assert np.all(out >= -1.0) and np.all(out <= 1.0)


def test_ring_shape_is_closed():
    layers = (2, 8, 8, 8, 1)
    rng = np.random.default_rng(1)
    pts = cppn.ring_shape(cppn.random_genome(layers, rng), layers, 360, 100.0, 300.0)
    gap = np.linalg.norm(pts[0] - pts[-1])
    spacing = np.linalg.norm(pts[1] - pts[0])
    assert gap < spacing * 3.0


def test_ring_shape_radius_within_bounds():
    layers = (2, 8, 8, 8, 1)
    rng = np.random.default_rng(2)
    pts = cppn.ring_shape(cppn.random_genome(layers, rng), layers, 360, 100.0, 300.0)
    r = np.linalg.norm(pts, axis=1)
    assert r.min() >= 100.0 - 1e-6
    assert r.max() <= 300.0 + 1e-6


def test_same_genome_gives_same_shape():
    layers = (2, 6, 6, 1)
    rng = np.random.default_rng(3)
    g = cppn.random_genome(layers, rng)
    a = cppn.ring_shape(g, layers, 24, 5.0, 14.0)
    b = cppn.ring_shape(g, layers, 24, 5.0, 14.0)
    assert np.allclose(a, b)


def test_different_genomes_give_different_shapes():
    layers = (2, 6, 6, 1)
    rng = np.random.default_rng(4)
    a = cppn.ring_shape(cppn.random_genome(layers, rng), layers, 24, 5.0, 14.0)
    b = cppn.ring_shape(cppn.random_genome(layers, rng), layers, 24, 5.0, 14.0)
    assert not np.allclose(a, b)


def test_shapes_are_varied_across_seeds():
    layers = (2, 8, 8, 8, 1)
    rng = np.random.default_rng(5)
    spreads = []
    for _ in range(50):
        pts = cppn.ring_shape(cppn.random_genome(layers, rng), layers, 180, 120.0, 330.0)
        r = np.linalg.norm(pts, axis=1)
        spreads.append(r.std())
    assert np.mean(spreads) > 5.0


def test_wrong_genome_length_raises():
    with pytest.raises(ValueError):
        cppn.forward(np.zeros(5), (2, 8, 1), np.zeros((3, 2)))
