import numpy as np
import pytest

import brain
import config as cfg


def naive_forward(genome, obs, n_in, hidden, n_out):
    k = n_in * hidden
    w1 = genome[:k].reshape(n_in, hidden)
    b1 = genome[k:k + hidden]
    k += hidden
    w2 = genome[k:k + hidden * n_out].reshape(hidden, n_out)
    k += hidden * n_out
    b2 = genome[k:k + n_out]
    return np.tanh(np.tanh(obs @ w1 + b1) @ w2 + b2)


def test_genome_size_for_the_default_network():
    assert brain.genome_size() == 8 * 10 + 10 + 10 * 2 + 2
    assert brain.genome_size() == 112


def test_genome_size_follows_the_shape():
    assert brain.genome_size(4, 5, 3) == 4 * 5 + 5 + 5 * 3 + 3


def test_output_shape():
    pop = np.zeros((7, brain.genome_size()))
    assert brain.forward(pop, np.zeros((7, cfg.N_IN))).shape == (7, cfg.N_OUT)


def test_output_is_bounded():
    rng = np.random.default_rng(0)
    pop = rng.normal(0, 5.0, (50, brain.genome_size()))
    out = brain.forward(pop, rng.normal(0, 3.0, (50, cfg.N_IN)))
    assert np.all(out >= -1.0) and np.all(out <= 1.0)


def test_zero_genome_gives_zero_output():
    pop = np.zeros((3, brain.genome_size()))
    assert np.allclose(brain.forward(pop, np.ones((3, cfg.N_IN))), 0.0)


def test_matches_the_naive_per_genome_computation():
    rng = np.random.default_rng(1)
    pop = rng.normal(0, 1.0, (5, brain.genome_size()))
    obs = rng.normal(0, 1.0, (5, cfg.N_IN))
    fast = brain.forward(pop, obs)
    for i in range(5):
        slow = naive_forward(pop[i], obs[i], cfg.N_IN, cfg.HIDDEN, cfg.N_OUT)
        assert np.allclose(fast[i], slow)


def test_each_genome_uses_only_its_own_weights():
    rng = np.random.default_rng(2)
    pop = rng.normal(0, 1.0, (4, brain.genome_size()))
    obs = rng.normal(0, 1.0, (4, cfg.N_IN))
    full = brain.forward(pop, obs)
    for i in range(4):
        alone = brain.forward(pop[i:i + 1], obs[i:i + 1])
        assert np.allclose(full[i], alone[0])


def test_different_observations_give_different_actions():
    rng = np.random.default_rng(3)
    pop = rng.normal(0, 1.0, (1, brain.genome_size()))
    a = brain.forward(pop, np.zeros((1, cfg.N_IN)))
    b = brain.forward(pop, np.ones((1, cfg.N_IN)))
    assert not np.allclose(a, b)


def test_wrong_genome_length_raises():
    with pytest.raises(ValueError):
        brain.forward(np.zeros((3, 50)), np.zeros((3, cfg.N_IN)))
