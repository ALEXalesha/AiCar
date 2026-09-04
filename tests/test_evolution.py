import numpy as np

import evolution


def test_random_population_shape():
    rng = np.random.default_rng(0)
    assert evolution.random_population(20, 7, rng).shape == (20, 7)


def test_evolve_keeps_shape():
    rng = np.random.default_rng(0)
    pop = evolution.random_population(20, 7, rng)
    assert evolution.evolve(pop, rng.random(20), rng).shape == (20, 7)


def test_elite_survives_unchanged():
    rng = np.random.default_rng(0)
    pop = evolution.random_population(20, 7, rng)
    new = evolution.evolve(pop, np.arange(20.0), rng, elite_frac=0.1)
    assert np.allclose(new[0], pop[19])


def test_elite_count_follows_fraction():
    rng = np.random.default_rng(0)
    pop = evolution.random_population(50, 7, rng)
    new = evolution.evolve(pop, np.arange(50.0), rng, elite_frac=0.1, mut_rate=1.0)
    order = np.argsort(np.arange(50.0))[::-1]
    assert np.allclose(new[:5], pop[order[:5]])


def test_crossover_takes_genes_only_from_parents():
    rng = np.random.default_rng(0)
    child = evolution.crossover(np.zeros(50), np.ones(50), rng)
    assert np.all((child == 0.0) | (child == 1.0))
    assert 0 < child.sum() < 50


def test_mutate_keeps_length():
    rng = np.random.default_rng(0)
    assert len(evolution.mutate(np.zeros(100), rng, 0.1, 0.5)) == 100


def test_mutate_changes_roughly_rate_fraction():
    rng = np.random.default_rng(0)
    out = evolution.mutate(np.zeros(10000), rng, 0.1, 0.2)
    changed = np.count_nonzero(out != 0.0)
    assert 1700 < changed < 2300


def test_tournament_prefers_better():
    rng = np.random.default_rng(0)
    pop = np.arange(10.0).reshape(10, 1)
    picks = [evolution.tournament(pop, np.arange(10.0), rng, 5)[0] for _ in range(300)]
    assert np.mean(picks) > 6.0


def test_tournament_of_one_is_random():
    rng = np.random.default_rng(0)
    pop = np.arange(10.0).reshape(10, 1)
    picks = [evolution.tournament(pop, np.arange(10.0), rng, 1)[0] for _ in range(2000)]
    assert 3.8 < np.mean(picks) < 5.2


def test_flat_fitness_resets_population():
    rng = np.random.default_rng(0)
    pop = evolution.random_population(20, 7, rng)
    new = evolution.evolve(pop, np.zeros(20), rng)
    assert new.shape == pop.shape
    assert not np.allclose(new, pop)


def test_evolution_improves_toy_problem():
    rng = np.random.default_rng(1)
    pop = evolution.random_population(40, 20, rng)
    first = pop.sum(axis=1).max()
    for _ in range(40):
        pop = evolution.evolve(pop, pop.sum(axis=1), rng)
    assert pop.sum(axis=1).max() > first + 5.0
