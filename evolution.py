import numpy as np

import config as cfg


def random_population(n, genome_size, rng, scale=1.0):
    return rng.normal(0.0, scale, (n, genome_size))


def tournament(pop, fitness, rng, k):
    idx = rng.integers(0, len(pop), k)
    return pop[idx[np.argmax(fitness[idx])]]


def crossover(a, b, rng):
    return np.where(rng.random(len(a)) < 0.5, a, b)


def mutate(genome, rng, sigma, rate):
    mask = rng.random(len(genome)) < rate
    return genome + mask * rng.normal(0.0, sigma, len(genome))


def evolve(pop, fitness, rng, elite_frac=None, mut_sigma=None,
           mut_rate=None, tournament_k=None):
    elite_frac = cfg.ELITE_FRAC if elite_frac is None else elite_frac
    mut_sigma = cfg.MUT_SIGMA if mut_sigma is None else mut_sigma
    mut_rate = cfg.MUT_RATE if mut_rate is None else mut_rate
    tournament_k = cfg.TOURNAMENT_K if tournament_k is None else tournament_k
    n, g = pop.shape
    if float(np.ptp(fitness)) < 1e-9:
        return random_population(n, g, rng, float(np.std(pop)) or 1.0)

    n_elite = max(1, int(round(n * elite_frac)))
    order = np.argsort(fitness)[::-1]
    new = np.empty_like(pop)
    new[:n_elite] = pop[order[:n_elite]]
    for i in range(n_elite, n):
        a = tournament(pop, fitness, rng, tournament_k)
        b = tournament(pop, fitness, rng, tournament_k)
        new[i] = mutate(crossover(a, b, rng), rng, mut_sigma, mut_rate)
    return new
