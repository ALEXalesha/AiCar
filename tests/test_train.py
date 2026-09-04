import functools

import numpy as np

import brain
import config as cfg
import evolution
import train


@functools.lru_cache(maxsize=4)
def world(seed=3):
    return train.build_world(np.random.default_rng(seed))


def brains(n=cfg.POP_SIZE, seed=3):
    return evolution.random_population(n, brain.genome_size(), np.random.default_rng(seed))


def learn(seed=3, generations=60, **kw):
    trk, fld, veh = world(seed)
    rng = np.random.default_rng(seed)
    return trk, train.train(trk, fld, veh, brains(seed=seed), rng, generations, **kw)


def test_build_world_returns_matching_pieces():
    trk, fld, veh = world()
    assert trk.n_checkpoints > 10
    assert np.all(fld.sample(trk.center[::9]) > 0.0)
    assert veh.half_width < trk.half


def test_history_never_exceeds_the_generation_limit():
    _, (_, _, history) = learn(generations=4, stop_on_finish=False)
    assert len(history) == 4
    assert [s.gen for s in history] == [0, 1, 2, 3]


def test_training_improves_the_best_score():
    _, (_, _, history) = learn()
    assert history[-1].best > history[0].best * 2.0


def test_training_stops_at_the_first_finish():
    _, (_, _, history) = learn()
    assert history[-1].finished > 0
    assert all(s.finished == 0 for s in history[:-1])


def test_keep_going_does_not_stop_at_the_finish():
    _, (_, _, history) = learn(generations=12, stop_on_finish=False)
    assert len(history) == 12


def test_the_returned_brain_is_the_best_of_the_last_generation():
    trk, (pop, best, history) = learn()
    assert best.shape == (brain.genome_size(),)
    assert any(np.allclose(best, g) for g in pop)


def test_same_seed_gives_the_same_history():
    a = learn()[1][2]
    b = learn()[1][2]
    assert [s.best for s in a] == [s.best for s in b]


def test_callback_sees_every_generation():
    seen = []
    learn(generations=5, stop_on_finish=False, on_generation=seen.append)
    assert [s.gen for s in seen] == [0, 1, 2, 3, 4]


def test_stats_fields_are_consistent():
    trk, (_, _, history) = learn()
    for s in history:
        assert 0 <= s.alive <= cfg.POP_SIZE
        assert 0 <= s.finished <= cfg.POP_SIZE
        assert 0 <= s.best_cp <= trk.n_checkpoints - 1
        assert s.mean <= s.best
        assert 0 < s.steps <= cfg.STEPS_PER_GEN
        assert s.best_time >= 0.0


def test_finish_time_is_reported_in_seconds():
    _, (_, _, history) = learn()
    last = history[-1]
    assert 0.0 < last.best_time <= cfg.STEPS_PER_GEN * cfg.DT


def test_a_tiny_population_still_runs():
    trk, fld, veh = world()
    rng = np.random.default_rng(0)
    _, _, history = train.train(trk, fld, veh, brains(n=2), rng, generations=2, stop_on_finish=False)
    assert len(history) == 2
