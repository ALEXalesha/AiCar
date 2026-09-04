import numpy as np

import config as cfg
import track


def circle(r=200.0, n=360):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def test_min_radius_of_circle():
    for r in (100.0, 200.0, 400.0):
        assert abs(track.min_radius(circle(r)) - r) < r * 0.05


def test_radius_factor_spans_easy_to_hard():
    assert track.radius_factor(0.0) == cfg.MIN_RADIUS_EASY
    assert track.radius_factor(1.0) == cfg.MIN_RADIUS_HARD
    assert track.radius_factor(2.0) == cfg.MIN_RADIUS_HARD


def test_walls_are_sane_for_wide_circle():
    assert track.walls_are_sane(circle(300.0), 90.0)


def test_walls_are_not_sane_for_tight_circle():
    assert not track.walls_are_sane(circle(30.0), 90.0)


def test_fitness_punishes_tight_corners():
    assert track.track_fitness(circle(30.0), 90.0, 0.5) < track.track_fitness(circle(300.0), 90.0, 0.5)


def test_circle_scores_the_bare_valid_base_because_variety_is_zero():
    assert abs(track.track_fitness(circle(300.0), 90.0, 0.5) - track.VALID_BASE) < 0.01


def test_any_valid_track_beats_any_invalid_one():
    invalid = [track.track_fitness(circle(r), 90.0, 0.5) for r in (10.0, 30.0, 60.0, 94.0)]
    assert max(invalid) < track.VALID_BASE


def test_tight_radius_score_grows_with_radius():
    scores = [track.track_fitness(circle(r), 90.0, 0.5) for r in (10.0, 30.0, 60.0)]
    assert scores[0] < scores[1] < scores[2]


def test_evolved_track_has_safe_corners():
    rng = np.random.default_rng(0)
    t = track.evolve_track(rng)
    assert t.min_radius >= t.width * track.radius_factor(cfg.DIFFICULTY)


def test_evolved_track_walls_do_not_fold():
    rng = np.random.default_rng(1)
    t = track.evolve_track(rng)
    assert not track.any_self_intersection(t.left)
    assert not track.any_self_intersection(t.right)


def test_evolved_track_length_is_in_range():
    rng = np.random.default_rng(2)
    t = track.evolve_track(rng)
    assert cfg.LEN_MIN <= t.length <= cfg.LEN_MAX


def test_evolved_track_is_more_varied_than_a_circle():
    rng = np.random.default_rng(3)
    t = track.evolve_track(rng)
    assert t.variety > track.Track(circle(300.0), 90.0).variety


def test_harder_difficulty_allows_tighter_corners():
    easy = [track.evolve_track(np.random.default_rng(s), difficulty=0.0).min_radius for s in (10, 11)]
    hard = [track.evolve_track(np.random.default_rng(s), difficulty=1.0).min_radius for s in (10, 11)]
    assert min(hard) < min(easy)


def test_harder_difficulty_gives_more_varied_track():
    easy = np.mean([track.evolve_track(np.random.default_rng(s), difficulty=0.0).variety for s in (10, 11)])
    hard = np.mean([track.evolve_track(np.random.default_rng(s), difficulty=1.0).variety for s in (10, 11)])
    assert hard > easy


def test_same_seed_gives_same_track():
    a = track.evolve_track(np.random.default_rng(5))
    b = track.evolve_track(np.random.default_rng(5))
    assert np.allclose(a.center, b.center)


def test_different_seeds_give_different_tracks():
    a = track.evolve_track(np.random.default_rng(6))
    b = track.evolve_track(np.random.default_rng(7))
    assert not np.allclose(a.center, b.center)


def test_impossible_width_falls_back_to_a_circle():
    t = track.evolve_track(np.random.default_rng(8), width=10000.0)
    assert np.linalg.norm(t.center, axis=1).std() < 1.0


def test_track_exposes_start_and_checkpoints():
    t = track.evolve_track(np.random.default_rng(9))
    assert t.n_checkpoints == cfg.TRACK_POINTS // cfg.CHECKPOINT_STEP
    assert np.allclose(t.start_pos, t.center[0])
    assert -np.pi <= t.start_angle <= np.pi
