import functools

import numpy as np
import pytest

import config as cfg
import dataset
import track
import trackgen

needs_model = pytest.mark.skipif(not trackgen.available(), reason="обученной модели нет")


@functools.lru_cache(maxsize=1)
def generator():
    return trackgen.Generator()


def test_normalise_and_back():
    radii = np.linspace(cfg.R_MIN, cfg.R_MAX, 50)
    assert np.allclose(dataset.denormalise(dataset.normalise(radii)), radii)


def test_normalised_profile_lands_in_the_unit_range():
    unit = dataset.normalise(np.linspace(cfg.R_MIN, cfg.R_MAX, 50))
    assert unit.min() == 0.0 and unit.max() == 1.0


def test_judge_rejects_a_track_that_is_too_small():
    assert dataset.judge(np.full(cfg.TRACK_POINTS, cfg.R_MIN), cfg.TRACK_WIDTH, cfg.DIFFICULTY) is None


def test_judge_accepts_a_circle_of_a_legal_length():
    radii = np.full(cfg.TRACK_POINTS, cfg.LEN_MIN * 1.05 / (2.0 * np.pi))
    assert dataset.judge(radii, cfg.TRACK_WIDTH, cfg.DIFFICULTY) is not None


def test_judge_rejects_a_circle_that_is_too_short():
    radii = np.full(cfg.TRACK_POINTS, cfg.LEN_MIN * 0.7 / (2.0 * np.pi))
    assert dataset.judge(radii, cfg.TRACK_WIDTH, cfg.DIFFICULTY) is None


def test_the_fallback_circle_has_a_legal_length():
    assert cfg.LEN_MIN <= track.polyline_length(track.circle_centerline()) <= cfg.LEN_MAX


def test_harvest_returns_normalised_profiles():
    profiles, scores = dataset.harvest(2, np.random.default_rng(0), pop_size=30,
                                       generations=4, keep_per_run=5)
    assert profiles.shape[1] == cfg.TRACK_POINTS
    assert profiles.min() >= 0.0 and profiles.max() <= 1.0
    assert len(scores) == len(profiles)


def test_dataset_round_trip(tmp_path):
    path = str(tmp_path / "data.npz")
    profiles = np.random.default_rng(0).random((7, cfg.TRACK_POINTS))
    dataset.save(profiles, np.arange(7.0), path)
    back, scores = dataset.load(path)
    assert np.allclose(back, profiles) and np.allclose(scores, np.arange(7.0))


def test_top_slice_keeps_the_best():
    profiles = np.arange(20.0).reshape(5, 4)
    scores = np.array([1.0, 9.0, 3.0, 7.0, 2.0])
    kept, kept_scores = dataset.top_slice(profiles, scores, 2)
    assert list(kept_scores) == [9.0, 7.0]
    assert np.allclose(kept[0], profiles[1])


@needs_model
def test_model_loads_with_the_expected_shape():
    gen = generator()
    assert gen.latent > 0
    assert gen.n_points == cfg.TRACK_POINTS
    assert gen.layers[-1][0].shape[1] == cfg.TRACK_POINTS


@needs_model
def test_decoder_output_is_a_unit_profile():
    out = generator().decode(np.zeros(generator().latent))
    assert out.shape == (1, cfg.TRACK_POINTS)
    assert out.min() >= 0.0 and out.max() <= 1.0


@needs_model
def test_sampled_radii_stay_inside_the_configured_range():
    gen, rng = generator(), np.random.default_rng(0)
    for _ in range(40):
        radii = gen.sample_radii(rng)
        assert radii.min() >= cfg.R_MIN - 1e-6
        assert radii.max() <= cfg.R_MAX + 1e-6


@needs_model
def test_made_track_is_valid():
    t = generator().make_track(np.random.default_rng(0))
    assert t.min_radius >= t.width * track.radius_factor(cfg.DIFFICULTY)
    assert not track.any_self_intersection(t.left)
    assert cfg.LEN_MIN <= t.length <= cfg.LEN_MAX


@needs_model
def test_made_tracks_are_as_interesting_as_evolved_ones():
    gen = generator()
    rng = np.random.default_rng(0)
    scores = [gen.make_track(rng).variety for _ in range(4)]
    assert np.median(scores) > 40.0


@needs_model
def test_same_seed_gives_the_same_track():
    a = generator().make_track(np.random.default_rng(5))
    b = generator().make_track(np.random.default_rng(5))
    assert np.allclose(a.center, b.center)


@needs_model
def test_different_seeds_give_different_tracks():
    a = generator().make_track(np.random.default_rng(6))
    b = generator().make_track(np.random.default_rng(7))
    assert not np.allclose(a.center, b.center)


@needs_model
def test_impossible_width_falls_back_to_a_circle():
    t = generator().make_track(np.random.default_rng(0), width=5000.0, attempts=5)
    assert np.linalg.norm(t.center, axis=1).std() < 1.0


def test_a_missing_model_is_reported_not_raised(tmp_path):
    assert not trackgen.available(str(tmp_path / "absent.npz"))
