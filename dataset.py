import argparse
import time

import numpy as np

import config as cfg
import cppn
import evolution
import track

DATASET_FILE = "saves/track_dataset.npz"


def normalise(radii):
    return (np.asarray(radii, dtype=float) - cfg.R_MIN) / (cfg.R_MAX - cfg.R_MIN)


def denormalise(unit):
    return cfg.R_MIN + np.asarray(unit, dtype=float) * (cfg.R_MAX - cfg.R_MIN)


def judge(radii, width, difficulty):
    center = track.centerline_from_radii(radii)
    if not (cfg.LEN_MIN <= track.polyline_length(center) <= cfg.LEN_MAX):
        return None
    if not track.walls_are_sane(center, width, difficulty):
        return None
    return track.interest(center)


def collect(candidates, rng, width=None, difficulty=None, report_every=0):
    width = cfg.TRACK_WIDTH if width is None else width
    difficulty = cfg.DIFFICULTY if difficulty is None else difficulty

    profiles, scores = [], []
    for i in range(candidates):
        radii = track.radii_from_genome(cppn.random_genome(cfg.TRACK_CPPN_LAYERS, rng))
        score = judge(radii, width, difficulty)
        if score is not None:
            profiles.append(normalise(radii))
            scores.append(score)
        if report_every and (i + 1) % report_every == 0:
            print(f"  просмотрено {i + 1}, годных {len(profiles)}")
    return np.array(profiles), np.array(scores)


def harvest_run(rng, pop_size, generations, width, difficulty, warmup=0.5):
    pop = np.stack([cppn.random_genome(cfg.TRACK_CPPN_LAYERS, rng) for _ in range(pop_size)])
    found, found_scores = [], []
    skip_until = int(generations * warmup)
    for gen in range(generations):
        radii = [track.radii_from_genome(g) for g in pop]
        lines = [track.centerline_from_radii(r) for r in radii]
        fits = np.array([track.track_fitness(c, width, difficulty) for c in lines])
        if gen >= skip_until:
            for r, c, f in zip(radii, lines, fits):
                if f >= track.VALID_BASE and cfg.LEN_MIN <= track.polyline_length(c) <= cfg.LEN_MAX:
                    found.append(normalise(r))
                    found_scores.append(track.interest(c))
        pop = evolution.evolve(pop, fits, rng)
    return found, found_scores


def harvest(runs, rng, pop_size=80, generations=12, keep_per_run=14,
            width=None, difficulty=None, min_interest=0.0, report_every=0):
    width = cfg.TRACK_WIDTH if width is None else width
    difficulty = cfg.DIFFICULTY if difficulty is None else difficulty

    profiles, scores = [], []
    for run in range(runs):
        found, found_scores = harvest_run(rng, pop_size, generations, width, difficulty)
        if found:
            order = [i for i in np.argsort(found_scores)[::-1][:keep_per_run]
                     if found_scores[i] >= min_interest]
            profiles.extend(found[i] for i in order)
            scores.extend(found_scores[i] for i in order)
        if report_every and (run + 1) % report_every == 0:
            print(f"  прогонов {run + 1}, собрано {len(profiles)}")
    return np.array(profiles), np.array(scores)


def top_slice(profiles, scores, keep):
    order = np.argsort(scores)[::-1][:keep]
    return profiles[order], scores[order]


def save(profiles, scores, path=DATASET_FILE):
    np.savez_compressed(path, profiles=profiles, scores=scores)
    return path


def load(path=DATASET_FILE):
    with np.load(path) as data:
        return data["profiles"], data["scores"]


def main():
    ap = argparse.ArgumentParser(description="Сбор датасета профилей трасс")
    ap.add_argument("--mode", choices=("harvest", "random"), default="harvest")
    ap.add_argument("--runs", type=int, default=400)
    ap.add_argument("--min-interest", type=float, default=35.0)
    ap.add_argument("--candidates", type=int, default=200000)
    ap.add_argument("--keep", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DATASET_FILE)
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    if args.mode == "harvest":
        profiles, scores = harvest(args.runs, rng, min_interest=args.min_interest,
                                   report_every=max(1, args.runs // 10))
        print(f"собрано {len(profiles)} из {args.runs} прогонов за {time.perf_counter() - t0:.0f} с")
    else:
        profiles, scores = collect(args.candidates, rng, report_every=max(1, args.candidates // 10))
        print(f"годных {len(profiles)} из {args.candidates} за {time.perf_counter() - t0:.0f} с")

    profiles, scores = top_slice(profiles, scores, args.keep)
    save(profiles, scores, args.out)
    print(f"оставлено {len(profiles)}, интерес от {scores.min():.1f} до {scores.max():.1f}, "
          f"медиана {np.median(scores):.1f}")
    print(f"записано в {args.out}")


if __name__ == "__main__":
    main()
