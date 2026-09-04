import argparse
import time
from dataclasses import dataclass

import numpy as np

import brain
import car
import config as cfg
import evolution
import field
import race
import track


@dataclass
class GenStats:
    gen: int
    best: float
    mean: float
    alive: int
    finished: int
    best_cp: int
    steps: int
    best_time: float


def gen_stats(gen, r):
    fit = r.fitness()
    arrived = r.finish_step[r.finish_step >= 0]
    return GenStats(
        gen=gen,
        best=float(fit.max()),
        mean=float(fit.mean()),
        alive=r.n_alive,
        finished=r.n_finished,
        best_cp=int(r.cp.max()),
        steps=r.steps,
        best_time=float(arrived.min()) * cfg.DT if len(arrived) else 0.0,
    )


def train(trk, fld, veh, brains, rng, generations=cfg.MAX_GENERATIONS,
          params=None, stop_on_finish=True, on_generation=None):
    params = params or {}
    history = []
    best = brains[0].copy()

    for g in range(generations):
        r = race.run(trk, fld, veh, brains)
        fit = r.fitness()
        best = brains[int(np.argmax(fit))].copy()

        stats = gen_stats(g, r)
        history.append(stats)
        if on_generation is not None:
            on_generation(stats)

        if stop_on_finish and r.n_finished:
            break
        brains = evolution.evolve(brains, fit, rng, **params)

    return brains, best, history


def build_world(rng, difficulty=cfg.DIFFICULTY, width=cfg.TRACK_WIDTH):
    trk = track.evolve_track(rng, width=width, difficulty=difficulty)
    return trk, field.build_for_track(trk), car.random_car(rng)


def main():
    ap = argparse.ArgumentParser(description="Обучение водителя без графики")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=cfg.MAX_GENERATIONS)
    ap.add_argument("--population", type=int, default=cfg.POP_SIZE)
    ap.add_argument("--difficulty", type=float, default=cfg.DIFFICULTY)
    ap.add_argument("--keep-going", action="store_true", help="не останавливаться на первом финише")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    trk, fld, veh = build_world(rng, args.difficulty)
    print(f"трасса: длина {trk.length:.0f}, самый крутой поворот {trk.min_radius:.0f}, "
          f"чекпоинтов {trk.n_checkpoints}, построена за {time.perf_counter() - t0:.1f} с")
    print(f"машинка: скорость {veh.max_speed:.0f}, руль {veh.max_steer:.2f}, "
          f"масса {veh.mass:.2f}, габариты {veh.length:.0f} на {veh.width:.0f}")
    print()

    brains = evolution.random_population(args.population, brain.genome_size(), rng)
    t0 = time.perf_counter()

    def report(s):
        line = (f"поколение {s.gen:3d}  лучший {s.best:8.1f}  средний {s.mean:8.1f}  "
                f"чекпоинтов {s.best_cp:3d}/{trk.n_checkpoints - 1}  выжило {s.alive:3d}")
        if s.finished:
            line += f"  ФИНИШ x{s.finished} за {s.best_time:.1f} с"
        print(line)

    _, _, history = train(trk, fld, veh, brains, rng, args.generations,
                          stop_on_finish=not args.keep_going, on_generation=report)

    spent = time.perf_counter() - t0
    print()
    print(f"{len(history)} поколений за {spent:.1f} с, {spent / len(history):.2f} с на поколение")
    print(f"первое поколение: {history[0].best:.1f}, последнее: {history[-1].best:.1f}")
    if history[-1].finished:
        print(f"доехало {history[-1].finished} машинок, лучшее время {history[-1].best_time:.1f} с")
    else:
        print(f"до финиша не доехал никто, лучший результат {history[-1].best_cp} чекпоинтов")


if __name__ == "__main__":
    main()
