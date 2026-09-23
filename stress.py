import argparse
import os
import sys
import tempfile
import time
import traceback

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
# Звук тоже заглушкой и тоже сразу. Раньше он ставился лениво, в `the_game`, а
# свойства панели ещё до того зовут `pygame.init()` - он поднимает все
# подсистемы, звук в том числе. На сервере сборки звуковой карты нет, и круг
# вставал там на двенадцать минут, пока его не снимал предел шага.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

import brain
import car
import config as cfg
import cppn
import dataset
import evolution
import field
import race
import sensors
import sound
import stats
import track
import trackgen

# Проверка гоняет настоящую игру, а та пишет мозги и статистику. Уводим запись
# во временную папку: испортить сохранения игрока прогоном проверки недопустимо.
SANDBOX = stats.use_sandbox("aicar-stress-")

FAST, SLOW, HEAVY = "fast", "slow", "heavy"
CHECKS = []


def check(name, cost=FAST):
    def keep(fn):
        CHECKS.append((name, fn, cost))
        return fn
    return keep


def finite(*arrays):
    for a in arrays:
        assert np.all(np.isfinite(a)), "нечисло или бесконечность"


# ---------------------------------------------------------------- cppn

@check("cppn: радиус внутри заданных границ")
def _(rng):
    lo, hi = sorted(rng.uniform(20.0, 400.0, 2))
    g = cppn.random_genome(cfg.TRACK_CPPN_LAYERS, rng, rng.uniform(0.1, 3.0))
    r = cppn.ring_radii(g, cfg.TRACK_CPPN_LAYERS, 120, lo, hi)
    finite(r)
    assert r.min() >= lo - 1e-9 and r.max() <= hi + 1e-9


@check("cppn: полный оборот возвращает тот же радиус")
def _(rng):
    layers = cfg.TRACK_CPPN_LAYERS
    g = cppn.random_genome(layers, rng, rng.uniform(0.1, 3.0))
    theta = rng.uniform(-20.0, 20.0, 40)
    a = cppn.forward(g, layers, cppn.angle_features(theta, layers[0]))
    b = cppn.forward(g, layers, cppn.angle_features(theta + 2.0 * np.pi, layers[0]))
    assert np.allclose(a, b, atol=1e-9)


@check("cppn: стык контура не длиннее обычного шага")
def _(rng):
    n = int(rng.integers(60, 400))
    g = cppn.random_genome(cfg.TRACK_CPPN_LAYERS, rng)
    pts = cppn.ring_shape(g, cfg.TRACK_CPPN_LAYERS, n, cfg.R_MIN, cfg.R_MAX)
    steps = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    assert steps[-1] <= steps.max()


@check("cppn: один геном даёт одну форму")
def _(rng):
    g = cppn.random_genome(cfg.CAR_CPPN_LAYERS, rng)
    a = cppn.ring_shape(g, cfg.CAR_CPPN_LAYERS, 40, 5.0, 14.0)
    b = cppn.ring_shape(g, cfg.CAR_CPPN_LAYERS, 40, 5.0, 14.0)
    assert np.array_equal(a, b)


@check("cppn: выход ограничен единицей")
def _(rng):
    layers = cfg.TRACK_CPPN_LAYERS
    g = rng.normal(0.0, rng.uniform(0.1, 8.0), cppn.genome_size(layers))
    out = cppn.forward(g, layers, rng.normal(0.0, 5.0, (40, layers[0])))
    finite(out)
    assert out.min() >= -1.0 and out.max() <= 1.0


@check("cppn: гармоники периодичны")
def _(rng):
    n = int(rng.integers(1, 4)) * 2
    theta = rng.uniform(-10.0, 10.0, 30)
    a = cppn.angle_features(theta, n)
    b = cppn.angle_features(theta + 2.0 * np.pi, n)
    assert np.allclose(a, b, atol=1e-9)


@check("cppn: чужая длина генома отвергается")
def _(rng):
    layers = cfg.TRACK_CPPN_LAYERS
    wrong = cppn.genome_size(layers) + int(rng.integers(1, 40))
    try:
        cppn.forward(np.zeros(wrong), layers, np.zeros((3, layers[0])))
    except ValueError:
        return
    raise AssertionError("геном чужой длины прошёл без ошибки")


# ---------------------------------------------------------------- evolution

@check("эволюция: форма популяции сохраняется")
def _(rng):
    n, g = int(rng.integers(2, 80)), int(rng.integers(1, 200))
    pop = evolution.random_population(n, g, rng)
    new = evolution.evolve(pop, rng.random(n) * rng.uniform(0.1, 1000.0), rng)
    assert new.shape == (n, g)
    finite(new)


@check("эволюция: элита переходит без изменений")
def _(rng):
    n, g = int(rng.integers(10, 60)), int(rng.integers(2, 50))
    pop = evolution.random_population(n, g, rng)
    fit = rng.permutation(n).astype(float)
    frac = float(rng.uniform(0.02, 0.4))
    new = evolution.evolve(pop, fit, rng, elite_frac=frac)
    keep = max(1, int(round(n * frac)))
    order = np.argsort(fit)[::-1]
    assert np.array_equal(new[:keep], pop[order[:keep]])


@check("эволюция: скрещивание не выдумывает генов")
def _(rng):
    g = int(rng.integers(1, 200))
    a, b = rng.normal(size=g), rng.normal(size=g)
    child = evolution.crossover(a, b, rng)
    assert np.all((child == a) | (child == b))


@check("эволюция: мутация меняет примерно заданную долю")
def _(rng):
    g = 4000
    rate = float(rng.uniform(0.05, 0.9))
    out = evolution.mutate(np.zeros(g), rng, 0.2, rate)
    share = np.count_nonzero(out) / g
    assert abs(share - rate) < 0.05


@check("эволюция: турнир возвращает существующую особь")
def _(rng):
    n, g = int(rng.integers(2, 40)), int(rng.integers(1, 20))
    pop = evolution.random_population(n, g, rng)
    fit = rng.random(n)
    pick = evolution.tournament(pop, fit, rng, int(rng.integers(1, n + 1)))
    assert any(np.array_equal(pick, row) for row in pop)


@check("эволюция: одинаковые оценки пересоздают популяцию")
def _(rng):
    n, g = int(rng.integers(4, 40)), int(rng.integers(2, 40))
    pop = evolution.random_population(n, g, rng)
    new = evolution.evolve(pop, np.full(n, rng.normal()), rng)
    assert new.shape == pop.shape and not np.array_equal(new, pop)


# ---------------------------------------------------------------- геометрия

def random_line(rng, points=360):
    g = cppn.random_genome(cfg.TRACK_CPPN_LAYERS, rng, rng.uniform(0.2, 1.2))
    return track.centerline(g, points)


@check("трасса: касательные единичной длины")
def _(rng):
    t = track.tangents(random_line(rng))
    finite(t)
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-9)


@check("трасса: нормаль перпендикулярна касательной")
def _(rng):
    line = random_line(rng)
    dot = np.sum(track.tangents(line) * track.normals(line), axis=1)
    assert np.allclose(dot, 0.0, atol=1e-9)


@check("трасса: стены отстоят ровно на половину ширины")
def _(rng):
    line = random_line(rng)
    half = float(rng.uniform(5.0, 90.0))
    left, right = track.offset_walls(line, half)
    assert np.allclose(np.linalg.norm(left - line, axis=1), half, atol=1e-9)
    assert np.allclose(np.linalg.norm(right - line, axis=1), half, atol=1e-9)
    assert np.allclose(left + right, 2.0 * line, atol=1e-9)


@check("трасса: кривизна неотрицательна и конечна")
def _(rng):
    k = track.curvature(random_line(rng))
    finite(k)
    assert k.min() >= 0.0


@check("трасса: длина положительна и совпадает с суммой шагов")
def _(rng):
    line = random_line(rng)
    steps = np.linalg.norm(np.roll(line, -1, axis=0) - line, axis=1)
    assert abs(track.polyline_length(line) - steps.sum()) < 1e-6
    assert track.polyline_length(line) > 0.0


@check("трасса: сглаживание не меняет число точек")
def _(rng):
    line = random_line(rng, int(rng.integers(60, 400)))
    w = int(rng.integers(1, 40))
    assert track.smooth_closed(line, w).shape == line.shape


@check("трасса: выпуклая кривая без самопересечений")
def _(rng):
    n = int(rng.integers(30, 200))
    r = float(rng.uniform(20.0, 500.0))
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    circle = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    assert not track.any_self_intersection(circle, step=max(1, n // 60))


@check("трасса: чекпоинтов столько, сколько шагов помещается")
def _(rng):
    line = random_line(rng)
    step = int(rng.integers(2, 40))
    left, right = track.offset_walls(line, 20.0)
    cps = track.build_checkpoints(left, right, step)
    assert len(cps) == len(range(0, len(line), step))


@check("трасса: радиус круга восстанавливается из кривизны")
def _(rng):
    r = float(rng.uniform(40.0, 600.0))
    t = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    circle = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    assert abs(track.min_radius(circle) - r) < r * 0.05


@check("трасса: годная оценка всегда выше бракованной")
def _(rng):
    width = float(rng.uniform(20.0, 80.0))
    difficulty = float(rng.uniform(0.0, 1.0))
    tight = float(rng.uniform(3.0, width * track.radius_factor(difficulty) * 0.9))
    t = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    bad = np.stack([tight * np.cos(t), tight * np.sin(t)], axis=1)
    assert track.track_fitness(bad, width, difficulty) < track.VALID_BASE


@check("трасса: эволюция даёт проходимую трассу", SLOW)
def _(rng):
    width = float(rng.uniform(28.0, 60.0))
    difficulty = float(rng.uniform(0.0, 1.0))
    t = track.evolve_track(rng, width, difficulty)
    assert t.min_radius >= width * track.radius_factor(difficulty) - 1e-6
    assert not track.any_self_intersection(t.left)
    assert not track.any_self_intersection(t.right)
    assert cfg.LEN_MIN <= t.length <= cfg.LEN_MAX
    finite(t.center, t.left, t.right, t.cp_mid, t.cp_dir)


@check("трасса: поля объекта согласованы", SLOW)
def _(rng):
    t = track.evolve_track(rng, float(rng.uniform(28.0, 60.0)))
    assert np.all(t.lo < t.hi)
    assert t.n_checkpoints == len(t.cp_mid) == len(t.cp_dir) == len(t.checkpoints)
    assert np.allclose(np.linalg.norm(t.cp_dir, axis=1), 1.0, atol=1e-9)
    assert np.allclose(t.checkpoints.mean(axis=1), t.cp_mid, atol=1e-9)


# ---------------------------------------------------------------- поле

@check("поле: на центральной линии значение равно полуширине", SLOW)
def _(rng):
    line = random_line(rng)
    half = float(rng.uniform(10.0, 45.0))
    f = field.build(line, half)
    values = f.sample(line[::17])
    assert np.all(np.abs(values - half) < cfg.FIELD_CELL * 1.5)


@check("поле: вне сетки читается как далеко снаружи")
def _(rng):
    f = field.build(random_line(rng), 20.0, cell=8.0)
    far = rng.uniform(1e5, 1e7, (5, 2)) * rng.choice([-1.0, 1.0], (5, 2))
    assert np.all(f.sample(far) == field.OUTSIDE)


@check("поле: форма результата теряет только последнюю ось")
def _(rng):
    f = field.build(random_line(rng), 20.0, cell=10.0)
    shape = tuple(int(rng.integers(1, 6)) for _ in range(int(rng.integers(1, 4))))
    assert f.sample(np.zeros(shape + (2,))).shape == shape


@check("поле: значения конечны")
def _(rng):
    f = field.build(random_line(rng), float(rng.uniform(10.0, 45.0)), cell=10.0)
    finite(f.values)


# ---------------------------------------------------------------- датчики

def small_world(rng):
    trk = track.Track(random_line(rng), float(rng.uniform(28.0, 50.0)))
    return trk, field.build_for_track(trk)


@check("датчики: показания в допустимом диапазоне", SLOW)
def _(rng):
    trk, fld = small_world(rng)
    n = int(rng.integers(1, 30))
    idx = rng.integers(0, len(trk.center), n)
    rays = sensors.cast(trk.center[idx], rng.uniform(-np.pi, np.pi, n), fld)
    finite(rays)
    assert rays.shape == (n, cfg.N_RAYS)
    assert np.all(rays > 0.0) and np.all(rays <= cfg.RAY_MAX)
    assert np.all(rays % cfg.RAY_STEP == 0.0)


@check("датчики: конец луча за стеной, шаг до него внутри", SLOW)
def _(rng):
    trk, fld = small_world(rng)
    n = 8
    idx = rng.integers(0, len(trk.center), n)
    pos, ang = trk.center[idx], rng.uniform(-np.pi, np.pi, n)
    rays = sensors.cast(pos, ang, fld)
    for i in range(n):
        for j, dist in enumerate(rays[i]):
            if dist >= cfg.RAY_MAX:
                continue
            direction = np.array([np.cos(ang[i] + cfg.RAY_ANGLES[j]),
                                  np.sin(ang[i] + cfg.RAY_ANGLES[j])])
            assert fld.sample(pos[i] + direction * dist) < 0.0
            assert fld.sample(pos[i] + direction * (dist - cfg.RAY_STEP)) >= 0.0


@check("датчики: поворот машинки поворачивает показания", SLOW)
def _(rng):
    trk, fld = small_world(rng)
    i = int(rng.integers(0, len(trk.center)))
    base = float(rng.uniform(-np.pi, np.pi))
    pos = trk.center[i:i + 1]
    turn = float(cfg.RAY_ANGLES[1] - cfg.RAY_ANGLES[0])
    a = sensors.cast(pos, np.array([base]), fld)[0]
    b = sensors.cast(pos, np.array([base + turn]), fld)[0]
    assert np.allclose(a[1:], b[:-1], atol=cfg.RAY_STEP * 1.01)


# ---------------------------------------------------------------- машинка

@check("машинка: параметры в объявленных пределах")
def _(rng):
    c = car.random_car(rng)
    assert car.MASS_MIN <= c.mass <= car.MASS_MAX
    assert car.SPEED_BASE <= c.max_speed <= car.SPEED_BASE + car.SPEED_GAIN
    assert car.STEER_BASE - car.STEER_DROP <= c.max_steer <= car.STEER_BASE
    assert c.length > c.width > 0.0 and c.area > 0.0
    assert abs(c.power - c.mass * c.accel) < 1e-9


@check("машинка: помещается в трассу")
def _(rng):
    assert car.random_car(rng).half_width < cfg.TRACK_WIDTH * 0.5


@check("машинка: детали конечны и не пусты")
def _(rng):
    c = car.random_car(rng)
    assert len(c.parts) == len(c.kinds) == len(c.slices)
    for poly, kind in c.parts:
        finite(poly)
        assert len(poly) >= 3, kind
    assert len(c.stacked) == sum(len(p) for p, _ in c.parts)


@check("машинка: колёса выступают за борт, кабина внутри")
def _(rng):
    c = car.random_car(rng)
    assert max(float(np.abs(w[:, 1]).max()) for w in c.wheels) > c.half_width
    assert np.abs(c.cockpit[:, 1]).max() < c.half_width
    assert np.abs(c.cockpit[:, 0]).max() < c.length * 0.5


@check("машинка: один геном даёт одну машину")
def _(rng):
    g = cppn.random_genome(cfg.CAR_CPPN_LAYERS, rng, cfg.CAR_INIT_SCALE)
    a, b = car.generate(g), car.generate(g)
    assert np.array_equal(a.stacked, b.stacked) and a.color == b.color


@check("машинка: цвет корректный")
def _(rng):
    for value in car.random_car(rng).color:
        assert 0 <= value <= 255


# ---------------------------------------------------------------- сеть

@check("сеть: выход ограничен и конечен")
def _(rng):
    n = int(rng.integers(1, 80))
    pop = rng.normal(0.0, rng.uniform(0.1, 20.0), (n, brain.genome_size()))
    obs = rng.normal(0.0, rng.uniform(0.1, 10.0), (n, cfg.N_IN))
    out = brain.forward(pop, obs)
    finite(out)
    assert out.shape == (n, cfg.N_OUT)
    assert out.min() >= -1.0 and out.max() <= 1.0


@check("сеть: каждый мозг считается по своим весам")
def _(rng):
    n = int(rng.integers(2, 12))
    pop = rng.normal(size=(n, brain.genome_size()))
    obs = rng.normal(size=(n, cfg.N_IN))
    full = brain.forward(pop, obs)
    i = int(rng.integers(0, n))
    alone = brain.forward(pop[i:i + 1], obs[i:i + 1])
    assert np.allclose(full[i], alone[0])


@check("сеть: нулевые веса дают нулевое управление")
def _(rng):
    n = int(rng.integers(1, 20))
    out = brain.forward(np.zeros((n, brain.genome_size())), rng.normal(size=(n, cfg.N_IN)))
    assert np.allclose(out, 0.0)


# ---------------------------------------------------------------- заезд

def small_race(rng, n=None):
    trk, fld = small_world(rng)
    veh = car.random_car(rng)
    n = n or int(rng.integers(2, 20))
    brains = rng.normal(0.0, rng.uniform(0.2, 3.0), (n, brain.genome_size()))
    return race.Race(trk, fld, veh, brains)


@check("заезд: скорость в пределах машины", SLOW)
def _(rng):
    r = small_race(rng)
    for _ in range(int(rng.integers(20, 200))):
        r.step()
    finite(r.pos, r.angle, r.speed)
    assert np.all(r.speed >= 0.0) and np.all(r.speed <= r.car.max_speed + 1e-9)


@check("заезд: мёртвые не двигаются", SLOW)
def _(rng):
    r = small_race(rng)
    for _ in range(60):
        r.step()
    dead = ~r.alive
    if not dead.any():
        return
    before = r.pos[dead].copy()
    for _ in range(20):
        r.step()
    assert np.allclose(r.pos[dead], before)


@check("заезд: чекпоинты только растут, живых не прибавляется", SLOW)
def _(rng):
    r = small_race(rng)
    cp, alive = r.cp.copy(), r.n_alive
    for _ in range(int(rng.integers(30, 300))):
        r.step()
        assert np.all(r.cp >= cp)
        assert r.n_alive <= alive
        cp, alive = r.cp.copy(), r.n_alive


@check("заезд: приспособленность конечна и растёт с чекпоинтами", SLOW)
def _(rng):
    r = small_race(rng)
    for _ in range(120):
        r.step()
    fit = r.fitness()
    finite(fit)
    assert np.all(fit >= 0.0)
    order = np.argsort(r.cp)
    grouped = [fit[r.cp == value].max() for value in np.unique(r.cp)]
    assert grouped == sorted(grouped)


@check("заезд: номер чекпоинта не выходит за границы", SLOW)
def _(rng):
    r = small_race(rng)
    for _ in range(200):
        r.step()
    assert np.all(r.cp >= 0) and np.all(r.cp <= r.last_cp)


@check("заезд: лидер жив, пока хоть кто-то едет", SLOW)
def _(rng):
    r = small_race(rng)
    for _ in range(int(rng.integers(20, 400))):
        r.step()
        leader = r.leader
        assert 0 <= leader < r.n
        if r.alive.any():
            assert r.alive[leader], "лидером стали обломки, хотя кто-то ещё едет"
            best = r.fitness()[r.alive].max()
            assert r.fitness()[leader] >= best - 1e-9, "лидер не лучший из живых"
        if r.done:
            break


@check("заезд: слежение всегда даёт существующую машинку", SLOW)
def _(rng):
    r = small_race(rng)
    for _ in range(int(rng.integers(10, 150))):
        r.step()
    for target in [None, -1, r.n, r.n + 5] + list(range(r.n)):
        assert 0 <= r.watch(target) < r.n


@check("заезд: наблюдения нормированы", SLOW)
def _(rng):
    r = small_race(rng)
    for _ in range(50):
        r.step()
    obs = r.observe()
    finite(obs)
    assert obs.shape == (r.n, cfg.N_IN)
    assert obs.min() >= 0.0 and obs.max() <= 1.0 + 1e-9


@check("заезд: прогон до конца укладывается в лимит", SLOW)
def _(rng):
    r = race.run(*small_world(rng)[:2], car.random_car(rng),
                 rng.normal(size=(int(rng.integers(2, 12)), brain.genome_size())))
    assert r.done and r.steps <= cfg.STEPS_PER_GEN


# ---------------------------------------------------------------- сохранения

@check("сохранения: мозги переживают запись и чтение")
def _(rng):
    import tempfile
    n = int(rng.integers(1, 40))
    g = rng.normal(size=(n, brain.genome_size()))
    fit = rng.random(n)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "brains.npz")
        stats.save_brains(g, fit, path)
        back, back_fit = stats.load_brains(path)
        assert np.allclose(back, g) and np.allclose(back_fit, fit)


@check("сохранения: чужой размер генома отвергается")
def _(rng):
    import tempfile
    size = brain.genome_size() + int(rng.integers(1, 30))
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "brains.npz")
        stats.save_brains(rng.normal(size=(3, size)), np.zeros(3), path)
        assert stats.load_brains(path) is None


@check("сохранения: счётчики не портятся и не мутируют исходник")
def _(rng):
    class Gen:
        def __init__(self, finished, best_time):
            self.finished, self.best_time = finished, best_time

    totals = dict(stats.EMPTY)
    for _ in range(int(rng.integers(1, 12))):
        copy = dict(totals)
        history = [Gen(int(rng.integers(0, 3)), float(rng.uniform(1.0, 40.0)))
                   for _ in range(int(rng.integers(1, 6)))]
        totals = stats.record_round(totals, history)
        assert copy == {k: v for k, v in copy.items()}
    assert totals["rounds"] >= 1
    assert totals["finished"] <= totals["rounds"]


# ---------------------------------------------------------------- звук

@check("звук: сэмплы не выходят за int16 и не пустые")
def _(rng):
    freq = float(rng.uniform(40.0, 400.0))
    s = sound.engine_sample(freq, rate=int(rng.choice([22050, 44100, 48000])))
    assert s.dtype == np.int16 and len(s) > 0
    assert np.abs(s).max() <= 32767


@check("звук: в сэмпле целое число периодов")
def _(rng):
    rate = int(rng.choice([22050, 44100, 48000]))
    freq = float(rng.uniform(40.0, 400.0))
    period = int(round(rate / freq))
    assert len(sound.engine_sample(freq, rate=rate)) % period == 0


@check("звук: моно разворачивается в нужное число каналов")
def _(rng):
    n, ch = int(rng.integers(1, 500)), int(rng.integers(1, 5))
    mono = rng.integers(-32000, 32000, n).astype(np.int16)
    out = sound.fit_channels(mono, ch)
    assert out.shape == ((n,) if ch <= 1 else (n, ch))


# ---------------------------------------------------------------- генератор

@check("генератор: сэмпл модели даёт радиусы в пределах")
def _(rng):
    if not trackgen.available():
        return
    radii = trackgen.Generator().sample_radii(rng, float(rng.uniform(0.3, 2.0)))
    finite(radii)
    assert radii.min() >= cfg.R_MIN - 1e-6 and radii.max() <= cfg.R_MAX + 1e-6


@check("генератор: нормировка обратима")
def _(rng):
    radii = rng.uniform(cfg.R_MIN, cfg.R_MAX, int(rng.integers(1, 400)))
    assert np.allclose(dataset.denormalise(dataset.normalise(radii)), radii)


@check("генератор: модель делает проходимую трассу", SLOW)
def _(rng):
    if not trackgen.available():
        return
    width = float(rng.uniform(28.0, 60.0))
    difficulty = float(rng.uniform(0.0, 1.0))
    t = trackgen.Generator().make_track(rng, width, difficulty)
    assert t.min_radius >= width * track.radius_factor(difficulty) - 1e-6
    assert not track.any_self_intersection(t.left)
    finite(t.center)


# ---------------------------------------------------------------- панель

@check("панель: ползунок не выходит за свои пределы")
def _(rng):
    import pygame
    import ui
    pygame.init()
    lo, hi = sorted(rng.uniform(-500.0, 500.0, 2))
    if hi - lo < 1e-6:
        return
    box = pygame.Rect(int(rng.integers(0, 400)), int(rng.integers(0, 400)), 200, ui.SLIDER_H)
    bar = ui.Slider(box, "тест", lo, hi, rng.uniform(lo, hi))
    for _ in range(6):
        bar.value = rng.uniform(-1e6, 1e6)
        assert lo - 1e-9 <= bar.value <= hi + 1e-9


@check("панель: клик по полосе задаёт значение по положению")
def _(rng):
    import pygame
    import ui
    pygame.init()
    bar = ui.Slider(pygame.Rect(100, 200, 200, ui.SLIDER_H), "тест", 0.0, 100.0, 50.0)
    share = float(rng.uniform(0.0, 1.0))
    x = bar.bar.left + int(share * (bar.bar.width - 1))
    bar.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, bar.bar.centery), button=1))
    assert abs(bar.value - share * 100.0) < 2.0


@check("панель: переключатель возвращается на круг")
def _(rng):
    import pygame
    import ui
    pygame.init()
    options = [f"пункт {i}" for i in range(int(rng.integers(2, 8)))]
    toggle = ui.Toggle(pygame.Rect(0, 0, 100, ui.TOGGLE_H), "тест", options)
    for _ in range(len(options)):
        toggle.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(50, 10), button=1))
    assert toggle.value == options[0]


@check("панель: кнопка не срабатывает при отпускании снаружи")
def _(rng):
    import pygame
    import ui
    pygame.init()
    b = ui.Button(pygame.Rect(0, 0, 100, ui.BUTTON_H), "тест")
    b.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(50, 10), button=1))
    away = (int(rng.integers(200, 900)), int(rng.integers(200, 900)))
    b.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=away, button=1))
    assert not b.take()


@check("панель: виджеты не наезжают друг на друга и не вылезают")
def _(rng):
    import pygame
    import ui
    pygame.init()
    panel = ui.Panel(pygame.Rect(int(rng.integers(0, 600)), 0, 300, 720), None)
    panel.skip(int(rng.integers(0, 200)))
    for i in range(int(rng.integers(1, 6))):
        panel.slider(f"s{i}", "п", 0, 1, 0.5)
    for i in range(int(rng.integers(1, 4))):
        panel.toggle(f"t{i}", "т", ["раз", "два"])
    panel.buttons([(f"b{i}", "к") for i in range(int(rng.integers(1, 5)))],
                  per_row=int(rng.integers(1, 4)))
    boxes = [w.rect for w in panel.widgets.values()]
    for i, box in enumerate(boxes):
        assert panel.rect.contains(box)
        for other in boxes[i + 1:]:
            assert not box.colliderect(other)


# ---------------------------------------------------------------- камера

@check("камера: вся область помещается в кадр")
def _(rng):
    import pygame
    import render
    lo = rng.uniform(-2000.0, 2000.0, 2)
    hi = lo + rng.uniform(10.0, 3000.0, 2)
    view = pygame.Rect(0, 0, int(rng.integers(200, 1400)), int(rng.integers(200, 1000)))
    cam = render.Camera(view, lo, hi)
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    pts = cam.to_screen(corners)
    assert np.all(pts[:, 0] >= view.left - 1e-6) and np.all(pts[:, 0] <= view.right + 1e-6)
    assert np.all(pts[:, 1] >= view.top - 1e-6) and np.all(pts[:, 1] <= view.bottom + 1e-6)


@check("камера: перевод туда и обратно возвращает точку")
def _(rng):
    import pygame
    import render
    lo = rng.uniform(-1000.0, 1000.0, 2)
    hi = lo + rng.uniform(50.0, 2000.0, 2)
    cam = render.Camera(pygame.Rect(0, 0, 980, 720), lo, hi)
    point = rng.uniform(lo, hi)
    assert np.allclose(cam.to_world(cam.to_screen(point)[0]), point, atol=1e-6)


@check("камера: масштаб одинаков по обеим осям")
def _(rng):
    import pygame
    import render
    lo = rng.uniform(-500.0, 500.0, 2)
    hi = lo + rng.uniform(50.0, 1500.0, 2)
    cam = render.Camera(pygame.Rect(0, 0, 980, 720), lo, hi)
    step = float(rng.uniform(1.0, 100.0))
    dx = cam.to_screen(np.array([[0.0, 0.0], [step, 0.0]]))
    dy = cam.to_screen(np.array([[0.0, 0.0], [0.0, step]]))
    assert abs(abs(dx[1][0] - dx[0][0]) - abs(dy[1][1] - dy[0][1])) < 1e-6


@check("отрисовка: поворот кузова не меняет размеров")
def _(rng):
    import render
    veh = car.random_car(rng)
    angle = float(rng.uniform(-20.0, 20.0))
    pos = rng.uniform(-500.0, 500.0, 2)
    moved = render.car_polygon(veh.stacked, pos, angle)
    before = np.linalg.norm(veh.stacked[0] - veh.stacked[1])
    after = np.linalg.norm(moved[0] - moved[1])
    assert abs(before - after) < 1e-9
    finite(moved)


@check("отрисовка: поворот на полный оборот ничего не меняет")
def _(rng):
    import render
    veh = car.random_car(rng)
    pos = rng.uniform(-100.0, 100.0, 2)
    a = render.car_polygon(veh.stacked, pos, 0.0)
    b = render.car_polygon(veh.stacked, pos, 2.0 * np.pi)
    assert np.allclose(a, b, atol=1e-9)


@check("отрисовка: детали машинки склеены в один массив")
def _(rng):
    veh = car.random_car(rng)
    joined = np.vstack([poly for poly, _ in veh.parts])
    assert np.array_equal(veh.stacked, joined)
    for (start, end), (poly, _) in zip(veh.slices, veh.parts):
        assert np.array_equal(veh.stacked[start:end], poly)


# ---------------------------------------------------------------- слой отрисовки

def canvas(ground, w=cfg.WINDOW_W, h=cfg.WINDOW_H):
    import pygame
    pygame.init()
    surf = pygame.Surface((w, h))
    surf.fill(ground)
    return surf


def ink(surf, ground):
    """Маска (ширина, высота): True там, где что-то нарисовали поверх фона."""
    import pygame
    return np.any(pygame.surfarray.array3d(surf) != np.array(ground), axis=2)


def field_view():
    import pygame
    return pygame.Rect(0, 0, cfg.WINDOW_W - cfg.PANEL_W, cfg.WINDOW_H)


def edge_positions(trk):
    """Четыре точки на внешней стене, самые дальние по каждой оси."""
    spots = []
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        k = int(np.argmax(sign * trk.center[:, axis]))
        offset = np.zeros(2)
        offset[axis] = sign * trk.half
        spots.append(trk.center[k] + offset)
    return np.array(spots)


@check("отрисовка: поле не залезает на боковую панель", SLOW)
def _(rng):
    import render
    # Через саму игру, а не через свою копию draw_field: проверять надо тот код,
    # который выполняется, иначе пропадёт обрезка в main и никто не заметит.
    game = the_game()
    # Ставим машинки вплотную к правой границе поля. В настоящем заезде они туда
    # не заедут, но проверяется не заезд, а сама защита: обрезка в draw_field.
    # Без неё кузов пересекает границу и красит панель.
    game.camera.fit(game.track.lo, game.track.hi)
    edge = [game.camera.to_world((game.view.right - k, y))
            for k in (2, 10, 20) for y in (100, game.view.centery, game.view.bottom - 100)]
    game.race.pos[:len(edge)] = np.array(edge)
    game.race.angle[:] = rng.uniform(-np.pi, np.pi, game.race.n)
    game.race.alive[:] = rng.integers(0, 2, game.race.n).astype(bool)

    game.screen.fill(render.BG)
    game.draw_field()
    assert not ink(game.screen, render.BG)[game.panel_rect.left:, :].any(), "краска на панели"


@check("отрисовка: стены трассы помещаются в кадр", SLOW)
def _(rng):
    import render
    trk, _ = small_world(rng)
    view = field_view()
    cam = render.Camera(view, trk.lo, trk.hi)
    for wall in (trk.left, trk.right, trk.center):
        pts = cam.to_screen(wall)
        assert np.all(pts[:, 0] >= view.left) and np.all(pts[:, 0] <= view.right)
        assert np.all(pts[:, 1] >= view.top) and np.all(pts[:, 1] <= view.bottom)


@check("отрисовка: короткий путь не рисуется и не падает")
def _(rng):
    import render
    view = field_view()
    cam = render.Camera(view, np.array([-100.0, -100.0]), np.array([100.0, 100.0]))
    surf = canvas(render.BG)
    for points in ([], [[0.0, 0.0]], np.zeros((1, 2))):
        render.draw_path(surf, cam, points, render.TEXT)
    assert not ink(surf, render.BG).any(), "из одной точки нарисовалась линия"


@check("отрисовка: путь из двух точек виден")
def _(rng):
    import render
    view = field_view()
    cam = render.Camera(view, np.array([-100.0, -100.0]), np.array([100.0, 100.0]))
    surf = canvas(render.BG)
    a, b = rng.uniform(-90.0, 90.0, (2, 2))
    render.draw_path(surf, cam, np.stack([a, b]), render.TEXT)
    if np.linalg.norm(a - b) * cam.scale >= 2.0:
        assert ink(surf, render.BG).any(), "линия не нарисовалась"


@check("палитра: цвета корректны, разбитая отличается от целой")
def _(rng):
    import render
    veh = car.random_car(rng)
    live, dead = render.palette(veh, False), render.palette(veh, True)
    assert set(live) == set(dead) == set(veh.kinds) | set(live)
    for colours in (live, dead):
        for kind, colour in colours.items():
            assert len(colour) == 3, kind
            assert all(0 <= c <= 255 for c in colour), (kind, colour)
    assert live != dead
    assert live["body"] == tuple(veh.color) or list(live["body"]) == list(veh.color)


@check("отрисовка: нос машинки впереди центра и внутри габарита")
def _(rng):
    import render
    veh = car.random_car(rng)
    nose = render.nose_point(veh)
    finite(nose)
    assert nose[0] > 0.0, "нос не впереди"
    assert nose[0] <= veh.shape[:, 0].max() + 1e-9
    assert abs(nose[1]) <= veh.width * 0.5 + 1e-9


@check("отрисовка: значок машинки помещается в отведённое место")
def _(rng):
    import render
    veh = car.random_car(rng)
    scale = float(rng.uniform(1.5, 3.0))
    at = (240, 120)
    surf = canvas(render.PANEL_BG, 480, 240)
    render.draw_car_badge(surf, veh, at, scale, float(rng.uniform(-np.pi, np.pi)))
    mask = ink(surf, render.PANEL_BG)
    assert mask.any(), "значок не нарисовался"
    xs = np.argwhere(mask.any(axis=1)).ravel()
    ys = np.argwhere(mask.any(axis=0)).ravel()
    # Граница - наибольший радиус точки контура, а не наибольшая координата:
    # поворот сохраняет радиус, поэтому угловая точка (9.2, 5.0) может уехать
    # по одной оси на 10.5. И не veh.width - колёса выступают за него на десятую.
    reach = float(np.linalg.norm(veh.stacked, axis=1).max()) * scale + 2.0
    assert at[0] - reach <= xs.min() and xs.max() <= at[0] + reach
    assert at[1] - reach <= ys.min() and ys.max() <= at[1] + reach


GAME = None


def the_game():
    """Одна игра на весь прогон: сборка стоит секунды, а нам нужна только раскладка."""
    global GAME
    if GAME is None:
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import main
        GAME = main.Game(seed=0)
    return GAME


def shuffle_panel(game, rng):
    """Подставить в панель случайные, в том числе непомерные, значения."""
    import main
    import train

    game.rounds = int(rng.integers(0, 1000000))
    game.gen = int(rng.integers(0, 100000))
    game.paused = bool(rng.integers(0, 2))
    game.state = str(rng.choice([main.TRAINING, main.SHOWCASE, main.DONE]))
    game.totals = dict(stats.EMPTY, rounds=int(rng.integers(0, 1000000)),
                       finished=int(rng.integers(0, 1000000)),
                       best_time=float(rng.uniform(0.0, 100000.0)))
    if rng.integers(0, 2):
        game.history = [train.GenStats(gen=int(rng.integers(0, 100000)),
                                       best=float(rng.uniform(0.0, 1e7)),
                                       mean=float(rng.uniform(0.0, 1e7)),
                                       alive=int(rng.integers(0, 200)),
                                       finished=int(rng.integers(0, 200)),
                                       best_cp=int(rng.integers(0, 200)),
                                       steps=int(rng.integers(0, 100000)),
                                       best_time=float(rng.uniform(0.0, 100000.0)))]
    else:
        game.history = []
    game.message_left = int(rng.integers(0, 2)) * 10
    game.message = "".join(rng.choice(list("абвгдеж жзийклмн 0123456789"),
                                      int(rng.integers(0, 120))))
    # Машинку тоже перебираем: от неё зависит ширина значка в панели, а значит и
    # место, остающееся строкам. На одной машинке проверка ничего не заметит.
    game.car = car.random_car(rng)


@check("панель: статистика не выходит за панель и не задевает виджеты")
def _(rng):
    import render
    game = the_game()
    shuffle_panel(game, rng)

    game.screen.fill(render.BG)
    game.draw_stats()
    mask = ink(game.screen, render.BG)
    assert mask.any(), "статистика не нарисовалась"

    xs = np.argwhere(mask.any(axis=1)).ravel()
    ys = np.argwhere(mask.any(axis=0)).ravel()
    # Вправо пиксели не проверяем: панель прижата к правому краю окна, поэтому
    # переполнение обрезается самой поверхностью и в маске его не видно. Ширину
    # текста стерегут свойства про fit_text, вылет значка - свойство ниже.
    panel = game.panel_rect
    assert panel.left <= xs.min(), f"вылезла влево: {xs.min()}"
    assert panel.top <= ys.min(), f"вылезла вверх: {ys.min()}"

    top_widget = min(w.rect.top for w in game.ui.widgets.values())
    assert ys.max() < top_widget, f"налезла на виджеты: низ {ys.max()}, виджет с {top_widget}"


@check("панель: строки статистики не заезжают под значок машинки")
def _(rng):
    import main
    import render
    game = the_game()
    shuffle_panel(game, rng)

    # Значок рисуется поверх текста, поэтому наложение после отрисовки не видно:
    # текст просто закрашен. Подменяем отрисовку значка на запись его координат,
    # получаем настоящий прямоугольник и смотрим, попал ли в него текст.
    spot = {}
    drawn = render.draw_car_badge

    def remember(surf, veh, at, scale, angle=0.0):
        spot["at"], spot["scale"] = at, scale

    render.draw_car_badge = remember
    try:
        game.screen.fill(render.BG)
        game.draw_stats()
    finally:
        render.draw_car_badge = drawn

    assert spot, "значок не рисовался"
    reach = main.badge_reach(game.car)
    tall = float(np.abs(game.car.stacked[:, 1]).max()) * spot["scale"]
    box = pygame.Rect(int(spot["at"][0] - reach), int(spot["at"][1] - tall),
                      int(2 * reach), int(2 * tall))

    mask = ink(game.screen, render.BG)
    under = mask[box.left:box.right, box.top:box.bottom]
    assert not under.any(), f"текст под значком: {int(under.sum())} точек в {tuple(box)}"


@check("панель: значок машинки не упирается в край окна")
def _(rng):
    import main
    # Считаем, а не смотрим на пиксели: за краем окна их просто нет.
    # И не по случайной машинке, а по объявленному потолку размеров - до края
    # достаёт примерно каждая сотая, случайная выборка ловила бы поломку изредка.
    longest = car.HALF_SIZE_MAX * np.sqrt(car.ASPECT_MAX)
    right = cfg.WINDOW_W - main.BADGE_INSET + longest * main.BADGE_SCALE
    assert right < cfg.WINDOW_W, f"значок доходит до {right:.1f} при окне {cfg.WINDOW_W}"
    # и заодно: настоящая машинка не длиннее объявленного потолка
    veh = car.random_car(rng)
    assert float(np.abs(veh.stacked[:, 0]).max()) <= longest + 1e-9


@check("панель: виджеты не выходят за панель")
def _(rng):
    import render
    game = the_game()
    shuffle_panel(game, rng)

    game.screen.fill(render.BG)
    game.ui.widgets["graph"].draw(game.screen, game.history)
    game.ui.draw(game.screen)
    mask = ink(game.screen, render.BG)
    xs = np.argwhere(mask.any(axis=1)).ravel()
    ys = np.argwhere(mask.any(axis=0)).ravel()
    panel = game.panel_rect
    assert panel.left <= xs.min() and xs.max() < panel.right
    assert panel.top <= ys.min() and ys.max() < panel.bottom


# ---------------------------------------------------------------- долгий прогон

def click(game, key):
    """Настоящий клик по кнопке панели: нажатие и отпускание внутри неё."""
    import pygame
    at = game.ui.widgets[key].rect.center
    for kind in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
        game.ui.handle(pygame.event.Event(kind, button=1, pos=at))


def poke(game, rng):
    """Случайное вмешательство игрока между кадрами."""
    import pygame

    roll = int(rng.integers(0, 14))
    if roll == 0:
        game.key(int(rng.choice([pygame.K_SPACE, pygame.K_l, pygame.K_n, pygame.K_s,
                                 pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4])))
    elif roll == 1:
        # "новую трассу" не трогаем: она эволюционирует трассу и стоит секунды.
        # "новая машина" трассу не пересобирает, поэтому дешёвая и участвует.
        click(game, str(rng.choice(["show", "pause", "save", "load", "car"])))
    elif roll == 2:
        name = str(rng.choice(["mut_sigma", "elite_frac", "pop_size", "generations", "volume"]))
        bar = game.ui.widgets[name]
        bar.value = float(rng.uniform(bar.lo, bar.hi))
    elif roll == 3:
        name = str(rng.choice(["speed", "replay", "brain", "generator", "level"]))
        widget = game.ui.widgets[name]
        widget.index = int(rng.integers(0, len(widget.options)))
    elif roll == 4:
        game.click_field((int(rng.integers(0, cfg.WINDOW_W)), int(rng.integers(0, cfg.WINDOW_H))))
    elif roll == 5 and game.hud_grab is not None:
        game.drag_hud((int(rng.integers(-500, 2000)), int(rng.integers(-500, 2000))))


def hold(game):
    """То, что обязано быть верно после любого кадра."""
    import main

    assert game.state in (main.TRAINING, main.SHOWCASE, main.DONE), game.state
    assert game.rounds >= 1
    assert game.gen >= 0

    # В обучении едет вся популяция, в показательном заезде - один победитель.
    # DONE достижимо двумя путями: после показательного заезда (машинка одна) и
    # напрямую из next_generation при выключенном повторе (машинок вся пачка).
    if game.state == main.TRAINING:
        assert game.race.n == len(game.brains), f"{game.race.n} против {len(game.brains)}"
        # Сверять только длину мало: после отбора популяция та же по размеру, но
        # уже другая. Заезд обязан ехать именно на текущих мозгах, а не на копии
        # прошлого поколения, поэтому сверяем тождество массива.
        assert game.race.brains is game.brains, "заезд едет на прошлом поколении"
    elif game.state == main.SHOWCASE:
        assert game.race.n == 1, f"в показательном заезде {game.race.n} машинок"
    else:
        assert game.race.n in (1, len(game.brains)), game.race.n

    finite(game.race.pos, game.race.angle, game.race.speed)
    assert np.all(game.race.speed >= 0.0)
    assert np.all(game.race.speed <= game.car.max_speed + 1e-6)
    assert np.all(game.race.cp >= 0) and np.all(game.race.cp <= game.race.last_cp)

    # Путь копится по точке за шаг и сбрасывается на каждом показательном заезде.
    # Если сброс когда-нибудь потеряется, список будет расти весь прогон.
    assert len(game.path) <= cfg.STEPS_PER_GEN + 1, f"путь разросся до {len(game.path)}"

    # То же про историю поколений: она обнуляется в new_round.
    assert len(game.history) <= game.ui.widgets["generations"].hi + 1

    if game.best_brain is not None:
        assert game.best_brain.shape == (brain.genome_size(),)

    assert game.watched in (main.WATCH_LEADER, main.WATCH_NONE) or         0 <= game.watched < game.race.n
    assert game.view.contains(game.hud)

    for key in ("rounds", "finished"):
        assert game.totals[key] >= 0
    if game.totals["best_time"] is not None:
        assert game.totals["best_time"] > 0.0


@check("долгий прогон: игра держит инварианты под случайными нажатиями", HEAVY)
def _(rng):
    game = the_game()
    game.new_round(new_car=bool(rng.integers(0, 2)))
    hold(game)

    import render
    for frame in range(int(rng.integers(200, 900))):
        poke(game, rng)
        game.apply_buttons()
        game.advance()
        hold(game)
        # Изредка рисуем полный кадр: раскладка обязана переживать любое
        # состояние, включая пустую историю, паузу и слежение ни за кем.
        if frame % 25 == 0:
            game.screen.fill(render.BG)
            game.draw_field()
            game.draw_panel()


@check("долгий прогон: счётчики только растут через несколько раундов", HEAVY)
def _(rng):
    import main
    game = the_game()
    game.ui.widgets["speed"].index = 3        # поколение за кадр
    game.ui.widgets["generations"].value = 3
    game.ui.widgets["pop_size"].value = 12

    # Одного раунда мало: первый заполняет пустые счётчики, и порча значения
    # прошлого раунда на нём не видна. Нужно как минимум два подряд.
    seen = (game.totals["rounds"], game.totals["finished"])
    for _ in range(3):
        game.new_round()
        for _ in range(40):
            game.apply_buttons()
            game.advance()
            now = (game.totals["rounds"], game.totals["finished"])
            assert now[0] >= seen[0], f"раунды: {now[0]} после {seen[0]}"
            assert now[1] >= seen[1], f"доехавшие: {now[1]} после {seen[1]}"
            seen = now
            if game.state != main.TRAINING:
                break
        assert game.state != main.TRAINING, "раунд не кончился за 40 кадров"
        assert len(game.history) >= 1
    assert game.totals["rounds"] >= 3, f"засчитано раундов: {game.totals['rounds']}"


@check("сохранения: мозг настоящего раунда переживает запись и чтение", HEAVY)
def _(rng):
    import main
    game = the_game()
    game.ui.widgets["speed"].index = 3
    game.ui.widgets["generations"].value = 2
    game.ui.widgets["pop_size"].value = 10
    game.ui.widgets["replay"].index = int(rng.integers(0, 3))
    game.new_round()

    # Никаких подставленных вручную мозгов: раунд отрабатывает по-настоящему,
    # в том числе показательный заезд. Именно на нём сохранение однажды и
    # ломалось - приспособленность бралась у заезда с одной машинкой.
    for _ in range(40):
        game.apply_buttons()
        game.advance()
        if game.state != main.TRAINING:
            break
    assert game.state != main.TRAINING

    # Ждём не "какой-нибудь из сохранённых", а именно лучший мозг последнего
    # обученного поколения. Слабая проверка проходит и тогда, когда сохранение
    # свалилось на нулевые оценки и записало первый попавшийся геном.
    # Гарантированно уводим игру в показательный заезд: там едет одна машинка,
    # а популяция остаётся полной. Именно это расхождение и ломало сохранение.
    click(game, "show")
    game.apply_buttons()
    assert game.race.n == 1 and len(game.brains) > 1

    assert game.last_fitness is not None and len(game.last_fitness) == len(game.brains)
    # Снимок до нажатий: загрузка заменяет и популяцию, и оценки, поэтому
    # сверять файл с тем, что осталось в игре после неё, бессмысленно.
    kept_pop = game.brains.copy()
    kept_fit = np.asarray(game.last_fitness, dtype=float).copy()
    want = kept_pop[int(np.argmax(kept_fit))].copy()

    click(game, "save")
    game.apply_buttons()
    game.best_brain = None
    click(game, "load")
    game.apply_buttons()

    assert game.best_brain is not None, "загрузка не нашла только что сохранённое"
    assert np.allclose(game.best_brain, want), "загрузился не лучший мозг поколения"

    # Прямая сверка с файлом. Без неё поломка видна только когда лучший мозг
    # оказался не первым, а при элитизме он как раз часто первый: победитель
    # переносится в новое поколение без изменений и садится в начало массива.
    saved = stats.load_brains()
    assert saved is not None, "файл сохранения не читается"
    genomes, fitness = saved
    assert np.allclose(genomes, kept_pop), "записана не та популяция"
    assert np.allclose(fitness, kept_fit), "записаны не те оценки"

    hold(game)


# ---------------------------------------------------------------- прочее

@check("трасса: сложность монотонно ослабляет требование к радиусу")
def _(rng):
    a, b = sorted(rng.uniform(0.0, 1.0, 2))
    assert track.radius_factor(a) >= track.radius_factor(b)


@check("трасса: интерес неотрицателен и конечен")
def _(rng):
    value = track.interest(random_line(rng))
    assert np.isfinite(value) and value >= 0.0


@check("трасса: сглаживание сигнала сохраняет среднее")
def _(rng):
    values = rng.normal(size=int(rng.integers(30, 400)))
    w = int(rng.integers(1, 20))
    assert abs(track.smooth_signal(values, w).mean() - values.mean()) < 1e-6


@check("поле: чем дальше от линии, тем меньше значение", SLOW)
def _(rng):
    line = random_line(rng)
    half = float(rng.uniform(15.0, 45.0))
    f = field.build(line, half, cell=6.0)
    i = int(rng.integers(0, len(line)))
    normal = track.normals(line)[i]
    values = [float(f.sample(line[i] + normal * d)) for d in (0.0, half * 0.5, half * 1.5)]
    assert values[0] > values[1] > values[2]


@check("заезд: ближайшая машинка действительно ближайшая", SLOW)
def _(rng):
    r = small_race(rng)
    for _ in range(int(rng.integers(5, 60))):
        r.step()
    point = rng.uniform(r.track.lo, r.track.hi)
    index = r.nearest_to(point)
    gaps = np.linalg.norm(r.pos - point, axis=1)
    assert gaps[index] <= gaps.min() + 1e-9


@check("заезд: продвижение вперёд поднимает приспособленность", SLOW)
def _(rng):
    r = small_race(rng, n=2)
    k = int(rng.integers(0, max(1, r.last_cp - 1)))
    r.cp[:] = k
    leg = r.track.cp_mid[k + 1] - r.track.cp_mid[k]
    scores = []
    for share in (0.0, 0.3, 0.6, 0.9):
        r.pos[:] = r.track.cp_mid[k] + leg * share
        scores.append(float(r.fitness()[0]))
    assert scores == sorted(scores)


@check("обучение: статистика поколения согласована", SLOW)
def _(rng):
    import train
    r = small_race(rng)
    while not r.done:
        r.step()
    s = train.gen_stats(0, r)
    assert 0 <= s.alive <= r.n and 0 <= s.finished <= r.n
    assert 0 <= s.best_cp <= r.last_cp
    assert s.mean <= s.best + 1e-9
    assert 0 < s.steps <= cfg.STEPS_PER_GEN
    assert s.best_time >= 0.0


@check("датасет: оценка либо положительна, либо её нет")
def _(rng):
    radii = rng.uniform(cfg.R_MIN, cfg.R_MAX, cfg.TRACK_POINTS)
    value = dataset.judge(radii, float(rng.uniform(20.0, 60.0)), float(rng.uniform(0.0, 1.0)))
    assert value is None or (np.isfinite(value) and value >= 0.0)


@check("звук: удар затухает к концу")
def _(rng):
    s = np.abs(sound.crash_sample(int(rng.choice([22050, 44100]))).astype(np.int32))
    quarter = max(1, len(s) // 4)
    assert s[:quarter].mean() > s[-quarter:].mean() * 5


@check("звук: ноты финиша идут вверх")
def _(rng):
    s = sound.finish_sample(int(rng.choice([22050, 44100]))).astype(np.int32)
    third = len(s) // 3
    crossings = [np.count_nonzero(np.diff(np.sign(s[i * third:(i + 1) * third])) != 0)
                 for i in range(3)]
    assert crossings[0] < crossings[1] < crossings[2]


@check("панель: любой текст обрезается по ширине")
def _(rng):
    import pygame
    import render
    pygame.init()
    font = pygame.font.SysFont("consolas", int(rng.integers(9, 24)))
    width = int(rng.integers(20, 400))
    alphabet = "абвгдеёжзиклмнопрстуфхцчшщыэюя 0123456789"
    text = "".join(rng.choice(list(alphabet), int(rng.integers(0, 200))))
    assert font.size(render.fit_text(font, text, width))[0] <= width


@check("панель: короткий текст не трогается")
def _(rng):
    import pygame
    import render
    pygame.init()
    font = pygame.font.SysFont("consolas", 15)
    text = "".join(rng.choice(list("абвгд "), int(rng.integers(0, 8))))
    assert render.fit_text(font, text, 400) == text


@check("телеметрия: ничего не выходит за окошко")
def _(rng):
    import pygame
    import render
    pygame.init()
    font = pygame.font.SysFont("consolas", 15)
    box = pygame.Rect(30, 30, render.HUD_W, render.HUD_H)
    surf = pygame.Surface((box.width + 60, box.height + 60))
    OUTSIDE = (255, 0, 255)
    surf.fill(OUTSIDE)

    veh = car.random_car(rng)
    watch = {
        "index": int(rng.integers(0, 1000)),
        "alive": bool(rng.integers(0, 2)),
        "finished": bool(rng.integers(0, 2)),
        "rays": rng.uniform(0.0, 1.0, cfg.N_RAYS),
        "speed": float(rng.uniform(0.0, 100000.0)),
        "steer": float(rng.uniform(-1.0, 1.0)),
        "throttle": float(rng.uniform(-1.0, 1.0)),
        "cp": f"{int(rng.integers(0, 100000))}/{int(rng.integers(0, 100000))}",
    }
    render.draw_telemetry(surf, box, font, veh, watch)

    pixels = pygame.surfarray.array3d(surf)
    outside = np.ones(pixels.shape[:2], dtype=bool)
    outside[box.left:box.right, box.top:box.bottom] = False
    painted = np.any(pixels != np.array(OUTSIDE), axis=2) & outside
    assert not painted.any(), f"краска за окошком в {np.argwhere(painted)[0].tolist()}"


@check("сохранения: строка итогов влезает в панель после обрезки")
def _(rng):
    import pygame
    import render
    pygame.init()
    font = pygame.font.SysFont("consolas", 15)
    rounds = int(rng.integers(0, 1000000))
    totals = dict(stats.EMPTY, rounds=rounds,
                  finished=int(rng.integers(0, rounds + 1)),
                  best_time=float(rng.uniform(0.0, 100000.0)))
    width = cfg.PANEL_W - 28
    assert font.size(render.fit_text(font, stats.summary(totals), width))[0] <= width


# ---------------------------------------------------------------- прогон

def run_round(seed, counts, verbose=False):
    rng = np.random.default_rng(seed)
    failures = []
    done = 0
    for name, fn, cost in CHECKS:
        cases = counts[cost]
        t0 = time.perf_counter()
        for case in range(cases):
            try:
                fn(rng)
            except Exception:
                failures.append((name, case, traceback.format_exc(limit=4)))
            done += 1
        if verbose:
            # flush: без него вывод в трубу копится до конца круга, и по логу
            # сервера сборки не понять, на каком свойстве прогон встал.
            print(f"    {time.perf_counter() - t0:6.1f} с  {name}: {cases}", flush=True)
    return done, failures


def main():
    ap = argparse.ArgumentParser(description="Проверка инвариантов случайными данными")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--fast", type=int, default=120)
    ap.add_argument("--slow", type=int, default=4)
    ap.add_argument("--heavy", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true", help="печатать каждое свойство и его время")
    args = ap.parse_args()
    # Вывод по-русски, а в трубе Windows берёт кодовую страницу системы. На английской
    # это cp1252, и первая же строка падает с UnicodeEncodeError - так и вышло на
    # сервере сборки, ещё до первой проверки.
    sys.stdout.reconfigure(encoding="utf-8")

    counts = {FAST: args.fast, SLOW: args.slow, HEAVY: args.heavy}
    tally = {tier: sum(1 for _, _, cost in CHECKS if cost == tier) for tier in counts}
    per_round = sum(tally[tier] * counts[tier] for tier in counts)
    print(f"свойств {len(CHECKS)} ({tally[FAST]} быстрых, {tally[SLOW]} тяжёлых, "
          f"{tally[HEAVY]} очень тяжёлых)")
    print(f"на круг {per_round} проверок, кругов {args.rounds}, всего {per_round * args.rounds}")
    print()

    started = time.perf_counter()
    total, broken = 0, []
    for r in range(args.rounds):
        t0 = time.perf_counter()
        done, failures = run_round(args.seed + r * 1000, counts, args.verbose)
        total += done
        broken += failures
        mark = "ОШИБКИ" if failures else "чисто"
        print(f"круг {r + 1:2d}: {done} проверок за {time.perf_counter() - t0:5.1f} с  {mark}"
              + (f" ({len(failures)})" if failures else ""))

    print()
    print(f"всего {total} проверок за {time.perf_counter() - started:.0f} с")
    if not broken:
        print("нарушений инвариантов нет")
        return

    seen = {}
    for name, case, tb in broken:
        seen.setdefault(name, []).append(tb)
    print(f"НАРУШЕНИЙ: {len(broken)} в {len(seen)} свойствах")
    for name, traces in seen.items():
        print()
        print(f"=== {name} ({len(traces)} раз) ===")
        print(traces[0])
    # Ненулевой код возврата: иначе прогон на сервере сборки проходит зелёным и при
    # найденных нарушениях - человек их увидит только если откроет вывод целиком.
    raise SystemExit(1)


if __name__ == "__main__":
    main()
