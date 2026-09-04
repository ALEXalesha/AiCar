# AI Car Racing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Игра на pygame, где CPPN генерирует трассу, CPPN генерирует машинку, а сеть-водитель обучается генетическим алгоритмом проезжать трассу от старта до финиша.

**Architecture:** Плоский набор модулей в корне репозитория, без пакетов. Вся математика на numpy и векторизована по всей популяции сразу: 50 машинок это не 50 объектов, а массивы формы `(50, ...)`. Лучи-датчики и столкновения считаются не пересечением отрезков, а обращением к заранее построенному полю расстояний до стен. Генетический алгоритм в `evolution.py` ничего не знает о смысле генома и потому используется и для трасс, и для мозгов.

**Tech Stack:** Python 3.13, numpy 2.4.4, pygame 2.6.1, pytest. Внешних ассетов нет.

**Спецификация:** [docs/superpowers/specs/2026-09-04-ai-car-racing-design.md](../specs/2026-09-04-ai-car-racing-design.md)

**Репозиторий:** `origin` = `http://gitea.local/ALEXaloysha/AiCar.git`, автор `ALEXaloysha <203467574+ALEXalesha@users.noreply.github.com>`.

---

## Структура файлов

Всё в корне репозитория, тесты в `tests/`, документация модулей в `docs/modules/`.

| Файл | Ответственность | Зависит от |
|---|---|---|
| `config.py` | все константы | - |
| `cppn.py` | CPPN: размер генома, прогон, замкнутая форма из радиусов | config |
| `evolution.py` | турнирный отбор, скрещивание, мутация, элита | - |
| `track.py` | центральная линия, стены, чекпоинты, кривизна, эволюция трассы | config, cppn, evolution |
| `field.py` | поле расстояний до стен, выборка значений | config |
| `car.py` | форма и физические параметры машинки из генома | config, cppn |
| `sensors.py` | лучи-датчики шагами по полю | config, field |
| `brain.py` | сеть водителя, прогон всей популяции | config |
| `race.py` | состояние заезда: физика, чекпоинты, приспособленность | config, sensors, brain |
| `train.py` | прогон поколения, обучение без графики, CLI | config, race, evolution |
| `render.py` | отрисовка трассы, машинок, лучей, камера | config |
| `ui.py` | боковая панель: кнопки, ползунки, график | config |
| `sound.py` | синтез звуков | config |
| `stats.py` | сбор, сохранение и загрузка статистики и мозгов | config |
| `main.py` | игровой цикл, машина состояний | всё |

Порядок задач подобран так, что каждый следующий модуль опирается только на уже готовые. После задачи 11 проект уже работает и обучается в консоли, после задачи 13 появляется картинка.

---

## Task 1: Скелет проекта

**Files:**
- Create: `config.py`
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `docs/modules/config.md`

- [ ] **Step 1: Написать `config.py`**

```python
import numpy as np

WINDOW_W, WINDOW_H = 1280, 720
PANEL_W = 300
FPS = 60
DT = 1.0 / 60.0

POP_SIZE = 50
HIDDEN = 10
N_OUT = 2
MAX_GENERATIONS = 60
STEPS_PER_GEN = 1800
IDLE_LIMIT = 300

ELITE_FRAC = 0.1
MUT_SIGMA = 0.15
MUT_RATE = 0.2
TOURNAMENT_K = 3

RAY_ANGLES = np.radians(np.array([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]))
N_RAYS = len(RAY_ANGLES)
RAY_SAMPLES = 32
RAY_STEP = 8.0
RAY_MAX = RAY_SAMPLES * RAY_STEP

TRACK_POINTS = 360
TRACK_WIDTH = 90.0
CHECKPOINT_STEP = 8
CHECKPOINT_RADIUS = 45.0
R_MIN, R_MAX = 120.0, 330.0
LEN_MIN, LEN_MAX = 1800.0, 4500.0
TRACK_POP = 200
TRACK_GENS = 15
SMOOTH_WINDOW = 15
VARIETY_SCALE = 10000.0
DIFFICULTY = 8.0

FIELD_CELL = 6.0
FIELD_PAD = 80.0

DRAG = 0.9
CAR_POINTS = 24
CAR_R_MIN, CAR_R_MAX = 5.0, 14.0

ENGINE_LEVELS = 12
SAMPLE_RATE = 22050
VOLUME = 0.4

TRACK_CPPN_LAYERS = (2, 8, 8, 8, 1)
CAR_CPPN_LAYERS = (2, 6, 6, 1)
CPPN_INIT_SCALE = 1.5

SAVE_DIR = "saves"
BRAIN_FILE = "saves/brains.npz"
STATS_FILE = "saves/stats.json"
```

- [ ] **Step 2: Написать `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
```

- [ ] **Step 3: Создать пустой `tests/__init__.py`**

Пустой файл, нужен чтобы тесты видели модули из корня.

- [ ] **Step 4: Проверить, что pytest запускается**

Run: `python -m pytest -q`
Expected: `no tests ran` без ошибок импорта.

- [ ] **Step 5: Написать `docs/modules/config.md`**

Документ описывает каждую константу: что задаёт, в каких единицах, что сломается при изменении. Обязательно объяснить:

- почему `DT` фиксированный, а не берётся из реального FPS (иначе эволюция даёт разные результаты на разных машинах и результаты нельзя сравнивать)
- почему `RAY_MAX` вычисляется как `RAY_SAMPLES * RAY_STEP`, а не задаётся отдельно (иначе луч возвращал бы значение дальше, чем реально проверял)
- почему `CHECKPOINT_RADIUS` (45) меньше расстояния между чекпоинтами (около 64), но больше пути за один шаг (максимум 4 пикселя): первое чтобы нельзя было засчитать два чекпоинта разом, второе чтобы нельзя было проскочить мимо

- [ ] **Step 6: Коммит**

```bash
git add config.py pytest.ini tests/__init__.py docs/modules/config.md
git commit -m "Скелет проекта: константы и настройка pytest"
git push origin main
```

---

## Task 2: CPPN

**Files:**
- Create: `cppn.py`
- Create: `tests/test_cppn.py`
- Create: `docs/modules/cppn.md`

- [ ] **Step 1: Написать падающие тесты в `tests/test_cppn.py`**

```python
import numpy as np
import pytest

import cppn


def test_genome_size_matches_layers():
    assert cppn.genome_size((2, 8, 1)) == 2 * 8 + 8 + 8 * 1 + 1


def test_forward_output_shape():
    layers = (2, 6, 6, 1)
    g = np.zeros(cppn.genome_size(layers))
    x = np.zeros((10, 2))
    assert cppn.forward(g, layers, x).shape == (10, 1)


def test_forward_output_is_bounded():
    layers = (2, 8, 8, 8, 1)
    rng = np.random.default_rng(0)
    g = rng.normal(0, 3.0, cppn.genome_size(layers))
    x = rng.normal(0, 1.0, (200, 2))
    out = cppn.forward(g, layers, x)
    assert np.all(out >= -1.0) and np.all(out <= 1.0)


def test_ring_shape_is_closed():
    layers = (2, 8, 8, 8, 1)
    rng = np.random.default_rng(1)
    g = rng.normal(0, 1.5, cppn.genome_size(layers))
    pts = cppn.ring_shape(g, layers, 360, 100.0, 300.0)
    gap = np.linalg.norm(pts[0] - pts[-1])
    spacing = np.linalg.norm(pts[1] - pts[0])
    assert gap < spacing * 3.0


def test_ring_shape_radius_within_bounds():
    layers = (2, 8, 8, 8, 1)
    rng = np.random.default_rng(2)
    g = rng.normal(0, 1.5, cppn.genome_size(layers))
    pts = cppn.ring_shape(g, layers, 360, 100.0, 300.0)
    r = np.linalg.norm(pts, axis=1)
    assert r.min() >= 100.0 - 1e-6
    assert r.max() <= 300.0 + 1e-6


def test_same_genome_gives_same_shape():
    layers = (2, 6, 6, 1)
    rng = np.random.default_rng(3)
    g = rng.normal(0, 1.5, cppn.genome_size(layers))
    a = cppn.ring_shape(g, layers, 24, 5.0, 14.0)
    b = cppn.ring_shape(g, layers, 24, 5.0, 14.0)
    assert np.allclose(a, b)


def test_different_genomes_give_different_shapes():
    layers = (2, 6, 6, 1)
    rng = np.random.default_rng(4)
    size = cppn.genome_size(layers)
    a = cppn.ring_shape(rng.normal(0, 1.5, size), layers, 24, 5.0, 14.0)
    b = cppn.ring_shape(rng.normal(0, 1.5, size), layers, 24, 5.0, 14.0)
    assert not np.allclose(a, b)


def test_wrong_genome_length_raises():
    with pytest.raises(ValueError):
        cppn.forward(np.zeros(5), (2, 8, 1), np.zeros((3, 2)))
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `python -m pytest tests/test_cppn.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'cppn'`.

- [ ] **Step 3: Написать `cppn.py`**

```python
import numpy as np


def _gauss(x):
    return np.exp(-x * x)


ACTS = (np.sin, _gauss, np.tanh, np.abs)


def genome_size(layers):
    return sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))


def _unpack(genome, layers):
    if len(genome) != genome_size(layers):
        raise ValueError(f"геном длины {len(genome)}, для слоёв {layers} нужно {genome_size(layers)}")
    ws, bs, k = [], [], 0
    for i in range(len(layers) - 1):
        n_in, n_out = layers[i], layers[i + 1]
        ws.append(genome[k:k + n_in * n_out].reshape(n_in, n_out))
        k += n_in * n_out
        bs.append(genome[k:k + n_out])
        k += n_out
    return ws, bs


def _mixed(h):
    out = np.empty_like(h)
    for j in range(h.shape[1]):
        out[:, j] = ACTS[j % len(ACTS)](h[:, j])
    return out


def forward(genome, layers, x):
    ws, bs = _unpack(genome, layers)
    h = x
    last = len(ws) - 1
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w + b
        h = np.tanh(h) if i == last else _mixed(h)
    return h


def ring_shape(genome, layers, n_points, r_min, r_max):
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    x = np.stack([np.sin(theta), np.cos(theta)], axis=1)
    r = forward(genome, layers, x)[:, 0]
    r = r_min + (r + 1.0) * 0.5 * (r_max - r_min)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def random_genome(layers, rng, scale=1.5):
    return rng.normal(0.0, scale, genome_size(layers))
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

Run: `python -m pytest tests/test_cppn.py -q`
Expected: 8 passed.

- [ ] **Step 5: Написать `docs/modules/cppn.md`**

Документ должен содержать:

- что такое CPPN и чем отличается от обычной сети: веса не обучают градиентом, случайные веса уже дают осмысленные формы, потому что активации внутри периодические и симметричные
- почему на вход подаются `sin θ` и `cos θ`, а не сам угол: при θ=0 и θ=2π вход одинаковый, значит и радиус одинаковый, значит кривая замкнута по построению. Если подать θ напрямую, концы не сойдутся
- почему активации перемешаны по индексу нейрона (`ACTS[j % 4]`): синус даёт повторяющиеся детали, гауссиана даёт локальные выступы, `tanh` сглаживает, модуль даёт изломы. Один тип на всю сеть даёт однообразные формы
- почему перемешивание идёт по индексу, а не случайно: иначе один и тот же геном давал бы разные формы при разных запусках и воспроизводимость по seed пропала бы
- почему выходной слой всегда `tanh`: нужен ограниченный диапазон, чтобы линейно отобразить его в `[r_min, r_max]`
- разбор `genome_size`, `forward`, `ring_shape`, `random_genome`: параметры, формы массивов, что возвращают
- пример: сгенерировать и напечатать 8 радиусов
- связь с `track.py` (генератор трассы) и `car.py` (генератор кузова)

- [ ] **Step 6: Коммит**

```bash
git add cppn.py tests/test_cppn.py docs/modules/cppn.md
git commit -m "CPPN: замкнутые формы из случайных весов"
git push origin main
```

---

## Task 3: Генетический алгоритм

**Files:**
- Create: `evolution.py`
- Create: `tests/test_evolution.py`
- Create: `docs/modules/evolution.md`

- [ ] **Step 1: Написать падающие тесты в `tests/test_evolution.py`**

```python
import numpy as np

import evolution


def test_random_population_shape():
    rng = np.random.default_rng(0)
    pop = evolution.random_population(20, 7, rng)
    assert pop.shape == (20, 7)


def test_evolve_keeps_shape():
    rng = np.random.default_rng(0)
    pop = evolution.random_population(20, 7, rng)
    fit = rng.random(20)
    assert evolution.evolve(pop, fit, rng).shape == (20, 7)


def test_elite_survives_unchanged():
    rng = np.random.default_rng(0)
    pop = evolution.random_population(20, 7, rng)
    fit = np.arange(20.0)
    new = evolution.evolve(pop, fit, rng, elite_frac=0.1)
    assert np.allclose(new[0], pop[19])


def test_crossover_takes_genes_only_from_parents():
    rng = np.random.default_rng(0)
    a = np.zeros(50)
    b = np.ones(50)
    child = evolution.crossover(a, b, rng)
    assert np.all((child == 0.0) | (child == 1.0))
    assert 0 < child.sum() < 50


def test_mutate_keeps_length():
    rng = np.random.default_rng(0)
    g = np.zeros(100)
    assert len(evolution.mutate(g, rng, 0.1, 0.5)) == 100


def test_mutate_changes_roughly_rate_fraction():
    rng = np.random.default_rng(0)
    g = np.zeros(10000)
    out = evolution.mutate(g, rng, 0.1, 0.2)
    changed = np.count_nonzero(out != 0.0)
    assert 1700 < changed < 2300


def test_tournament_prefers_better():
    rng = np.random.default_rng(0)
    pop = np.arange(10.0).reshape(10, 1)
    fit = np.arange(10.0)
    picks = [evolution.tournament(pop, fit, rng, 5)[0] for _ in range(200)]
    assert np.mean(picks) > 6.0


def test_flat_fitness_resets_population():
    rng = np.random.default_rng(0)
    pop = evolution.random_population(20, 7, rng)
    fit = np.zeros(20)
    new = evolution.evolve(pop, fit, rng)
    assert new.shape == pop.shape
    assert not np.allclose(new, pop)
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `python -m pytest tests/test_evolution.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'evolution'`.

- [ ] **Step 3: Написать `evolution.py`**

```python
import numpy as np

import config as cfg


def random_population(n, genome_size, rng, scale=1.0):
    return rng.normal(0.0, scale, (n, genome_size))


def tournament(pop, fitness, rng, k):
    idx = rng.integers(0, len(pop), k)
    return pop[idx[np.argmax(fitness[idx])]]


def crossover(a, b, rng):
    mask = rng.random(len(a)) < 0.5
    return np.where(mask, a, b)


def mutate(genome, rng, sigma, rate):
    mask = rng.random(len(genome)) < rate
    return genome + mask * rng.normal(0.0, sigma, len(genome))


def evolve(pop, fitness, rng, elite_frac=cfg.ELITE_FRAC, mut_sigma=cfg.MUT_SIGMA,
           mut_rate=cfg.MUT_RATE, tournament_k=cfg.TOURNAMENT_K):
    n, g = pop.shape
    if float(np.ptp(fitness)) < 1e-9:
        scale = float(np.std(pop)) or 1.0
        return random_population(n, g, rng, scale)

    n_elite = max(1, int(round(n * elite_frac)))
    order = np.argsort(fitness)[::-1]
    new = np.empty_like(pop)
    new[:n_elite] = pop[order[:n_elite]]
    for i in range(n_elite, n):
        a = tournament(pop, fitness, rng, tournament_k)
        b = tournament(pop, fitness, rng, tournament_k)
        new[i] = mutate(crossover(a, b, rng), rng, mut_sigma, mut_rate)
    return new
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

Run: `python -m pytest tests/test_evolution.py -q`
Expected: 8 passed.

- [ ] **Step 5: Написать `docs/modules/evolution.md`**

Документ должен содержать:

- почему модуль ничего не знает о смысле генома: на вход массив `(N, G)` и оценки `(N,)`, на выход новый массив `(N, G)`. Именно поэтому один и тот же код работает и для трасс (177 чисел), и для мозгов (112 чисел)
- разбор турнирного отбора: берём `k` случайных, выбираем лучшего. При `k=1` это чистый случай, при `k=N` всегда побеждает лучший и разнообразие умирает. `k=3` даёт умеренное давление отбора
- разбор равномерного скрещивания и почему оно, а не одноточечное: у нас геном это веса сети, соседние гены не связаны сильнее далёких, разрезать по одной точке нет смысла
- разбор мутации: `rate` задаёт долю изменяемых генов, `sigma` силу сдвига. Большая `sigma` ломает найденное решение, маленькая не даёт выбраться из локального минимума
- зачем элита: без неё лучшее решение может пропасть из-за неудачной мутации, и приспособленность будет прыгать вниз
- зачем сброс при нулевой дисперсии приспособленности: если все машинки умерли мгновенно и у всех оценка 0, турнирный отбор вырождается в случайный, скрещивание перемешивает одинаково плохие геномы, популяция навсегда застревает. Проще сгенерировать новую
- пример: 30 поколений на игрушечной задаче "максимизировать сумму генов", напечатать рост
- связь с `track.py` и `train.py`

- [ ] **Step 6: Коммит**

```bash
git add evolution.py tests/test_evolution.py docs/modules/evolution.md
git commit -m "Генетический алгоритм: отбор, скрещивание, мутация, элита"
git push origin main
```

---

## Задачи 4-18

Ниже задачи в сжатом виде: файлы, ключевые решения и что проверяют тесты. Код пишется по спецификации, каждая задача заканчивается коммитом вместе с документом модуля.

### Task 4: Геометрия трассы

Files: `track.py`, `tests/test_track_geometry.py`, `docs/modules/track.md` (первая половина)

Функции: `centerline`, `smooth_closed`, `tangents`, `offset_walls`, `curvature`, `polyline_length`, `build_checkpoints`, `any_self_intersection`.

Ключевое: сглаживание циклическое, края склеиваются через `np.roll`, иначе на стыке θ=0 будет излом. Нормаль это касательная, повёрнутая на 90°. Кривизна по центральной разности выходит в 4 раза меньше истинной из-за шага `2h` в первой производной. Это не ошибка, а константа масштаба, важны только относительные значения. Проверка самопересечений векторная, на прореженной вчетверо линии.

Тесты: линия замкнута; сглаживание не рвёт стык; стены отстоят ровно на `half` по нормали; кривизна круга радиуса R равна 1/(4R) с точностью 5%; длина окружности радиуса 100 равна 628 с точностью 1%; чекпоинты идут по порядку и их концы лежат на стенах; восьмёрка распознаётся как самопересекающаяся, круг нет.

### Task 5: Эволюция трассы

Files: `track.py` (дополнить), `tests/test_track_evolve.py`, `docs/modules/track.md` (вторая половина)

Функции: `track_fitness`, `evolve_track`, класс `Track` (центр, стены, чекпоинты, середины чекпоинтов, габариты).

Порядок проверок в приспособленности: минимальный радиус кривизны, длина, самопересечение стен, разнообразие кривизны. Каждая ступень возвращает маленькое, но различимое число вместо жёсткого нуля, чтобы у отбора был градиент. Если за три попытки валидной трассы нет, требования смягчаются, на последней возвращается круг.

Тесты: `evolve_track` даёт трассу с минимальным радиусом больше `width*0.75`; длина в пределах; стены не самопересекаются; один seed даёт одну трассу; при невыполнимых требованиях возвращается круг, а не исключение.

### Task 6: Поле расстояний

Files: `field.py`, `tests/test_field.py`, `docs/modules/field.md`

Класс `Field(values, origin, cell)` с методом `sample(pts)` для массива любой формы `(..., 2)`. Функция `build(center, half_width, cell, pad)`. Расстояние до центральной линии считается кусками по 4096 клеток, иначе матрица клетки на точки съедает под сотню мегабайт.

Тесты: значение в точке на центральной линии равно `half_width`; на стене около нуля; далеко снаружи отрицательное; вне границ сетки возвращается большое отрицательное; форма результата совпадает с формой входа без последней оси; построение поля укладывается в секунду.

### Task 7: Лучи-датчики

Files: `sensors.py`, `tests/test_sensors.py`, `docs/modules/sensors.md`

Функция `cast(pos, angle, field)` возвращает `(N, 7)` расстояний. Внутри массив проб `(N, 7, 32, 2)`, первая проба с отрицательным значением поля даёт расстояние. `np.argmax` по булевой оси находит первый `True`, но при полном отсутствии `True` вернёт 0, поэтому результат перекрывается маской `any`. Это самая частая ошибка в таком коде.

Тесты: в центре прямого коридора шириной 90 боковые лучи дают около 45, передний `RAY_MAX`; при повороте машинки показания сдвигаются по кругу; вне трассы все лучи дают минимум; форма результата `(N, 7)`.

### Task 8: Генератор машинки

Files: `car.py`, `tests/test_car.py`, `docs/modules/car.md`

Класс `Car` с полями `shape, mass, accel, max_speed, max_steer, half_width, length, color`. Функция `generate(genome)`. Силуэт разворачивается по главной оси через собственные векторы ковариации, чтобы длина всегда шла вдоль x независимо от того, как CPPN нарисовал форму. Из площади получается масса, из отношения длины к ширине - максимальная скорость и угол поворота руля: длинная узкая машина быстрее, но хуже вписывается.

Тесты: при 200 случайных геномах параметры в пределах (скорость 150-280, руль 1.8-3.6, масса 0.8-2.5); один геном даёт одну машину; разные геномы дают разные; после разворота длина вдоль x не меньше ширины вдоль y; площадь силуэта положительна.

### Task 9: Сеть-водитель

Files: `brain.py`, `tests/test_brain.py`, `docs/modules/brain.md`

Функции `genome_size(n_in, hidden, n_out)` и `forward(pop, obs, n_in, hidden, n_out)`. Вся популяция считается двумя вызовами `np.einsum`, без цикла по машинкам.

Тесты: размер генома 112 при 8 входах и 10 скрытых; форма выхода `(N, 2)`; выход в пределах `[-1, 1]`; нулевой геном даёт нулевой выход; результат совпадает с наивным поматричным расчётом для трёх случайных геномов.

### Task 10: Заезд

Files: `race.py`, `tests/test_race.py`, `docs/modules/race.md`

Класс `Race` с состоянием `pos, angle, speed, alive, cp, idle, finish_step`. Методы `step()`, `fitness()`, свойство `done`.

Приспособленность: взятые чекпоинты по 100, плюс доля пути до следующего через проекцию вектора на отрезок между серединами чекпоинтов, плюс бонус за скорость финиша.

Тесты: мёртвая машинка не двигается после шага; стоящая не поворачивает даже при полном руле; при взятии чекпоинта приспособленность растёт примерно на 100; машинка с `idle` больше лимита умирает; машинка вне трассы умирает на первом шаге; `done` истинно когда все мертвы; при движении вперёд приспособленность растёт плавно.

### Task 11: Обучение без графики, первая веха

Files: `train.py`, `tests/test_train.py`, `docs/modules/train.md`

Функции `run_generation(track, field, car, brains)` и `train(...)`, возвращающая историю поколений. Запуск `python train.py --seed 1 --generations 40` печатает по строке на поколение и итог.

Тесты: длина истории не больше числа поколений; лучшая приспособленность за 40 поколений растёт минимум вдвое относительно первого; обучение останавливается досрочно при финише; один seed даёт один результат.

После этой задачи проект уже работает: обучение идёт, машинка доезжает, всё видно в консоли.

### Task 12: Отрисовка

Files: `render.py`, `tests/test_render.py`, `docs/modules/render.md`

Класс `Camera` (мир в экран, подгонка масштаба под габариты трассы, слежение за лидером). Функции `draw_track`, `draw_cars`, `draw_rays`, `draw_checkpoints`. Мёртвые машинки бледные, у лидера видны лучи.

Тесты только на камеру, без окна: центр трассы попадает в центр области; вся трасса влезает; преобразование мир в экран и обратно возвращает исходную точку.

### Task 13: Игровой цикл, вторая веха

Files: `main.py`, `docs/modules/main.md`

Машина состояний `TRACK_GEN`, `CAR_GEN`, `TRAINING`, `SHOWCASE`, `PAUSED`. Множитель скорости считает N шагов физики за кадр и рисует только последний. Клавиши: пробел пауза, `R` новый раунд, `T` новая трасса, `S` показательный заезд, `1/2/3/4` скорость.

После этой задачи видно, как машинки учатся.

### Task 14: Панель

Files: `ui.py`, `tests/test_ui.py`, `docs/modules/ui.md`

Классы `Slider`, `Button`, `Toggle`, `Graph`, `Panel`. Всё на прямоугольниках и событиях мыши, без внешних библиотек. График рисуется линиями по истории поколений.

Тесты на логику без окна: ползунок переводит позицию мыши в значение и обратно; значение зажимается в пределах; кнопка срабатывает только при отпускании внутри своей области; график с пустой историей не падает.

### Task 15: Звук

Files: `sound.py`, `tests/test_sound.py`, `docs/modules/sound.md`

Функции синтеза: `engine_bank()` даёт 12 зацикленных пилообразных сэмплов, `crash()` всплеск шума с затуханием, `finish()` три ноты. Класс `SoundBank` с методом `update(speed, max_speed)` и обработкой отсутствия аудиоустройства.

Пилообразный сэмпл обязан содержать целое число периодов, иначе на стыке цикла будет щелчок. Длина сэмпла подгоняется под частоту.

Тесты на синтез без микшера: сэмплы в диапазоне int16; число периодов целое; длина `crash` около 120 мс; отсутствие микшера не роняет `SoundBank`.

### Task 16: Статистика

Files: `stats.py`, `tests/test_stats.py`, `docs/modules/stats.md`

Класс `RunStats` для текущего обучения и функции `save_brains`, `load_brains`, `load_totals`, `save_totals`. Мозг с несовпадающим размером генома отбрасывается с сообщением, а не роняет игру.

Тесты: сохранение и загрузка возвращают тот же массив; геном другой длины отбрасывается и возвращается `None`; отсутствующий файл даёт пустую статистику; счётчики складываются между запусками.

### Task 17: README и сверка документации

Files: `README.md`, правки в `docs/modules/*`

README: что это, установка, запуск, управление, структура проекта, ссылки на документы модулей, короткое объяснение всех трёх сетей.

Сверка: каждая публичная функция упомянута в своём документе, формы массивов совпадают с кодом, примеры вызова запускаются.

### Task 18: Проверка на баги

Прогон полного цикла с разными seed, замер времени поколения, проверка краевых случаев из раздела 12 спецификации: невыполнимые требования к трассе, нулевая дисперсия приспособленности, луч в пустоту, смена размера популяции на ходу, отсутствие аудио, битый файл сохранения. Отчёт с найденным и исправленным.

---

## Самопроверка плана

Сверка с разделами спецификации:

- разделы 4 и 5 (генераторы) закрыты задачами 2, 4, 5, 8
- раздел 6 (водитель) закрыт задачами 3, 9, 10
- раздел 7 (физика и поле) закрыт задачами 6, 7, 10
- раздел 8 (цикл игры) закрыт задачей 13
- раздел 9 (интерфейс) закрыт задачами 12, 14
- раздел 10 (звук) закрыт задачей 15
- раздел 11 (сохранение) закрыт задачей 16
- раздел 12 (краевые случаи) закрыт задачей 18 и отдельными тестами в 5, 6, 15, 16
- раздел 13 (константы) закрыт задачей 1
- раздел 14 (тесты) распределён по всем задачам
- раздел 15 (документация) идёт внутри каждой задачи, сверка в 17
- раздел 16 (репозиторий) закрыт задачей 1 и коммитами в каждой задаче

Имена сверены между задачами: `evolve` из задачи 3 используется в 5 и 11, `Field.sample` из задачи 6 в 7 и 10, `Car.max_speed` из задачи 8 в 10, `run_generation` из задачи 11 в 13.
