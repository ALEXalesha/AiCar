"""AI Car Racing: игра без окна (`Game`) и запуск (`main`).

`Game` - вся игра: раунды, поколения, слежение, панель, звук, и рисование в QPainter. Окна
у неё нет: окно (`window.MainWindow`) крутит её таймером, отдаёт ей мышь и клавиши и зовёт
`paint`. Так тесты, stress.py, самопроверка и кадры для README рисуют ту же игру в картинку.
"""
import argparse
import os
import sys
import time

import numpy as np
from PySide6.QtCore import Qt

import brain
import car
import config as cfg
import evolution
import field
import offscreen
import race
import render
import sound
import stats
import paths
import track
import trackgen
import train
import ui
from rect import Rect

TRAINING = "training"
SHOWCASE = "showcase"
DONE = "done"

STATE_NAME = {TRAINING: "обучение", SHOWCASE: "показательный заезд", DONE: "заезд окончен"}
SPEED_NAMES = ("x1", "x5", "x20", "без отрисовки")
BRAIN_NAMES = ("с нуля", "продолжить")
REPLAY_ONCE, REPLAY_LOOP, REPLAY_OFF = "один раз", "по кругу", "не показывать"
REPLAY_NAMES = (REPLAY_ONCE, REPLAY_LOOP, REPLAY_OFF)
WATCH_LEADER, WATCH_NONE = "leader", "none"

LEVELS = (
    ("лёгкий", 46, 0.4),
    ("обычный", 40, 0.7),
    ("сложный", 34, 1.0),
    ("адский", 30, 1.0),
)
LEVEL_NAMES = tuple(name for name, _, _ in LEVELS)
LEVEL_CUSTOM = "свой"
DEFAULT_LEVEL = 2
PICK_RADIUS = 40.0
GEN_CPPN, GEN_MODEL = "CPPN + эволюция", "обученная модель"

# Окно (2.0.0): 1280x720 по умолчанию, как было у pygame, но теперь его можно тянуть и
# разворачивать. Минимум - чтобы влезало в экран 1024x768 с панелью задач и заголовком.
MIN_W, MIN_H = 960, 640

STATS_H, STATS_LEAST = 172, 160
BADGE_INSET, BADGE_SCALE = 52, 1.8


def badge_reach(veh):
    """Половина ширины значка в пикселях: сколько места он отнимает у строк."""
    return float(np.abs(veh.stacked[:, 0]).max()) * BADGE_SCALE


GRAPH_H, GRAPH_LEAST = 54, 44
MESSAGE_FRAMES = 150

# Клавиши - коды Qt.Key. Буквы и цифры совпадают с кодами клавиш Windows (VK_R = Key_R),
# поэтому окно подставляет код физической клавиши: на русской раскладке R - это «К».
KEY_QUIT = int(Qt.Key.Key_Escape)
KEY_PAUSE = int(Qt.Key.Key_Space)
KEY_ROUND, KEY_TRACK, KEY_CAR = int(Qt.Key.Key_R), int(Qt.Key.Key_T), int(Qt.Key.Key_M)
KEY_SHOW, KEY_LEADER, KEY_NOBODY = int(Qt.Key.Key_S), int(Qt.Key.Key_L), int(Qt.Key.Key_N)
KEY_SPEED_1, KEY_SPEED_4 = int(Qt.Key.Key_1), int(Qt.Key.Key_4)


class Game:
    def __init__(self, seed=None, audio=None, size=(cfg.WINDOW_W, cfg.WINDOW_H), start=True):
        offscreen.app()                  # шрифтам и картинкам Qt нужно приложение
        self.width, self.height = int(size[0]), int(size[1])
        self.view = Rect(0, 0, self.width - cfg.PANEL_W, self.height)
        self.panel_rect = Rect(self.view.width, 0, cfg.PANEL_W, self.height)
        self.font = render.font(15)
        self.big = render.font(21, bold=True)
        self.track_layer = render.TrackLayer()

        self.rng = np.random.default_rng(seed)
        self.generator = trackgen.Generator() if trackgen.available() else None
        self.build_panel()
        self.level_shown = DEFAULT_LEVEL
        # Звук по умолчанию выключен: тесты и stress.py не должны гудеть. Окно игры
        # передаёт настоящий (main).
        self.audio = audio if audio is not None else sound.SoundBank(enabled=False)
        self.totals = stats.load_totals()
        self.message = ""
        self.message_left = 0
        self.paused = False
        self.best_brain = None
        self.watched = WATCH_LEADER
        self.hud = render.hud_rect(self.view)
        self.hud_grab = None
        self.hud_moved = False
        self.last_fitness = None
        self.rounds = 0
        self.running = True
        self.mouse = None                # где мышь - для подсветки кнопок
        self.splash_text = ""
        self.on_splash = None            # окно: показать заставку, пока строится трасса
        self.want_stats = False          # нажата кнопка «статистика»: окно откроет экран
        self.race = None
        if start:
            self.new_round(new_car=True)

    @property
    def started(self):
        return self.race is not None

    def build_panel(self):
        p = ui.Panel(self.panel_rect, self.font)
        p.skip(STATS_H, STATS_LEAST)
        p.graph("graph", GRAPH_H, GRAPH_LEAST)
        p.slider("pop_size", "популяция", 10, 120, cfg.POP_SIZE, integer=True)
        p.slider("generations", "поколений", 5, 120, cfg.MAX_GENERATIONS, integer=True)
        p.slider("mut_sigma", "мутация", 0.01, 0.6, cfg.MUT_SIGMA)
        p.slider("elite_frac", "элита", 0.0, 0.4, cfg.ELITE_FRAC)
        p.slider("width", "ширина трассы", 25, 80, cfg.TRACK_WIDTH, integer=True)
        p.slider("difficulty", "сложность", 0.0, 1.0, cfg.DIFFICULTY)
        p.slider("volume", "громкость", 0.0, 1.0, cfg.VOLUME)
        p.toggle("level", "уровень", LEVEL_NAMES, DEFAULT_LEVEL)
        names = (GEN_CPPN, GEN_MODEL) if self.generator else (GEN_CPPN,)
        p.toggle("generator", "генератор", names)
        # «скорость», а не «скорость показа», как было до 2.0.0: с «без отрисовки» справа
        # длинная подпись не влезала в 272 пикселя шрифтом на 10% крупнее (тест test_ui).
        p.toggle("speed", "скорость", SPEED_NAMES)
        p.toggle("brain", "мозг", BRAIN_NAMES)
        p.toggle("replay", "повтор заезда", REPLAY_NAMES)
        p.buttons([("track", "новая трасса"), ("car", "новая машина")])
        p.buttons([("show", "заезд"), ("pause", "пауза")])
        p.buttons([("save", "сохранить"), ("load", "загрузить")])
        p.buttons([("stats", "статистика")], per_row=1)
        self.ui = p

    def resize(self, width, height):
        """Окно сменило размер: поле - всё, кроме панели; панель раскладывается заново."""
        self.width, self.height = max(1, int(width)), max(1, int(height))
        self.view = Rect(0, 0, max(1, self.width - cfg.PANEL_W), self.height)
        self.panel_rect = Rect(self.view.width, 0, cfg.PANEL_W, self.height)
        self.ui.layout(self.panel_rect)
        if self.started:
            self.camera.rect = self.view
            self.camera.fit(self.track.lo, self.track.hi)
        # Окошко телеметрии, которое не двигали, остаётся в левом нижнем углу поля;
        # сдвинутое мышью - там, куда его поставили, только внутри поля.
        if self.hud_moved:
            self.hud.clamp_ip(self.view)
        else:
            self.hud = render.hud_rect(self.view)

    @property
    def speed(self):
        return cfg.SPEED_STEPS[self.ui.widgets["speed"].index]

    @property
    def keep_brain(self):
        return self.ui.widgets["brain"].index == 1

    def splash(self, text):
        self.splash_text = text
        if self.on_splash is not None:
            self.on_splash()

    def make_track(self):
        width, difficulty = self.ui.value("width"), self.ui.value("difficulty")
        if self.generator is not None and self.ui.value("generator") == GEN_MODEL:
            self.track_generator = "model"
            return self.generator.make_track(self.rng, width, difficulty)
        self.track_generator = "cppn"
        return track.evolve_track(self.rng, width, difficulty)

    def new_round(self, new_track=True, new_car=True):
        """Начать раунд заново, поменяв трассу, машинку или и то, и другое.

        Обе части раунда меняются по отдельности: посмотреть, как одна и та же
        машинка справляется с разными трассами, и как разные машинки справляются
        с одной трассой - это два разных опыта, и кнопка на каждый своя.
        """
        if new_track or not hasattr(self, "track"):
            self.splash("генерация трассы...")
            started = time.perf_counter()
            self.track = self.make_track()
            self.track_seconds = time.perf_counter() - started
            self.field = field.build_for_track(self.track)
            self.camera = render.Camera(self.view, self.track.lo, self.track.hi)
            self.splash_text = ""
        if new_car or not hasattr(self, "car"):
            self.car = car.random_car(self.rng)

        n = self.ui.value("pop_size")
        if self.keep_brain and self.best_brain is not None:
            self.brains = self.seed_from(self.best_brain, n)
        else:
            self.brains = evolution.random_population(n, brain.genome_size(), self.rng)

        self.gen = 0
        self.history = []
        self.path = []
        self.last_fitness = None
        self.rounds += 1
        self.paused = False
        self.state = TRAINING
        self.watched = WATCH_LEADER
        self.race = race.Race(self.track, self.field, self.car, self.brains)

    def say(self, text):
        self.message = text
        self.message_left = MESSAGE_FRAMES

    def seed_from(self, genome, n):
        seeded = np.tile(genome, (n, 1))
        return evolution.evolve(seeded, np.arange(n, dtype=float), self.rng,
                                elite_frac=1.0 / n, mut_sigma=self.ui.value("mut_sigma"))

    def level_name(self):
        """Уровень по ползункам: ширина и сложность сдвинуты руками - «свой»."""
        width, difficulty = self.ui.value("width"), self.ui.value("difficulty")
        for name, w, d in LEVELS:
            if width == w and abs(difficulty - d) < 1e-9:
                return name
        return LEVEL_CUSTOM

    def round_info(self):
        """Что запомнить о раунде для экрана статистики (stats.record_round)."""
        last_cp = max(1, self.track.n_checkpoints - 1)
        best_cp = max((s.best_cp for s in self.history), default=0)
        return {
            "level": self.level_name(),
            "generator": getattr(self, "track_generator", "cppn"),
            "pop": int(len(self.brains)),
            "progress": round(min(1.0, best_cp / last_cp), 4),
            "track": {"length": round(float(self.track.length), 1),
                      "radius": round(float(self.track.min_radius), 1),
                      "interest": round(float(self.track.variety), 2),
                      "width": round(float(self.track.width), 1),
                      "build_s": round(float(getattr(self, "track_seconds", 0.0)), 3)},
            "car": {"speed": round(float(self.car.max_speed), 1),
                    "steer": round(float(self.car.max_steer), 3),
                    "mass": round(float(self.car.mass), 3)},
        }

    def next_generation(self):
        fit = self.race.fitness()
        self.last_fitness = fit
        self.history.append(train.gen_stats(self.gen, self.race))
        self.best_brain = self.brains[int(np.argmax(fit))].copy()

        if self.race.n_finished or self.gen + 1 >= self.ui.value("generations"):
            self.totals = stats.record_round(self.totals, self.history, self.round_info())
            stats.save_totals(self.totals)
            if self.ui.value("replay") == REPLAY_OFF:
                self.state = DONE
            else:
                self.start_showcase()
            return

        self.brains = evolution.evolve(self.brains, fit, self.rng,
                                       elite_frac=self.ui.value("elite_frac"),
                                       mut_sigma=self.ui.value("mut_sigma"))
        self.gen += 1
        self.race = race.Race(self.track, self.field, self.car, self.brains)

    def start_showcase(self):
        self.state = SHOWCASE
        self.race = race.Race(self.track, self.field, self.car, self.best_brain[None, :])
        self.watched = 0
        self.path = [self.race.pos[0].copy()]

    def wrecks(self):
        return int((~self.race.alive & (self.race.finish_step < 0)).sum())

    def advance(self):
        if self.paused or self.state == DONE:
            self.audio.stop()
            return

        wrecks_before, finished_before = self.wrecks(), self.race.n_finished
        tracing = self.state == SHOWCASE
        for _ in range(self.speed or cfg.STEPS_PER_GEN):
            if self.race.done:
                break
            self.race.step()
            if tracing:
                self.path.append(self.race.pos[0].copy())
        self.play_sounds(wrecks_before, finished_before)

        if not self.race.done:
            return
        if self.state == TRAINING:
            self.next_generation()
        elif self.ui.value("replay") == REPLAY_LOOP:
            self.start_showcase()
        else:
            self.state = DONE

    def frame(self):
        """Один кадр игры: кнопки панели, шаги физики, отсчёт сообщения."""
        self.apply_buttons()
        self.advance()
        if self.message_left > 0:
            self.message_left -= 1

    def play_sounds(self, wrecks_before, finished_before):
        self.audio.set_volume(self.ui.value("volume"))
        if self.race.n_finished > finished_before:
            self.audio.play_finish()
        elif self.wrecks() > wrecks_before:
            self.audio.play_crash()

        if self.race.alive.any():
            self.audio.update(float(self.race.speed[self.race.alive].max()), self.car.max_speed)
        else:
            self.audio.stop()

    def key(self, code):
        if code == KEY_QUIT:
            self.running = False
        elif code == KEY_PAUSE:
            self.paused = not self.paused
        elif code == KEY_ROUND:
            self.new_round(new_track=True, new_car=True)
        elif code == KEY_TRACK:
            self.new_round(new_track=True, new_car=False)
        elif code == KEY_CAR:
            self.new_round(new_track=False, new_car=True)
        elif code == KEY_SHOW and self.best_brain is not None:
            self.start_showcase()
        elif code == KEY_LEADER:
            self.watched = WATCH_LEADER
        elif code == KEY_NOBODY:
            self.watched = WATCH_NONE
        elif KEY_SPEED_1 <= code <= KEY_SPEED_4:
            self.ui.widgets["speed"].index = code - KEY_SPEED_1

    def click_field(self, pos):
        if not self.view.collidepoint(pos):
            return
        watch = self.telemetry()
        if watch and self.hud.collidepoint(pos):
            if render.hud_close_rect(self.hud).collidepoint(pos):
                self.watched = WATCH_NONE
            else:
                self.hud_grab = (pos[0] - self.hud.left, pos[1] - self.hud.top)
            return

        world = self.camera.to_world(pos)
        index = self.race.nearest_to(world)
        if float(np.linalg.norm(self.race.pos[index] - world)) * self.camera.scale > PICK_RADIUS:
            self.watched = WATCH_NONE
            return
        # Повторный клик по той же машинке снимает слежение. Сравниваем с тем,
        # что сейчас показано, а не с self.watched: в режиме "за лидером" там
        # не номер, но на экране всё равно конкретная машинка.
        already = watch is not None and watch["index"] == index
        self.watched = WATCH_NONE if already else index

    def drag_hud(self, pos):
        if self.hud_grab is None:
            return
        self.hud.topleft = (pos[0] - self.hud_grab[0], pos[1] - self.hud_grab[1])
        self.hud.clamp_ip(self.view)
        self.hud_moved = True

    # --- мышь от окна (раньше - разбор событий pygame в run) -----------------------------

    def mouse_down(self, pos, button=1):
        if button == 1 and self.started:
            self.click_field(pos)
        self.ui.handle(ui.press(pos, button))

    def mouse_move(self, pos):
        self.mouse = pos
        if self.hud_grab is not None:
            self.drag_hud(pos)
        else:
            self.ui.handle(ui.move(pos))

    def mouse_up(self, pos, button=1):
        self.hud_grab = None
        self.ui.handle(ui.release(pos, button))

    def apply_level(self):
        index = self.ui.widgets["level"].index
        if index == self.level_shown:
            return
        self.level_shown = index
        _, width, difficulty = LEVELS[index]
        self.ui.widgets["width"].value = width
        self.ui.widgets["difficulty"].value = difficulty
        self.say(f"уровень {LEVEL_NAMES[index]}, трасса {width} px")

    def apply_buttons(self):
        self.apply_level()
        if self.ui.clicked("track"):
            self.new_round(new_track=True, new_car=False)
        if self.ui.clicked("car"):
            self.new_round(new_track=False, new_car=True)
        if self.ui.clicked("show") and self.best_brain is not None:
            self.start_showcase()
        if self.ui.clicked("pause"):
            self.paused = not self.paused
        if self.ui.clicked("save"):
            self.save_brains()
        if self.ui.clicked("load"):
            self.load_brains()
        if self.ui.clicked("stats"):
            self.want_stats = True

    def save_brains(self):
        fitness = self.last_fitness
        if fitness is None or len(fitness) != len(self.brains):
            fitness = np.zeros(len(self.brains))
        stats.save_brains(self.brains, fitness)
        self.say(f"мозги сохранены: {len(self.brains)} шт")

    def load_brains(self):
        genome = stats.best_brain()
        if genome is None:
            self.say("сохранения нет или оно не подходит")
            return
        self.best_brain = genome
        self.brains = self.seed_from(genome, self.ui.value("pop_size"))
        self.ui.widgets["brain"].index = 1
        # Счётчик поколений и историю надо обнулить вместе с популяцией. Иначе
        # загрузка посреди доигранного раунда оставляет gen выше предела, и
        # первый же заезд закрывает раунд: статистика получает лишний раунд, а
        # загруженный мозг - ни одного поколения обучения.
        self.gen = 0
        self.history = []
        self.path = []
        self.last_fitness = None
        self.state = TRAINING
        self.race = race.Race(self.track, self.field, self.car, self.brains)
        self.say("мозг загружен, обучение начато заново")

    def telemetry(self):
        if self.watched == WATCH_NONE:
            return None
        if self.watched == WATCH_LEADER:
            index = self.race.leader
        else:
            index = self.race.watch(self.watched)
            self.watched = index
        obs = self.race.observe()
        action = brain.forward(self.race.brains, obs)
        return {
            "index": index,
            "alive": bool(self.race.alive[index]),
            "finished": bool(self.race.finish_step[index] >= 0),
            "rays": obs[index, :cfg.N_RAYS],
            "speed": float(self.race.speed[index]),
            "steer": float(action[index, 0]),
            "throttle": float(action[index, 1]),
            "cp": f"{int(self.race.cp[index])}/{self.track.n_checkpoints - 1}",
        }

    # --- рисование ----------------------------------------------------------------------

    def draw_field(self, p):
        self.camera.fit(self.track.lo, self.track.hi)
        # Поле обрезает себя по своему прямоугольнику само, а не потому, что панель
        # потом закрасит свой (docs/bug-hunt.md: машинка у края красила панель).
        p.save()
        p.setClipRect(self.view.qrectf())

        self.track_layer.draw(p, self.camera, self.track)
        render.draw_checkpoints(p, self.camera, self.track, int(self.race.cp.max()))
        if self.state in (SHOWCASE, DONE):
            render.draw_path(p, self.camera, self.path, render._lighter(self.car.color))

        watch = self.telemetry()
        render.draw_cars(p, self.camera, self.race, self.car, watch["index"] if watch else None)
        if watch:
            render.draw_rays(p, self.camera, self.race.pos[watch["index"]],
                             self.race.angle[watch["index"]], watch["rays"] * cfg.RAY_MAX)
            render.draw_telemetry(p, self.hud, self.font, self.car, watch)
        p.restore()

    def draw_stats(self, p):
        x = self.panel_rect.left + self.ui.pad
        y = self.panel_rect.top + self.ui.top_pad

        width = self.panel_rect.width - 2 * self.ui.pad
        badge_column = self.panel_rect.width - BADGE_INSET - badge_reach(self.car) - self.ui.pad - 8

        def line(text, colour=render.TEXT, step=18, room=None):
            nonlocal y
            shown = render.fit_text(self.font, text, width if room is None else room)
            render.text(p, self.font, shown, colour, x, y)
            y += step

        def split_line(left, right, colour, step=18):
            """Левая половина у левого края, правая у правого.

            Так строка занимает всю ширину панели, а не половину, и растущие
            числа съедают запас между половинами, а не вылезают за край.
            """
            nonlocal y
            room = width - render.text_width(self.font, left) - 8
            if render.text_width(self.font, right) > room:
                left, right = stats.summary_parts(self.totals, short=True)
                room = width - render.text_width(self.font, left) - 8
            right = render.fit_text(self.font, right, room)
            render.text(p, self.font, left, colour, x, y)
            render.text(p, self.font, right, colour, x + width - render.text_width(self.font, right), y)
            y += step

        title = "ПАУЗА" if self.paused else STATE_NAME[self.state]
        colour = render.LEADER_RING if self.paused else render.TEXT
        render.text(p, self.big, title, colour, x, y)
        y += 30

        last = self.history[-1] if self.history else None
        line(f"раунд {self.rounds}   поколение {self.gen}", render.TEXT_DIM)
        line(f"живых {self.race.n_alive}/{self.race.n}   чекпоинт "
             f"{int(self.race.cp.max())}/{self.track.n_checkpoints - 1}")
        if last:
            line(f"лучший {last.best:.0f}   средний {last.mean:.0f}")
            time_part = f"   время {last.best_time:.1f} с" if last.best_time else ""
            line(f"доехало {last.finished}{time_part}",
                 render.LEADER_RING if last.finished else render.TEXT)
        else:
            line("лучший -   средний -")
            line("доехало -")

        # Сообщения (say) с 2.0.0 всплывают над полем целиком (window.GameView), а не
        # подменяют эту строку: здесь длинное обрезалось многоточием.
        left, right = stats.summary_parts(self.totals)
        if right:
            split_line(left, right, render.TEXT_DIM)
        else:
            line(left, render.TEXT_DIM)

        # Последние две строки делят место со значком машинки, поэтому им
        # отведена своя, укороченная ширина: значок стоит справа и наезжал бы
        # на хвост строки при любых значениях. Единицы убраны ради места -
        # "трасса" и "машина" и так говорят, что это за числа.
        line(f"трасса {self.track.length:.0f} радиус {self.track.min_radius:.0f}",
             render.TEXT_DIM, 17, badge_column)
        line(f"машина {self.car.max_speed:.0f} руль {self.car.max_steer:.2f}",
             render.TEXT_DIM, 17, badge_column)
        # Значок стоит по центру этих двух строк: они занимают 34 пикселя, и
        # масштаб 1.8 подобран так, чтобы значок в эту полосу помещался. При
        # 2.4 он был 47 пикселей высотой и задевал строку итогов сверху.
        render.draw_car_badge(p, self.car, (self.panel_rect.right - BADGE_INSET, y - 17), BADGE_SCALE)

    def draw_panel(self, p):
        p.fillRect(self.panel_rect.qrectf(), render.qc(render.PANEL_BG))
        self.draw_stats(p)
        self.ui.widgets["graph"].draw(p, self.history)
        self.ui.draw(p, self.mouse)

    def draw_splash(self, p):
        p.fillRect(Rect(0, 0, self.width, self.height).qrectf(), render.qc(render.BG))
        text = self.splash_text or "генерация трассы..."
        w = render.text_width(self.big, text)
        h = render.line_height(self.big)
        render.text(p, self.big, text, render.TEXT, self.view.centerx - w / 2, self.view.centery - h / 2)

    def paint(self, p):
        """Весь кадр: поле и панель (или заставка, пока строится трасса)."""
        if self.splash_text or not self.started:
            self.draw_splash(p)
            return
        p.fillRect(Rect(0, 0, self.width, self.height).qrectf(), render.qc(render.BG))
        self.draw_field(p)
        self.draw_panel(p)

    def snapshot(self, *draws):
        """Картинка окна: фон и то, что нарисуют draws (по умолчанию - весь кадр)."""
        img = render.canvas(self.width, self.height, render.BG)
        p = render.painter(img)
        for draw in draws or (self.paint,):
            draw(p)
        p.end()
        return img


def selftest(report_path, frames=400):
    """Раунд без окна на экране и отчёт в файл - проверка собранной игры.

    Идёт через настоящее окно (MainWindow), только offscreen: так проверяется, что в
    сборку попали Qt-виджеты, платформа offscreen, шрифты и звук."""
    offscreen.setup(force=True)
    offscreen.app()
    stats.use_sandbox("aicar-selftest-")

    import window
    from PySide6.QtGui import QFontInfo

    audio = sound.SoundBank(volume=0.0)           # настоящее устройство, если есть, но молча
    game = Game(seed=0, audio=audio)
    win = window.MainWindow(game)
    win.resize(cfg.WINDOW_W, cfg.WINDOW_H)
    game.ui.widgets["speed"].index = 2
    paint_ms = []
    for n in range(frames):
        game.frame()
        audio.pump()
        started = time.perf_counter()
        img = game.snapshot()
        paint_ms.append((time.perf_counter() - started) * 1000)
        if n % 100 == 0:
            win.view.grab()                       # тот же кадр через paintEvent окна
        if game.state == DONE:
            break
    win.show_stats()
    screen = win.stats_screen.grab()
    win.close()
    audio.close()

    info = QFontInfo(game.font)
    from PySide6.QtGui import QGuiApplication
    lines = [
        f"запуск из архива: {paths.frozen()}",
        f"папка данных:     {paths.data_dir()}",
        f"модель трасс:     {'есть' if game.generator else 'нет'} ({cfg.MODEL_FILE})",
        f"звук:             {'есть' if audio.rate else 'нет, ' + (audio.reason or 'нет устройства')}",
        f"формат звука:     {audio.rate} Гц, каналов {audio.channels} (QtMultimedia, int16)",
        f"платформа Qt:     {QGuiApplication.platformName()}",
        f"шрифт:            {info.family()} {info.pixelSize()} px"
        f"{'' if info.family() == render.FONT_FAMILY else ' - НЕ ' + render.FONT_FAMILY}",
        f"трасса:           длина {game.track.length:.0f}, чекпоинтов {game.track.n_checkpoints}",
        f"обучение:         поколений {len(game.history)}, состояние {game.state}",
        f"лучший результат: {game.history[-1].best:.0f}" if game.history else "лучший результат: -",
        f"кадр:             {np.mean(paint_ms):.1f} мс в среднем, кадров {len(paint_ms)}, "
        f"{img.width()}x{img.height()}",
        f"статистика:       экран {screen.width()}x{screen.height()}, раундов {game.totals['rounds']}",
        "OK",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="AI Car Racing")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--selftest", metavar="ФАЙЛ", help="прогнать раунд без окна и записать отчёт")
    args = ap.parse_args(argv)

    if args.selftest:
        for line in selftest(args.selftest):
            print(line)
        return 0

    if sys.platform == "win32":
        try:   # свой значок на панели задач, а не значок python.exe
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AiCar")
        except (OSError, AttributeError):
            pass
    from PySide6.QtWidgets import QApplication

    import theme
    import window
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AiCar")
    app.setApplicationDisplayName("AI Car Racing")
    theme.apply(app)
    game = Game(seed=args.seed, audio=sound.SoundBank(cfg.VOLUME), start=False)
    win = window.MainWindow(game, window_path=paths.user_file("window.json"))
    win.show_window()
    win.begin()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
