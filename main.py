import argparse
import os
import sys

import numpy as np
import pygame

import brain
import car
import config as cfg
import evolution
import field
import race
import render
import sound
import stats
import paths
import track
import trackgen
import train
import ui

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
DEFAULT_LEVEL = 2
PICK_RADIUS = 40.0
GEN_CPPN, GEN_MODEL = "CPPN + эволюция", "обученная модель"

STATS_H = 172
GRAPH_H = 54
MESSAGE_FRAMES = 150


class Game:
    def __init__(self, seed=None):
        pygame.init()
        pygame.display.set_caption("AI Car Racing")
        self.screen = pygame.display.set_mode((cfg.WINDOW_W, cfg.WINDOW_H))
        self.clock = pygame.time.Clock()
        self.view = pygame.Rect(0, 0, cfg.WINDOW_W - cfg.PANEL_W, cfg.WINDOW_H)
        self.panel_rect = pygame.Rect(self.view.width, 0, cfg.PANEL_W, cfg.WINDOW_H)
        self.font = pygame.font.SysFont("consolas", 15)
        self.big = pygame.font.SysFont("consolas", 21, bold=True)

        self.rng = np.random.default_rng(seed)
        self.generator = trackgen.Generator() if trackgen.available() else None
        self.build_panel()
        self.level_shown = DEFAULT_LEVEL
        self.audio = sound.SoundBank(self.ui.value("volume"))
        self.totals = stats.load_totals()
        self.message = ""
        self.message_left = 0
        self.paused = False
        self.best_brain = None
        self.watched = WATCH_LEADER
        self.hud = render.hud_rect(self.view)
        self.hud_grab = None
        self.last_fitness = None
        self.rounds = 0
        self.running = True
        self.new_round(new_car=True)

    def build_panel(self):
        p = ui.Panel(self.panel_rect, self.font)
        p.skip(STATS_H)
        p.graph("graph", GRAPH_H)
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
        p.toggle("speed", "скорость показа", SPEED_NAMES)
        p.toggle("brain", "мозг", BRAIN_NAMES)
        p.toggle("replay", "повтор заезда", REPLAY_NAMES)
        p.buttons([("round", "трасса и машина"), ("track", "только трасса")])
        p.buttons([("show", "заезд"), ("pause", "пауза")])
        p.buttons([("save", "сохранить"), ("load", "загрузить")])
        self.ui = p

    @property
    def speed(self):
        return cfg.SPEED_STEPS[self.ui.widgets["speed"].index]

    @property
    def keep_brain(self):
        return self.ui.widgets["brain"].index == 1

    def splash(self, text):
        self.screen.fill(render.BG)
        label = self.big.render(text, True, render.TEXT)
        self.screen.blit(label, label.get_rect(center=self.view.center))
        pygame.display.flip()

    def make_track(self):
        width, difficulty = self.ui.value("width"), self.ui.value("difficulty")
        if self.generator is not None and self.ui.value("generator") == GEN_MODEL:
            return self.generator.make_track(self.rng, width, difficulty)
        return track.evolve_track(self.rng, width, difficulty)

    def new_round(self, new_car=True):
        self.splash("генерация трассы...")
        self.track = self.make_track()
        self.field = field.build_for_track(self.track)
        if new_car or not hasattr(self, "car"):
            self.car = car.random_car(self.rng)
        self.camera = render.Camera(self.view, self.track.lo, self.track.hi)

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

    def next_generation(self):
        fit = self.race.fitness()
        self.last_fitness = fit
        self.history.append(train.gen_stats(self.gen, self.race))
        self.best_brain = self.brains[int(np.argmax(fit))].copy()

        if self.race.n_finished or self.gen + 1 >= self.ui.value("generations"):
            self.totals = stats.record_round(self.totals, self.history)
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
        if code == pygame.K_ESCAPE:
            self.running = False
        elif code == pygame.K_SPACE:
            self.paused = not self.paused
        elif code == pygame.K_r:
            self.new_round(new_car=True)
        elif code == pygame.K_t:
            self.new_round(new_car=False)
        elif code == pygame.K_s and self.best_brain is not None:
            self.start_showcase()
        elif code == pygame.K_l:
            self.watched = WATCH_LEADER
        elif code == pygame.K_n:
            self.watched = WATCH_NONE
        elif pygame.K_1 <= code <= pygame.K_4:
            self.ui.widgets["speed"].index = code - pygame.K_1

    def click_field(self, pos):
        if not self.view.collidepoint(pos):
            return
        if self.telemetry() and self.hud.collidepoint(pos):
            self.hud_grab = (pos[0] - self.hud.left, pos[1] - self.hud.top)
            return
        world = self.camera.to_world(pos)
        index = self.race.nearest_to(world)
        gap = float(np.linalg.norm(self.race.pos[index] - world))
        self.watched = index if gap * self.camera.scale <= PICK_RADIUS else WATCH_NONE

    def drag_hud(self, pos):
        if self.hud_grab is None:
            return
        self.hud.topleft = (pos[0] - self.hud_grab[0], pos[1] - self.hud_grab[1])
        self.hud.clamp_ip(self.view)

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
        if self.ui.clicked("round"):
            self.new_round(new_car=True)
        if self.ui.clicked("track"):
            self.new_round(new_car=False)
        if self.ui.clicked("show") and self.best_brain is not None:
            self.start_showcase()
        if self.ui.clicked("pause"):
            self.paused = not self.paused
        if self.ui.clicked("save"):
            self.save_brains()
        if self.ui.clicked("load"):
            self.load_brains()

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
        self.state = TRAINING
        self.race = race.Race(self.track, self.field, self.car, self.brains)
        self.say("мозг загружен, поколение перезапущено")

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

    def draw_field(self):
        self.camera.fit(self.track.lo, self.track.hi)

        render.draw_track(self.screen, self.camera, self.track)
        render.draw_checkpoints(self.screen, self.camera, self.track, int(self.race.cp.max()))
        if self.state in (SHOWCASE, DONE):
            render.draw_path(self.screen, self.camera, self.path, render._lighter(self.car.color))

        watch = self.telemetry()
        render.draw_cars(self.screen, self.camera, self.race, self.car,
                         watch["index"] if watch else None)
        if watch:
            render.draw_rays(self.screen, self.camera, self.race.pos[watch["index"]],
                             self.race.angle[watch["index"]], watch["rays"] * cfg.RAY_MAX)
            render.draw_telemetry(self.screen, self.hud, self.font, self.car, watch)

    def draw_stats(self):
        x = self.panel_rect.left + self.ui.pad
        y = self.panel_rect.top + self.ui.pad

        def line(text, colour=render.TEXT, step=18):
            nonlocal y
            self.screen.blit(self.font.render(text, True, colour), (x, y))
            y += step

        title = "ПАУЗА" if self.paused else STATE_NAME[self.state]
        colour = render.LEADER_RING if self.paused else render.TEXT
        self.screen.blit(self.big.render(title, True, colour), (x, y))
        y += 30

        last = self.history[-1] if self.history else None
        line(f"раунд {self.rounds}   поколение {self.gen}", render.TEXT_DIM)
        line(f"живых {self.race.n_alive}/{self.race.n}   чекпоинт "
             f"{int(self.race.cp.max())}/{self.track.n_checkpoints - 1}")
        if last:
            line(f"лучший {last.best:.0f}   средний {last.mean:.0f}")
            time = f"   время {last.best_time:.1f} с" if last.best_time else ""
            line(f"доехало {last.finished}{time}",
                 render.LEADER_RING if last.finished else render.TEXT)
        else:
            line("лучший -   средний -")
            line("доехало -")

        if self.message_left > 0:
            self.message_left -= 1
            line(self.message, render.LEADER_RING)
        else:
            line(stats.summary(self.totals), render.TEXT_DIM)

        line(f"трасса {self.track.length:.0f} px  поворот {self.track.min_radius:.0f}",
             render.TEXT_DIM, 17)
        line(f"машина {self.car.max_speed:.0f} px/s руль {self.car.max_steer:.2f}",
             render.TEXT_DIM, 17)
        render.draw_car_badge(self.screen, self.car, (self.panel_rect.right - 44, y - 14), 2.4)

    def draw_panel(self):
        pygame.draw.rect(self.screen, render.PANEL_BG, self.panel_rect)
        self.draw_stats()
        self.ui.widgets["graph"].draw(self.screen, self.history)
        self.ui.draw(self.screen)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.click_field(event.pos)
                    self.ui.handle(event)
                elif event.type == pygame.MOUSEMOTION and self.hud_grab is not None:
                    self.drag_hud(event.pos)
                else:
                    if event.type == pygame.MOUSEBUTTONUP:
                        self.hud_grab = None
                    self.ui.handle(event)
            self.apply_buttons()

            self.advance()
            self.screen.fill(render.BG)
            self.draw_field()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(cfg.FPS)
        self.audio.stop()
        pygame.quit()


def selftest(report_path, frames=400):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    game = Game(seed=0)
    game.ui.widgets["speed"].index = 2
    for _ in range(frames):
        game.advance()
        game.screen.fill(render.BG)
        game.draw_field()
        game.draw_panel()
        if game.state == DONE:
            break

    lines = [
        f"запуск из архива: {paths.frozen()}",
        f"папка данных:     {paths.data_dir()}",
        f"модель трасс:     {'есть' if game.generator else 'нет'} ({cfg.MODEL_FILE})",
        f"звук:             {'есть' if game.audio.enabled else 'нет, ' + (game.audio.reason or 'нет устройства')}",
        f"трасса:           длина {game.track.length:.0f}, чекпоинтов {game.track.n_checkpoints}",
        f"обучение:         поколений {len(game.history)}, состояние {game.state}",
        f"лучший результат: {game.history[-1].best:.0f}" if game.history else "лучший результат: -",
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
        return
    Game(seed=args.seed).run()


if __name__ == "__main__":
    main()
