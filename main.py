import numpy as np
import pygame

import brain
import car
import config as cfg
import evolution
import field
import race
import render
import track
import train

TRACK_GEN = "track"
TRAINING = "training"
SHOWCASE = "showcase"
DONE = "done"

HELP = [
    "R  новый раунд",
    "T  новая трасса",
    "S  показательный заезд",
    "N  мозг с нуля / продолжить",
    "1 2 3 4  скорость",
    "пробел  пауза",
]


class Game:
    def __init__(self, seed=None):
        pygame.init()
        pygame.display.set_caption("AI Car Racing")
        self.screen = pygame.display.set_mode((cfg.WINDOW_W, cfg.WINDOW_H))
        self.clock = pygame.time.Clock()
        self.view = pygame.Rect(0, 0, cfg.WINDOW_W - cfg.PANEL_W, cfg.WINDOW_H)
        self.panel = pygame.Rect(self.view.width, 0, cfg.PANEL_W, cfg.WINDOW_H)
        self.font = pygame.font.SysFont("consolas", 15)
        self.big = pygame.font.SysFont("consolas", 21, bold=True)

        self.rng = np.random.default_rng(seed)
        self.settings = {
            "pop_size": cfg.POP_SIZE,
            "generations": cfg.MAX_GENERATIONS,
            "mut_sigma": cfg.MUT_SIGMA,
            "elite_frac": cfg.ELITE_FRAC,
            "width": cfg.TRACK_WIDTH,
            "difficulty": cfg.DIFFICULTY,
        }
        self.speed_idx = 0
        self.paused = False
        self.keep_brain = False
        self.best_brain = None
        self.rounds = 0
        self.running = True
        self.new_round(new_car=True)

    @property
    def speed(self):
        return cfg.SPEED_STEPS[self.speed_idx]

    def splash(self, text):
        self.screen.fill(render.BG)
        label = self.big.render(text, True, render.TEXT)
        self.screen.blit(label, label.get_rect(center=self.view.center))
        pygame.display.flip()

    def new_round(self, new_car=True):
        self.splash("генерация трассы...")
        self.track = track.evolve_track(self.rng, self.settings["width"], self.settings["difficulty"])
        self.field = field.build_for_track(self.track)
        if new_car or not hasattr(self, "car"):
            self.car = car.random_car(self.rng)
        self.camera = render.Camera(self.view, self.track.lo, self.track.hi)

        n = self.settings["pop_size"]
        if self.keep_brain and self.best_brain is not None:
            self.brains = np.tile(self.best_brain, (n, 1))
            self.brains[1:] = evolution.evolve(self.brains, np.arange(n, dtype=float), self.rng,
                                               elite_frac=1.0 / n)[1:]
        else:
            self.brains = evolution.random_population(n, brain.genome_size(), self.rng)

        self.gen = 0
        self.history = []
        self.rounds += 1
        self.state = TRAINING
        self.race = race.Race(self.track, self.field, self.car, self.brains)

    def next_generation(self):
        fit = self.race.fitness()
        self.history.append(train.gen_stats(self.gen, self.race))
        self.best_brain = self.brains[int(np.argmax(fit))].copy()

        if self.race.n_finished or self.gen + 1 >= self.settings["generations"]:
            self.start_showcase()
            return

        self.brains = evolution.evolve(self.brains, fit, self.rng,
                                       elite_frac=self.settings["elite_frac"],
                                       mut_sigma=self.settings["mut_sigma"])
        self.gen += 1
        self.race = race.Race(self.track, self.field, self.car, self.brains)

    def start_showcase(self):
        self.state = SHOWCASE
        self.race = race.Race(self.track, self.field, self.car, self.best_brain[None, :])

    def advance(self):
        if self.paused:
            return
        steps = self.speed or cfg.STEPS_PER_GEN
        for _ in range(steps):
            if self.race.done:
                break
            self.race.step()

        if not self.race.done:
            return
        if self.state == TRAINING:
            self.next_generation()
        elif self.state == SHOWCASE:
            self.state = DONE

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
        elif code == pygame.K_n:
            self.keep_brain = not self.keep_brain
        elif pygame.K_1 <= code <= pygame.K_4:
            self.speed_idx = code - pygame.K_1

    def draw_field(self):
        if self.state in (SHOWCASE, DONE):
            self.camera.follow(self.race.pos[0], render.SHOWCASE_SCALE)
        else:
            self.camera.fit(self.track.lo, self.track.hi)

        render.draw_track(self.screen, self.camera, self.track)
        render.draw_checkpoints(self.screen, self.camera, self.track, int(self.race.cp.max()))

        leader = self.race.leader
        render.draw_cars(self.screen, self.camera, self.race, self.car, leader)
        if self.race.alive[leader]:
            rays = self.race.observe()[leader, :cfg.N_RAYS] * cfg.RAY_MAX
            render.draw_rays(self.screen, self.camera, self.race.pos[leader],
                             self.race.angle[leader], rays)

    def draw_graph(self, top, height):
        box = pygame.Rect(self.panel.left + 14, top, cfg.PANEL_W - 28, height)
        pygame.draw.rect(self.screen, render.BG, box)
        pygame.draw.rect(self.screen, render.MIDLINE, box, 1)
        if len(self.history) < 2:
            return
        best = np.array([s.best for s in self.history])
        mean = np.array([s.mean for s in self.history])
        top_value = max(best.max() * 1.08, 1.0)
        xs = np.linspace(box.left, box.right, len(best))
        for values, colour in ((mean, render.TEXT_DIM), (best, render.LEADER_RING)):
            pts = [(x, box.bottom - v / top_value * box.height) for x, v in zip(xs, values)]
            pygame.draw.lines(self.screen, colour, False, pts, 2)

    def draw_panel(self):
        pygame.draw.rect(self.screen, render.PANEL_BG, self.panel)
        x = self.panel.left + 14
        y = 16

        def line(text, colour=render.TEXT, step=20):
            nonlocal y
            self.screen.blit(self.font.render(text, True, colour), (x, y))
            y += step

        state = {TRAINING: "обучение", SHOWCASE: "показательный заезд",
                 DONE: "заезд окончен", TRACK_GEN: "генерация"}[self.state]
        self.screen.blit(self.big.render(state, True, render.TEXT), (x, y))
        y += 34
        if self.paused:
            line("ПАУЗА", render.LEADER_RING)

        line(f"раунд {self.rounds}   поколение {self.gen}", render.TEXT_DIM)
        line(f"живых    {self.race.n_alive} / {self.race.n}")
        line(f"чекпоинт {int(self.race.cp.max())} / {self.track.n_checkpoints - 1}")
        if self.history:
            last = self.history[-1]
            line(f"лучший   {last.best:.0f}")
            line(f"средний  {last.mean:.0f}")
            line(f"доехало  {last.finished}")
            if last.best_time:
                line(f"время    {last.best_time:.1f} с", render.LEADER_RING)
        y += 8

        self.draw_graph(y, 110)
        y += 124

        line("трасса", render.TEXT_DIM)
        line(f"длина {self.track.length:.0f}  поворот {self.track.min_radius:.0f}")
        line(f"ширина {self.track.width:.0f}  интерес {self.track.variety:.0f}")
        y += 8
        line("машинка", render.TEXT_DIM)
        line(f"скорость {self.car.max_speed:.0f}  руль {self.car.max_steer:.2f}")
        line(f"масса {self.car.mass:.2f}  {self.car.length:.0f} на {self.car.width:.0f}")
        render.draw_car_badge(self.screen, self.car, (self.panel.right - 40, y - 30), 1.6)
        y += 8

        speed = "без отрисовки" if self.speed == 0 else f"x{self.speed}"
        line(f"скорость показа {speed}", render.TEXT_DIM)
        line(f"мозг {'продолжить' if self.keep_brain else 'с нуля'}", render.TEXT_DIM)
        y += 8
        for row in HELP:
            line(row, render.TEXT_DIM, 18)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.key(event.key)

            self.advance()
            self.screen.fill(render.BG)
            self.draw_field()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(cfg.FPS)
        pygame.quit()


def main():
    Game().run()


if __name__ == "__main__":
    main()
