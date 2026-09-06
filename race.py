import numpy as np

import brain
import config as cfg
import sensors

CHECKPOINT_REWARD = 100.0
FINISH_BONUS_PER_STEP = 0.5


class Race:
    def __init__(self, trk, fld, vehicle, brains):
        self.track = trk
        self.field = fld
        self.car = vehicle
        self.brains = brains

        n = len(brains)
        self.n = n
        self.pos = np.tile(trk.start_pos, (n, 1)).astype(float)
        self.angle = np.full(n, trk.start_angle)
        self.speed = np.zeros(n)
        self.alive = np.ones(n, dtype=bool)
        self.cp = np.zeros(n, dtype=np.int64)
        self.idle = np.zeros(n, dtype=np.int64)
        self.finish_step = np.full(n, -1, dtype=np.int64)
        self.steps = 0
        self.last_cp = trk.n_checkpoints - 1

    def observe(self):
        rays = sensors.cast(self.pos, self.angle, self.field)
        speed = (self.speed / self.car.max_speed)[:, None]
        return np.concatenate([rays / cfg.RAY_MAX, speed], axis=1)

    def _drive(self, steer, throttle):
        live = self.alive.astype(float)
        self.speed += (throttle * self.car.accel - self.speed * cfg.DRAG) * cfg.DT * live
        np.clip(self.speed, 0.0, self.car.max_speed, out=self.speed)
        self.angle += steer * self.car.max_steer * (self.speed / self.car.max_speed) * cfg.DT * live
        heading = np.stack([np.cos(self.angle), np.sin(self.angle)], axis=1)
        self.pos += heading * (self.speed * cfg.DT * live)[:, None]

    def _take_checkpoints(self):
        nxt = np.minimum(self.cp + 1, self.last_cp)
        ahead = np.sum((self.pos - self.track.cp_mid[nxt]) * self.track.cp_dir[nxt], axis=1)
        took = self.alive & (ahead >= 0.0) & (self.cp < self.last_cp)
        self.cp = self.cp + took
        self.idle = np.where(took, 0, self.idle + self.alive)

    def step(self):
        action = brain.forward(self.brains, self.observe())
        self._drive(action[:, 0], action[:, 1])
        self._take_checkpoints()

        crashed = self.alive & (self.field.sample(self.pos) < self.car.half_width)
        stalled = self.alive & (self.idle >= cfg.IDLE_LIMIT)
        arrived = self.alive & (self.cp >= self.last_cp) & (self.finish_step < 0)

        self.finish_step = np.where(arrived, self.steps, self.finish_step)
        self.alive &= ~(crashed | stalled | arrived)
        self.steps += 1

    def fitness(self):
        nxt = np.minimum(self.cp + 1, self.last_cp)
        here = self.track.cp_mid[self.cp]
        leg = self.track.cp_mid[nxt] - here
        along = np.sum(leg * (self.pos - here), axis=1) / (np.sum(leg * leg, axis=1) + 1e-9)
        score = (self.cp + np.clip(along, 0.0, 1.0)) * CHECKPOINT_REWARD
        bonus = (cfg.STEPS_PER_GEN - self.finish_step) * FINISH_BONUS_PER_STEP
        return score + (self.finish_step >= 0) * bonus

    @property
    def done(self):
        return not self.alive.any() or self.steps >= cfg.STEPS_PER_GEN

    @property
    def n_alive(self):
        return int(self.alive.sum())

    @property
    def n_finished(self):
        return int((self.finish_step >= 0).sum())

    @property
    def leader(self):
        """Лучшая из едущих; если все стоят - лучшая вообще.

        Считать лидером просто лучшего по приспособленности нельзя: как только
        самая продвинувшаяся машинка разбивается, слежение прилипает к обломкам
        и показывает застывшие цифры, пока остальные ещё едут. Разбившаяся
        остаётся лидером только когда живых не осталось - в конце заезда и в
        показательном заезде это как раз то, что нужно показать.
        """
        fit = self.fitness()
        if self.alive.any():
            alive = np.flatnonzero(self.alive)
            return int(alive[int(np.argmax(fit[alive]))])
        return int(np.argmax(fit))

    def nearest_alive(self, index):
        alive = np.flatnonzero(self.alive)
        if not len(alive):
            return index
        gap = np.linalg.norm(self.pos[alive] - self.pos[index], axis=1)
        return int(alive[int(np.argmin(gap))])

    def nearest_to(self, point):
        return int(np.argmin(np.linalg.norm(self.pos - np.asarray(point, dtype=float), axis=1)))

    def watch(self, index):
        if index is None or not 0 <= index < self.n:
            return self.leader
        return index if self.alive[index] else self.nearest_alive(index)


def run(trk, fld, vehicle, brains):
    race = Race(trk, fld, vehicle, brains)
    while not race.done:
        race.step()
    return race
