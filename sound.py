import numpy as np
import pygame

import config as cfg

ENGINE_PERIODS = 12
ENGINE_AMP = 0.28
ENGINE_NOISE = 0.06
ENGINE_SMOOTH = 3
SWITCH_FADE_MS = 45

CRASH_TAU = 0.028
CRASH_THUMP_HZ = 70.0
FINISH_NOTES = (523.0, 659.0, 784.0)
FINISH_NOTE_MS = 130


def _to_int16(wave):
    return (np.clip(wave, -1.0, 1.0) * 32767.0).astype(np.int16)


def _smooth(wave, w):
    if w <= 1:
        return wave
    ext = np.concatenate([wave[-w:], wave, wave[:w]])
    return np.convolve(ext, np.ones(w) / w, mode="same")[w:-w]


def fit_channels(sample, channels):
    if channels <= 1:
        return sample
    return np.repeat(sample[:, None], channels, axis=1)


def mixer_format():
    ready = pygame.mixer.get_init()
    if ready is None:
        pygame.mixer.init()
        ready = pygame.mixer.get_init()
    rate, _, channels = ready
    return int(rate), int(abs(channels))


def engine_sample(freq, rate=None, rng=None):
    rate = cfg.SAMPLE_RATE if rate is None else rate
    rng = np.random.default_rng(0) if rng is None else rng

    period = int(round(rate / freq))
    n = period * ENGINE_PERIODS
    phase = (np.arange(n) % period) / period

    saw = 2.0 * phase - 1.0
    second = 0.35 * np.sin(4.0 * np.pi * phase)
    wave = _smooth(saw + second, ENGINE_SMOOTH)
    wave += ENGINE_NOISE * _smooth(rng.normal(0.0, 1.0, n), 5)
    return _to_int16(ENGINE_AMP * wave / np.abs(wave).max())


def engine_bank(levels=None, rate=None):
    levels = cfg.ENGINE_LEVELS if levels is None else levels
    rng = np.random.default_rng(0)
    freqs = np.linspace(cfg.ENGINE_F_MIN, cfg.ENGINE_F_MAX, levels)
    return [engine_sample(f, rate, rng) for f in freqs]


def crash_sample(rate=None):
    rate = cfg.SAMPLE_RATE if rate is None else rate
    n = int(rate * cfg.CRASH_MS / 1000)
    t = np.arange(n) / rate
    decay = np.exp(-t / CRASH_TAU)
    rng = np.random.default_rng(1)
    noise = _smooth(rng.normal(0.0, 1.0, n), 3)
    thump = np.sin(2.0 * np.pi * CRASH_THUMP_HZ * t)
    return _to_int16(0.55 * decay * (0.7 * noise + 0.6 * thump))


def finish_sample(rate=None):
    rate = cfg.SAMPLE_RATE if rate is None else rate
    n = int(rate * FINISH_NOTE_MS / 1000)
    t = np.arange(n) / rate
    envelope = np.minimum(1.0, 40.0 * t) * np.exp(-t / 0.09)
    notes = [envelope * np.sin(2.0 * np.pi * f * t) for f in FINISH_NOTES]
    return _to_int16(0.4 * np.concatenate(notes))


class SoundBank:
    def __init__(self, volume=None):
        self.volume = cfg.VOLUME if volume is None else volume
        self.level = -1
        self.enabled = False
        self.reason = ""
        try:
            rate, channels = mixer_format()
            self.rate, self.channels = rate, channels
            make = pygame.sndarray.make_sound
            self.engine = [make(fit_channels(s, channels)) for s in engine_bank(rate=rate)]
            self.crash = make(fit_channels(crash_sample(rate), channels))
            self.finish = make(fit_channels(finish_sample(rate), channels))
            self.motor = pygame.mixer.Channel(0)
            self.effects = pygame.mixer.Channel(1)
        except (pygame.error, AttributeError, ValueError) as problem:
            self.reason = str(problem) or type(problem).__name__
            return
        self.enabled = True
        self.set_volume(self.volume)

    def set_volume(self, volume):
        self.volume = float(np.clip(volume, 0.0, 1.0))
        if self.enabled:
            self.motor.set_volume(self.volume * 0.5)
            self.effects.set_volume(self.volume)

    def update(self, speed, max_speed):
        if not self.enabled or self.volume <= 0.0:
            self.stop()
            return
        share = float(np.clip(speed / max(max_speed, 1e-9), 0.0, 1.0))
        level = int(round(share * (len(self.engine) - 1)))
        if level != self.level:
            self.level = level
            self.motor.play(self.engine[level], loops=-1, fade_ms=SWITCH_FADE_MS)

    def play_crash(self):
        if self.enabled and self.volume > 0.0:
            self.effects.play(self.crash)

    def play_finish(self):
        if self.enabled and self.volume > 0.0:
            self.effects.play(self.finish)

    def stop(self):
        if self.enabled:
            self.motor.stop()
        self.level = -1
