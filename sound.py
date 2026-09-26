"""Звук: синтез на numpy и проигрывание через QtMultimedia (с 2.0.0; до того - pygame.mixer).

Синтез (`engine_sample`, `crash_sample`, `finish_sample`) не изменился. Играет `Mixer` - один
поток отсчётов на всё, и `SoundBank` дописывает его в QAudioSink каждый тик окна. Без
звуковой карты, без QtMultimedia или если устройство не открылось - `enabled = False`,
игра молчит и не падает.
"""
import numpy as np

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


class Mixer:
    """Всё, что играет, одним потоком отсчётов (2.0.0). Чистый numpy, без устройства.

    Мотор - двенадцать петель разной высоты (`engine_bank`). Уровень меняется по скорости
    лидера, и при смене старая петля затухает, а новая нарастает за SWITCH_FADE_MS -
    одновременно, без провала и без щелчка. У pygame смена была обрывом старой петли.
    Каждая петля помнит, где остановилась: при возврате на уровень волна продолжается,
    а не начинается заново. Удар и финиш смешиваются поверх мотора и играют один раз.

    Громкость мотора - половина общей, эффектов - вся (как каналы у pygame-версии).
    Громкость тоже меняется не скачком, а с той же скоростью, что затухание петель.
    Все переходы считаются по отсчётам, а не по кускам: нарезка `render` на куски любой
    длины даёт тот же звук, что один большой кусок.
    """

    def __init__(self, rate, engine, crash, finish):
        self.rate = int(rate)
        self.engine = [np.asarray(s, np.float64) / 32767.0 for s in engine]
        self.samples = {"crash": np.asarray(crash, np.float64) / 32767.0,
                        "finish": np.asarray(finish, np.float64) / 32767.0}
        self.pos = np.zeros(len(self.engine), dtype=np.int64)
        self.gain = np.zeros(len(self.engine))
        self.level = -1
        self.step = 1.0 / max(1, int(self.rate * SWITCH_FADE_MS / 1000))
        self.effects = []
        self.volume = 0.0
        self._shown_volume = 0.0

    def set_volume(self, volume):
        self.volume = float(np.clip(volume, 0.0, 1.0))

    def set_level(self, level):
        """Какую петлю мотора играть; -1 - заглушить мотор (плавно)."""
        self.level = int(level) if 0 <= level < len(self.engine) else -1

    def play(self, name):
        self.effects.append([self.samples[name], 0])

    def playing_levels(self):
        return [i for i in range(len(self.engine)) if self.gain[i] > 0.0]

    def _engine(self, n):
        out = np.zeros(n)
        t = np.arange(1, n + 1)
        for i, wave in enumerate(self.engine):
            target = 1.0 if i == self.level else 0.0
            g0 = self.gain[i]
            if g0 == 0.0 and target == 0.0:
                continue
            ramp = np.clip(g0 + np.sign(target - g0) * self.step * t, min(g0, target), max(g0, target))
            idx = (self.pos[i] + np.arange(n)) % len(wave)
            out += ramp * wave[idx]
            self.pos[i] = (self.pos[i] + n) % len(wave)
            self.gain[i] = ramp[-1]
        return out

    def _effects(self, n):
        out = np.zeros(n)
        alive = []
        for effect in self.effects:
            wave, at = effect
            piece = wave[at:at + n]
            out[:len(piece)] += piece
            effect[1] = at + len(piece)
            if effect[1] < len(wave):
                alive.append(effect)
        self.effects = alive
        return out

    def render(self, n):
        """Следующие n отсчётов, int16 моно."""
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.int16)
        v0, v1 = self._shown_volume, self.volume
        volume = np.clip(v0 + np.sign(v1 - v0) * self.step * np.arange(1, n + 1), min(v0, v1), max(v0, v1))
        self._shown_volume = float(volume[-1])
        mix = volume * (0.5 * self._engine(n) + self._effects(n))
        return _to_int16(mix)


AHEAD_MS = 70          # сколько звука держать в устройстве наперёд: тик окна - 16 мс
BUFFER_MS = 250        # сколько устройство вообще готово принять
FORMATS = ((None, None), (48000, 2), (44100, 2), (44100, 1), (22050, 1))


def open_sink():
    """Открыть устройство вывода по умолчанию в формате int16. (sink, io, rate, channels)
    или исключение с понятным текстом. Формат берётся у открытого устройства, а не тот,
    что просили: в 1.0 pygame отдал стерео на просьбу о моно, и звук молчал (docs/modules/sound.md)."""
    from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSink, QMediaDevices

    device = QMediaDevices.defaultAudioOutput()
    if device.isNull():
        raise RuntimeError("нет устройства вывода звука")
    preferred = device.preferredFormat()
    for rate, channels in FORMATS:
        fmt = QAudioFormat()
        fmt.setSampleRate(rate or preferred.sampleRate() or 44100)
        fmt.setChannelCount(channels or min(2, max(1, preferred.channelCount())))
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if device.isFormatSupported(fmt):
            break
    else:
        raise RuntimeError("устройство не принимает 16-битный звук")
    sink = QAudioSink(device, fmt)
    sink.setBufferSize(int(fmt.sampleRate() * fmt.channelCount() * 2 * BUFFER_MS / 1000))
    io = sink.start()
    if io is None or sink.error() != QAudio.Error.NoError:
        sink.stop()
        raise RuntimeError(f"устройство не открылось ({sink.error().name})")
    real = sink.format()
    return sink, io, real.sampleRate(), real.channelCount()


class SoundBank:
    """Звук игры. Снаружи - как в pygame-версии: update(скорость), play_crash, play_finish,
    stop, set_volume; плюс pump() - дописать в устройство то, что микшер успел насчитать
    (окно зовёт его по таймеру, и когда игра стоит, тоже: иначе затухание не доиграет)."""

    def __init__(self, volume=None, enabled=True):
        self.volume = float(np.clip(cfg.VOLUME if volume is None else volume, 0.0, 1.0))
        self.level = -1
        self.enabled = False
        self.reason = ""
        self.rate, self.channels = 0, 0
        self.mixer = None
        self.sink = self.io = None
        if not enabled:
            self.reason = "выключен"
            return
        try:
            self.sink, self.io, self.rate, self.channels = open_sink()
            self.mixer = Mixer(self.rate, engine_bank(rate=self.rate), crash_sample(self.rate),
                               finish_sample(self.rate))
        except Exception as problem:  # noqa: BLE001 - нет QtMultimedia, устройства, формата
            self.reason = str(problem) or type(problem).__name__
            self.close()
            return
        self.enabled = True
        self.set_volume(self.volume)

    @property
    def engine(self):
        return self.mixer.engine if self.mixer is not None else []

    def set_volume(self, volume):
        self.volume = float(np.clip(volume, 0.0, 1.0))
        if self.enabled:
            self.mixer.set_volume(self.volume)

    def update(self, speed, max_speed):
        if not self.enabled or self.volume <= 0.0:
            self.stop()
            return
        share = float(np.clip(speed / max(max_speed, 1e-9), 0.0, 1.0))
        level = int(round(share * (len(self.engine) - 1)))
        if level != self.level:
            self.level = level
            self.mixer.set_level(level)

    def play_crash(self):
        if self.enabled and self.volume > 0.0:
            self.mixer.play("crash")

    def play_finish(self):
        if self.enabled and self.volume > 0.0:
            self.mixer.play("finish")

    def stop(self):
        if self.enabled:
            self.mixer.set_level(-1)
        self.level = -1

    def pump(self):
        """Дописать звук так, чтобы в устройстве было AHEAD_MS наперёд. Вернуть, сколько
        отсчётов записано."""
        if not self.enabled:
            return 0
        try:
            frame = 2 * self.channels
            queued = (self.sink.bufferSize() - self.sink.bytesFree()) // frame
            want = int(self.rate * AHEAD_MS / 1000) - queued
            room = self.sink.bytesFree() // frame
            n = min(want, room)
            if n <= 0:
                return 0
            self.io.write(fit_channels(self.mixer.render(n), self.channels).tobytes())
            return n
        except Exception as problem:  # noqa: BLE001 - устройство пропало посреди игры
            self.reason = str(problem) or type(problem).__name__
            self.close()
            return 0

    def close(self):
        if self.sink is not None:
            try:
                self.sink.stop()
            except Exception:  # noqa: BLE001
                pass
        self.sink = self.io = None
        self.enabled = False
