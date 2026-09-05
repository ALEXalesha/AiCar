import numpy as np
import pygame
import pytest

import config as cfg
import sound


def test_engine_bank_has_one_sample_per_level():
    assert len(sound.engine_bank()) == cfg.ENGINE_LEVELS


def test_engine_samples_are_int16_and_not_clipped():
    for s in sound.engine_bank():
        assert s.dtype == np.int16
        assert 0 < np.abs(s).max() < 32767


def test_engine_sample_holds_a_whole_number_of_periods():
    for freq in (60.0, 137.0, 260.0):
        period = int(round(cfg.SAMPLE_RATE / freq))
        assert len(sound.engine_sample(freq)) % period == 0


def test_engine_loop_has_no_click_at_the_seam():
    for s in sound.engine_bank():
        seam = abs(int(s[0]) - int(s[-1]))
        inside = np.abs(np.diff(s.astype(np.int32))).max()
        assert seam <= inside * 2


def test_higher_levels_are_higher_pitched():
    bank = sound.engine_bank()
    crossings = [np.count_nonzero(np.diff(np.sign(s.astype(np.int32))) != 0) / len(s) for s in bank]
    assert crossings[-1] > crossings[0]


def test_engine_bank_is_reproducible():
    assert np.array_equal(sound.engine_bank()[3], sound.engine_bank()[3])


def test_crash_lasts_the_configured_time():
    expected = cfg.SAMPLE_RATE * cfg.CRASH_MS / 1000
    assert abs(len(sound.crash_sample()) - expected) < 2


def test_crash_decays():
    s = np.abs(sound.crash_sample().astype(np.int32))
    head, tail = s[:len(s) // 4], s[-len(s) // 4:]
    assert head.mean() > tail.mean() * 10


def test_finish_has_three_notes():
    expected = 3 * int(cfg.SAMPLE_RATE * sound.FINISH_NOTE_MS / 1000)
    assert len(sound.finish_sample()) == expected


def test_finish_notes_rise_in_pitch():
    s = sound.finish_sample().astype(np.int32)
    third = len(s) // 3
    rate = [np.count_nonzero(np.diff(np.sign(s[i * third:(i + 1) * third])) != 0) for i in range(3)]
    assert rate[0] < rate[1] < rate[2]


def test_all_samples_stay_inside_int16():
    for s in list(sound.engine_bank()) + [sound.crash_sample(), sound.finish_sample()]:
        assert s.dtype == np.int16
        assert np.abs(s).max() <= 32767


def test_fit_channels_keeps_mono_as_is():
    mono = np.arange(10, dtype=np.int16)
    assert sound.fit_channels(mono, 1) is mono


def test_fit_channels_doubles_a_mono_sample_for_stereo():
    mono = np.arange(10, dtype=np.int16)
    stereo = sound.fit_channels(mono, 2)
    assert stereo.shape == (10, 2)
    assert np.array_equal(stereo[:, 0], mono) and np.array_equal(stereo[:, 1], mono)


def test_soundbank_works_on_a_stereo_mixer():
    pygame.mixer.quit()
    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    bank = sound.SoundBank(0.4)
    assert bank.enabled, bank.reason
    assert bank.channels == 2 and bank.rate == 44100
    assert len(bank.engine) == cfg.ENGINE_LEVELS


def test_soundbank_always_matches_the_real_mixer_format():
    for rate, asked in ((22050, 1), (44100, 2), (48000, 2)):
        pygame.mixer.quit()
        pygame.mixer.init(frequency=rate, size=-16, channels=asked)
        real_rate, _, real_channels = pygame.mixer.get_init()
        bank = sound.SoundBank(0.4)
        assert bank.enabled, bank.reason
        assert (bank.rate, bank.channels) == (real_rate, abs(real_channels))


def test_soundbank_survives_a_missing_audio_device(monkeypatch):
    def refuse(*args, **kwargs):
        raise pygame.error("нет аудиоустройства")

    monkeypatch.setattr(pygame.mixer, "init", refuse)
    monkeypatch.setattr(pygame.mixer, "get_init", lambda: None)
    bank = sound.SoundBank()
    assert not bank.enabled
    assert bank.reason
    bank.update(100.0, 200.0)
    bank.play_crash()
    bank.play_finish()
    bank.set_volume(0.5)
    bank.stop()


def test_volume_is_clamped(monkeypatch):
    monkeypatch.setattr(pygame.mixer, "get_init", lambda: None)
    monkeypatch.setattr(pygame.mixer, "init", lambda *a, **k: (_ for _ in ()).throw(pygame.error("нет")))
    bank = sound.SoundBank()
    bank.set_volume(5.0)
    assert bank.volume == 1.0
    bank.set_volume(-1.0)
    assert bank.volume == 0.0
