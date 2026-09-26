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


# --- микшер (2.0.0): мотор с перекрёстным затуханием и эффекты поверх ------------------

RATE = 22050


def mixer(volume=1.0):
    m = sound.Mixer(RATE, sound.engine_bank(rate=RATE), sound.crash_sample(RATE),
                    sound.finish_sample(RATE))
    m.set_volume(volume)
    return m


def test_mixer_renders_exactly_what_was_asked():
    out = mixer().render(1234)
    assert out.dtype == np.int16 and out.shape == (1234,)


def test_mixer_is_silent_without_the_engine():
    assert not mixer().render(4000).any()


def test_engine_plays_after_a_level_is_set():
    m = mixer()
    m.set_level(3)
    assert np.abs(m.render(4000).astype(np.int32)).max() > 1000


def test_chunks_join_without_a_seam():
    whole, parts = mixer(), mixer()
    for m in (whole, parts):
        m.set_level(2)
    a = whole.render(3000)
    b = np.concatenate([parts.render(n) for n in (1, 299, 700, 2000)])
    assert np.array_equal(a, b)


def test_switching_the_level_crossfades_without_a_dip():
    """Шаг соседних отсчётов тут ничего не докажет: мотор - пила, у неё и так обрыв
    каждый период. Проверяется сам переход: старая петля затухает, новая нарастает,
    вместе они всё время дают полную громкость."""
    m = mixer()
    m.set_level(2)
    m.render(RATE)
    m.set_level(9)
    fade = int(RATE * sound.SWITCH_FADE_MS / 1000)
    shares = []
    for _ in range(4):
        m.render(fade // 4)
        shares.append((m.gain[2], m.gain[9]))
    assert all(abs(old + new - 1.0) < 1e-9 for old, new in shares)
    olds = [old for old, _ in shares]
    assert olds == sorted(olds, reverse=True) and 0.1 < olds[1] < 0.9


def test_the_old_level_fades_out_in_the_switch_time():
    m = mixer()
    m.set_level(2)
    m.render(RATE)
    m.set_level(9)
    m.render(int(RATE * sound.SWITCH_FADE_MS / 1000) + 2)
    assert m.playing_levels() == [9]


def test_stop_fades_out_instead_of_cutting():
    m = mixer()
    m.set_level(5)
    m.render(RATE)
    m.set_level(-1)
    fade = m.render(int(RATE * sound.SWITCH_FADE_MS / 1000) + 2).astype(np.float64)
    quarters = [np.sqrt(np.mean(q ** 2)) for q in np.array_split(fade[:-2], 4)]
    assert quarters[0] > 500, "мотор оборвался сразу"
    assert quarters == sorted(quarters, reverse=True)
    assert not m.render(2000).any()
    assert m.playing_levels() == []


def test_effects_play_once_on_top_of_the_engine():
    m = mixer()
    m.play("crash")
    out = m.render(len(sound.crash_sample(RATE)) + 500).astype(np.int32)
    assert np.abs(out[:500]).max() > 1000
    assert not out[-400:].any()
    assert m.effects == []


def test_loud_mix_is_clipped_not_wrapped():
    m = mixer(volume=1.0)
    m.set_level(11)
    for _ in range(6):
        m.play("finish")
        m.play("crash")
    out = m.render(3000).astype(np.int32)
    assert out.max() <= 32767 and out.min() >= -32767
    assert not np.any((out[1:] > 20000) & (out[:-1] < -20000)), "перелив через край int16"


def test_zero_volume_is_silence():
    m = mixer(volume=0.0)
    m.set_level(4)
    m.play("crash")
    assert not m.render(3000).any()


def test_the_engine_is_half_as_loud_as_the_effects():
    engine_peak = np.abs(sound.engine_bank(rate=RATE)[4].astype(np.int32)).max()
    m = mixer(volume=0.8)
    m.set_level(4)
    m.render(RATE)
    out = np.abs(m.render(RATE // 2).astype(np.int32)).max()
    assert abs(out - engine_peak * 0.8 * 0.5) <= engine_peak * 0.02
