"""Кадры для README: собираются программой, а не руками.

    python tools\\make_previews.py

Игра запускается без окна на экране - тем же способом, что и её собственная самопроверка:
Qt с платформой offscreen рисует в картинку в памяти, шрифты берёт из папки Windows
(offscreen.py, иначе вместо букв квадратики). Кадр снимается с окна (widget.grab()) или
с той же отрисовки игры (Game.snapshot()).

Снимок экрана тут не годится дважды. Окно может оказаться позади других, и в кадр
попадёт чужое содержимое. А главное - нужный момент руками не поймать: обучение
идёт поколениями, и кадр «половина популяции уже разбилась в одном повороте»
живёт доли секунды. Скрипт вместо этого прокручивает ровно столько кадров,
сколько нужно, с постоянным зерном - и картинка повторяется от прогона к прогону.

Статистика на кадрах - из раундов, сыгранных здесь же во временной папке, а не личная.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"
sys.path.insert(0, str(ROOT))
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import offscreen  # noqa: E402

offscreen.setup(force=True)

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import car  # noqa: E402
import config as cfg  # noqa: E402
import main  # noqa: E402
import render  # noqa: E402
import stats  # noqa: E402
import track  # noqa: E402
import trackgen  # noqa: E402
import window  # noqa: E402
from rect import Rect  # noqa: E402

SEED = 7
# Докуда крутить. Первый кадр - середина обучения: нужен момент, когда часть
# популяции ещё едет, а часть уже лежит серыми - по их скоплению и видно, в каком
# повороте гибнет поколение. Второй - конец: победитель на финише и вся сводка.
MID_GENERATION = 4
LIMIT = 4000
LABEL = (150, 158, 176)


def save(image, name):
    path = OUT / name
    image.save(str(path))
    print(f"  {name} ({image.width()}x{image.height()}, {path.stat().st_size // 1024} КБ)")


def game_frames():
    game = main.Game(seed=SEED)
    game.ui.widgets["speed"].index = 2   # x20 - быстрее прокрутка
    shot = False
    for n in range(LIMIT):
        game.frame()
        # Момент для первого кадра ищется по состоянию гонки, а не по номеру
        # кадра: на другой трассе то же число кадров пришлось бы на другое место.
        if not shot and len(game.history) >= MID_GENERATION and 5 <= game.race.n_alive < game.race.n:
            save(game.snapshot(), "preview_game.png")
            save(game.snapshot().copy(game.panel_rect.qrect()), "preview_panel.png")
            save(game.snapshot(game.draw_field).copy(game.view.qrect()), "preview_track.png")
            shot = True
        if game.state == main.DONE:
            break
    print(f"прокручено кадров: {n + 1}, поколений {len(game.history)}, состояние {game.state}")
    if not shot:
        save(game.snapshot(), "preview_game.png")
    save(game.snapshot(), "preview_showcase.png")


def play_round(game, level, generator):
    """Настоящий раунд до конца: уровень и генератор - как если бы игрок их выбрал."""
    game.ui.widgets["level"].index = main.LEVEL_NAMES.index(level)
    game.apply_level()
    names = game.ui.widgets["generator"].options
    game.ui.widgets["generator"].index = names.index(generator) if generator in names else 0
    game.new_round(new_track=True, new_car=True)
    for _ in range(200):
        game.frame()
        if game.state != main.TRAINING:
            break
    last = game.history[-1]
    print(f"  раунд {game.rounds}: {level}, {generator}, поколений {len(game.history)}, "
          f"доехало {last.finished}")


def screens():
    """Меню, статистика после нескольких настоящих раундов, настройки."""
    game = main.Game(seed=21, start=False)
    win = window.MainWindow(game)
    win.resize(cfg.WINDOW_W, cfg.WINDOW_H)
    win.show()
    QApplication.processEvents()
    save(win.grab().toImage(), "preview_menu.png")

    game.ui.widgets["speed"].index = 3                 # поколение за кадр
    for level, generator in (("сложный", main.GEN_CPPN), ("обычный", main.GEN_MODEL),
                             ("лёгкий", main.GEN_MODEL), ("сложный", main.GEN_MODEL),
                             ("адский", main.GEN_CPPN), ("сложный", main.GEN_MODEL),
                             ("сложный", main.GEN_CPPN)):
        play_round(game, level, generator)
    win.show_stats(win.menu)
    QApplication.processEvents()
    save(win.grab().toImage(), "preview_stats.png")

    win.show_settings()
    QApplication.processEvents()
    save(win.grab().toImage(), "preview_settings.png")
    win.close()


def label(p, font, text, x, y):
    render.text(p, font, text, LABEL, x, y)


def draw_track_cell(p, trk, cell, font, caption):
    cam = render.Camera(cell, trk.lo, trk.hi, margin=22.0)
    p.save()
    p.setClipRect(cell.qrectf())
    render.draw_track(p, cam, trk)
    p.restore()
    label(p, font, caption, cell.left + 8, cell.top + 6)


def tracks():
    font = render.font(13)
    w, h = 330, 240
    image = render.canvas(3 * w, 2 * h)
    p = render.painter(image)
    for seed in range(6):
        trk = track.evolve_track(np.random.default_rng(seed))
        cell = Rect((seed % 3) * w, (seed // 3) * h, w, h)
        draw_track_cell(p, trk, cell, font,
                        f"seed {seed}  интерес {trk.variety:.0f}  радиус {trk.min_radius:.0f}")
    p.end()
    save(image, "preview_tracks.png")


def generators():
    if not trackgen.available():
        print("  модели трасс нет - preview_generators.png пропущен")
        return
    font = render.font(13)
    w, h = 330, 240
    image = render.canvas(4 * w, 2 * h)
    p = render.painter(image)
    model = trackgen.Generator()
    for i in range(4):
        trk = model.make_track(np.random.default_rng(100 + i))
        draw_track_cell(p, trk, Rect(i * w, 0, w, h), font, f"модель #{i + 1}  интерес {trk.variety:.0f}")
        trk = track.evolve_track(np.random.default_rng(200 + i))
        draw_track_cell(p, trk, Rect(i * w, h, w, h), font, f"CPPN #{i + 1}  интерес {trk.variety:.0f}")
    p.end()
    save(image, "preview_generators.png")


def cars():
    font = render.font(13)
    w, h = 170, 140
    image = render.canvas(5 * w, 3 * h, render.PANEL_BG)
    p = render.painter(image)
    rng = np.random.default_rng(3)
    for i in range(15):
        veh = car.random_car(rng)
        cell = Rect((i % 5) * w, (i // 5) * h, w, h)
        render.draw_car_badge(p, veh, (cell.centerx, cell.top + 58), 3.2)
        label(p, font, f"{veh.length:.0f}x{veh.width:.0f}  v{veh.max_speed:.0f}  руль {veh.max_steer:.1f}",
              cell.left + 8, cell.bottom - 30)
    p.end()
    save(image, "preview_cars.png")


def main_previews() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    offscreen.app()
    # Скрипт доигрывает раунды до конца, а конец раунда пишет статистику игрока.
    stats.use_sandbox("aicar-previews-")
    print("игра:")
    game_frames()
    print("экраны:")
    screens()
    print("трассы, генераторы, машинки:")
    tracks()
    generators()
    cars()
    print(f"Готово: {OUT}")


if __name__ == "__main__":
    main_previews()
