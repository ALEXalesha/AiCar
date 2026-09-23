"""Кадры игры для README: собираются программой, а не руками.

    python tools\\make_previews.py

Игра запускается без окна - тем же способом, что и её собственная самопроверка:
SDL с драйвером dummy рисует в обычную поверхность в памяти. Кадр берётся прямо с
неё, через pygame.image.save.

Снимок экрана тут не годится дважды. Окно может оказаться позади других, и в кадр
попадёт чужое содержимое. А главное - нужный момент руками не поймать: обучение
идёт поколениями, и кадр «половина популяции уже разбилась в одном повороте»
живёт доли секунды. Скрипт вместо этого прокручивает ровно столько кадров,
сколько нужно, с постоянным зерном - и картинка повторяется от прогона к прогону.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"
sys.path.insert(0, str(ROOT))

# До импорта pygame: иначе он уже создаст настоящее окно.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

import render  # noqa: E402
from main import DONE, Game  # noqa: E402

SEED = 7
# Докуда крутить. Первый кадр - середина обучения: нужен момент, когда часть
# популяции ещё едет, а часть уже лежит серыми - по их скоплению и видно, в каком
# повороте гибнет поколение. Второй - конец: победитель на финише и вся сводка.
MID_GENERATION = 4
LIMIT = 4000


def кадр(game, name):
    game.screen.fill(render.BG)
    game.draw_field()
    game.draw_panel()
    path = OUT / name
    pygame.image.save(game.screen, str(path))
    print(f"  {name} ({path.stat().st_size // 1024} КБ)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    game = Game(seed=SEED)
    game.ui.widgets["speed"].index = 2   # без отрисовки каждого кадра - быстрее прокрутка

    снято = False
    for n in range(LIMIT):
        game.advance()
        # Момент для первого кадра ищется по состоянию гонки, а не по номеру
        # кадра: на другой трассе то же число кадров пришлось бы на другое место.
        if (not снято and len(game.history) >= MID_GENERATION
                and 5 <= game.race.n_alive < game.race.n):
            кадр(game, "preview_game.png")
            снято = True
        if game.state == DONE:
            break
    print(f"прокручено кадров: {n + 1}, поколений {len(game.history)}, состояние {game.state}")
    if not снято:
        кадр(game, "preview_game.png")
    кадр(game, "preview_showcase.png")
    pygame.quit()
    print(f"Готово: {OUT}")


if __name__ == "__main__":
    main()
