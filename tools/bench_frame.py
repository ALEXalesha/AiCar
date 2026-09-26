"""Время кадра: трасса, 50 машинок (10 разбиты), слежение за лидером с лучами и телеметрией,
панель со статистикой и графиком. Тот же сценарий, что мерил pygame-версию 1.1.0.

    python tools\\bench_frame.py            кадр в картинку в памяти (Qt offscreen)
    python tools\\bench_frame.py --window   в настоящем окне: paintEvent и вывод на экран

Числа 2.0.0 и 1.1.0 - в README (раздел «Почему всё считается быстро»).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(var, "4")


def scenario(game):
    import numpy as np
    import main
    n = game.race.n
    step = len(game.track.center) // n
    game.race.pos[:] = game.track.center[:n * step:step]
    game.race.angle[:] = np.linspace(0, 6.28, n)
    game.race.alive[:] = True
    game.race.alive[::5] = False          # десять разбитых серыми
    game.watched = main.WATCH_LEADER


def stats_line(name, times_ms):
    import numpy as np
    t = np.array(times_ms)
    return (f"{name}: среднее {t.mean():.2f} мс, медиана {np.median(t):.2f}, "
            f"95% {np.percentile(t, 95):.2f}, кадров {len(t)}")


def main_bench(argv=None):
    ap = argparse.ArgumentParser(description="Замер времени кадра")
    ap.add_argument("--window", action="store_true", help="мерить в настоящем окне")
    ap.add_argument("--frames", type=int, default=400)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    import offscreen
    if not args.window:
        offscreen.setup(force=True)
    app = offscreen.app()
    import main
    import stats
    stats.use_sandbox("aicar-bench-")

    game = main.Game(seed=0)
    scenario(game)
    lines = []
    if not args.window:
        import render
        image = render.canvas(game.width, game.height)
        times = []
        for i in range(args.frames + 30):
            started = time.perf_counter()
            p = render.painter(image)
            game.paint(p)
            p.end()
            if i >= 30:
                times.append((time.perf_counter() - started) * 1000)
        lines.append(stats_line("Qt offscreen, кадр в картинку", times))
    else:
        import window
        from PySide6.QtWidgets import QApplication
        win = window.MainWindow(game)
        win.resize(game.width, game.height)
        win.show()
        win.stack.setCurrentWidget(win.view)
        for _ in range(30):
            QApplication.processEvents()
        win.view.paint_ms.clear()
        win.view.paint_ms = __import__("collections").deque(maxlen=args.frames)
        total = []
        for _ in range(args.frames):
            started = time.perf_counter()
            win.view.repaint()                      # paintEvent и вывод на экран, сразу
            total.append((time.perf_counter() - started) * 1000)
            QApplication.processEvents()
        lines.append(stats_line("Qt окно, paintEvent", list(win.view.paint_ms)))
        lines.append(stats_line("Qt окно, repaint целиком (с выводом на экран)", total))
        win.close()

    # Полный кадр игры на x1: шаг физики и отрисовка.
    import render
    game.ui.widgets["speed"].index = 0
    image = render.canvas(game.width, game.height)
    times = []
    for _ in range(300):
        started = time.perf_counter()
        game.advance()
        p = render.painter(image)
        game.paint(p)
        p.end()
        times.append((time.perf_counter() - started) * 1000)
    lines.append(stats_line("Qt, advance x1 + кадр", times))
    for line in lines:
        print(line)
    app.quit()
    return lines


if __name__ == "__main__":
    main_bench()
