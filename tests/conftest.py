import os
import sys

# Процессор делят обучения других проектов: numpy не берёт все ядра.
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "4")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))       # qt_app

import pytest  # noqa: E402

import offscreen  # noqa: E402

# Без экрана: Qt рисует в память, шрифты - из папки Windows. Ставится до любого импорта Qt.
offscreen.setup()


@pytest.fixture
def qapp():
    """Одно QApplication на весь прогон (без pytest-qt)."""
    return offscreen.app()


@pytest.fixture(autouse=True)
def _player_saves_stay_untouched(request, tmp_path_factory, monkeypatch):
    """Мозги и статистика каждого теста - во временной папке, а не в saves/ проекта:
    тесты доигрывают раунды до конца, а конец раунда пишет статистику.
    Отметка real_config - тест проверяет сами пути из config.py."""
    if request.node.get_closest_marker("real_config"):
        return
    import config as cfg
    folder = tmp_path_factory.mktemp("saves")
    monkeypatch.setattr(cfg, "SAVE_DIR", str(folder))
    monkeypatch.setattr(cfg, "BRAIN_FILE", str(folder / "brains.npz"))
    monkeypatch.setattr(cfg, "STATS_FILE", str(folder / "stats.json"))
