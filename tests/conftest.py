import os
import sys

# Процессор делят обучения других проектов: numpy не берёт все ядра.
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "4")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import offscreen  # noqa: E402

# Без экрана: Qt рисует в память, шрифты - из папки Windows. Ставится до любого импорта Qt.
offscreen.setup()


@pytest.fixture
def qapp():
    """Одно QApplication на весь прогон (без pytest-qt)."""
    return offscreen.app()
