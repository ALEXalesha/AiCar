"""Qt для тестов, которым приложение нужно вне фикстуры (кэш игры, вспомогательные функции)."""
import offscreen


def qapp():
    return offscreen.app()


def ink(image, ground):
    """Маска (ширина, высота): True там, где на QImage нарисовано что-то поверх фона.

    Раскладка [x, y] - как у pygame.surfarray, чтобы срезы в тестах читались как раньше."""
    import numpy as np
    from PySide6.QtGui import QImage

    img = image.convertToFormat(QImage.Format.Format_RGB32)
    w, h = img.width(), img.height()
    raw = np.frombuffer(img.constBits(), np.uint8, count=img.bytesPerLine() * h)
    rgb = raw.reshape(h, img.bytesPerLine() // 4, 4)[:, :w, 2::-1]   # BGRA -> RGB
    return np.any(rgb != np.array(ground, dtype=np.uint8), axis=2).T
