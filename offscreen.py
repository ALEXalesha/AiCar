"""Qt без экрана - для тестов, stress.py, самопроверки, кадров README и иконки сборки.

У платформы offscreen своя база шрифтов, и пустая: вместо букв рисуются квадратики.
Ей показывается папка шрифтов Windows - тогда текст на кадрах тот же, что в окне
(Consolas, как в окне игры). Звать до создания QApplication.
"""
import os


def setup(force=False):
    if force:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    if os.path.isdir(fonts):
        os.environ.setdefault("QT_QPA_FONTDIR", fonts)


def app():
    """Одно QApplication на процесс: второе Qt создать не даст. Ссылка держится здесь,
    иначе сборщик мусора убьёт приложение раньше окон."""
    from PySide6.QtWidgets import QApplication
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


_APP = []
