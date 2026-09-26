"""Qt для тестов, которым приложение нужно вне фикстуры (кэш игры, вспомогательные функции)."""
import offscreen


def qapp():
    return offscreen.app()
