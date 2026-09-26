"""pygame ушёл из игры совсем (2.0.0): ни импорта, ни зависимости, ни настроек SDL в CI.

Игры и приложения больше не делаются на pygame; окно, отрисовка, звук и цикл - на Qt.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {".git", "build", "dist", "__pycache__", ".pytest_cache", ".remember", ".ci-venv", "venv", ".venv"}
IMPORT = re.compile(r"^\s*(import\s+pygame\b|from\s+pygame\b)", re.M)


def sources():
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(folder, name)


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def test_no_source_file_imports_pygame():
    found = []
    for path in sources():
        with open(path, encoding="utf-8") as f:
            if IMPORT.search(f.read()):
                found.append(os.path.relpath(path, ROOT))
    assert found == []


def test_the_scan_really_finds_an_import():
    """Проверка проверки: регулярное выражение ловит импорт в любом виде."""
    for text in ("import pygame", "    import pygame\n", "from pygame import mixer", "import pygame.mixer"):
        assert IMPORT.search(text), text
    for text in ("# pygame был до 2.0.0", "import pygame_gui_lookalike", "x = 'import pygame'"):
        assert not IMPORT.search(text), text


def test_requirements_have_pyside6_and_no_pygame():
    for name in ("requirements.txt", "requirements-dev.txt"):
        text = read(name).lower()
        assert "pygame" not in text, name
    assert "pyside6" in read("requirements.txt").lower()


def test_ci_does_not_set_sdl_drivers_and_runs_qt_offscreen():
    for ci in ((".github", "workflows", "ci.yml"), (".gitea", "workflows", "ci.yml")):
        text = read(*ci)
        assert "SDL_" not in text and "pygame" not in text.lower(), ci
        assert "QT_QPA_PLATFORM" in text and "offscreen" in text, ci


def test_pytest_ini_has_no_pygame_leftovers():
    assert "pygame" not in read("pytest.ini")
