"""Сборка (build.py, installer.nsi): версия 2.0.0, PySide6 вместо pygame, лишний Qt - вон."""
import os
import re

import pytest

import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nsi():
    with open(os.path.join(ROOT, "installer.nsi"), encoding="utf-8-sig") as f:
        return f.read()


def test_the_version_is_2_0_0():
    # окно переехало с pygame на Qt - смена движка окна, версия мажорная
    assert build.version() == "2.0.0"


def test_installer_registers_under_the_app_name_in_russian():
    text = nsi()
    assert re.search(r'!define APP "AiCar"', text)
    assert '!insertmacro MUI_LANGUAGE "Russian"' in text
    assert 'OutFile "dist\\${APP}-${VERSION}-setup.exe"' in text
    assert "RequestExecutionLevel user" in text


def test_installer_does_not_ship_the_portable_marker():
    assert 'Delete "$INSTDIR\\portable.txt"' in nsi()


def test_pygame_and_unused_qt_are_left_out():
    for module in ("pygame", "tkinter", "torch", "PIL", "pytest",
                   "PySide6.QtWebEngineCore", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtTest",
                   "PySide6.QtPdf", "PySide6.Qt3DCore", "PySide6.QtCharts"):
        assert module in build.EXCLUDE, module
    for needed in ("PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets", "PySide6.QtMultimedia"):
        assert needed not in build.EXCLUDE, needed


def test_nothing_is_hidden_imported_for_pygame_any_more():
    assert "pkg_resources" not in getattr(build, "HIDDEN", ())


def test_models_are_packed_into_the_build():
    assert "models" in open(os.path.join(ROOT, "build.py"), encoding="utf-8").read()
    assert os.path.exists(os.path.join(ROOT, "models", "track_vae.npz"))


def test_icon_is_drawn_in_every_size(tmp_path, qapp):
    Image = pytest.importorskip("PIL.Image")
    path = build.make_icon(str(tmp_path / "icon.ico"))
    with Image.open(path) as ico:
        assert {(16, 16), (32, 32), (48, 48), (256, 256)} <= set(ico.info["sizes"])


def test_trim_removes_only_what_the_game_does_not_need(tmp_path):
    qt = tmp_path / "_internal" / "PySide6"
    (qt / "translations").mkdir(parents=True)
    (qt / "plugins" / "imageformats").mkdir(parents=True)
    (qt / "plugins" / "platforms").mkdir(parents=True)
    (qt / "plugins" / "multimedia").mkdir(parents=True)
    keep = [qt / "Qt6Core.dll", qt / "Qt6Multimedia.dll", qt / "translations" / "qtbase_ru.qm",
            qt / "plugins" / "imageformats" / "qico.dll", qt / "plugins" / "platforms" / "qwindows.dll",
            qt / "plugins" / "platforms" / "qoffscreen.dll"]
    gone = [qt / "opengl32sw.dll", qt / "Qt6Pdf.dll", qt / "translations" / "qtbase_de.qm",
            qt / "plugins" / "imageformats" / "qpdf.dll"]
    # FFmpeg - видео и чужие форматы для QMediaPlayer; звук игры идёт через QAudioSink без него
    gone += [qt / f"{prefix}61.dll" for prefix in build.FFMPEG_PREFIXES]
    gone += [qt / "plugins" / "multimedia" / "ffmpegmediaplugin.dll"]
    # экранная клавиатура с Qt Quick и QML, OpenSSL рядом с python*.dll
    (qt / "plugins" / "platforminputcontexts").mkdir(parents=True)
    gone += [qt / "Qt6Quick.dll", qt / "Qt6Qml.dll", qt / "Qt6VirtualKeyboard.dll",
             qt / "plugins" / "platforminputcontexts" / "qtvirtualkeyboardplugin.dll",
             tmp_path / "_internal" / "libcrypto-3-x64.dll", tmp_path / "_internal" / "libssl-3.dll"]
    keep += [qt / "plugins" / "multimedia" / "windowsmediaplugin.dll", qt / "Qt6Network.dll",
             tmp_path / "_internal" / "python313.dll"]
    for p in keep + gone:
        p.write_bytes(b"12345")
    assert build.trim(str(tmp_path)) == 5 * len(gone)
    assert all(p.exists() for p in keep)
    assert not any(p.exists() for p in gone)


def test_portable_marker_and_docs_go_into_the_archive(tmp_path, monkeypatch):
    folder = tmp_path / "AiCar"
    folder.mkdir()
    (folder / "AiCar.exe").write_bytes(b"MZ")
    monkeypatch.setattr(build, "DIST", str(tmp_path))
    archive = build.make_portable(str(folder))
    import zipfile
    names = set(zipfile.ZipFile(archive).namelist())
    assert {"AiCar/AiCar.exe", "AiCar/portable.txt", "AiCar/README.md", "AiCar/README.ru.md",
            "AiCar/LICENSE"} <= names
    assert os.path.basename(archive) == "AiCar-2.0.0-portable.zip"
