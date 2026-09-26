"""Сборка для Windows: папка dist/AiCar, portable-архив и установщик NSIS.

    pip install pyinstaller pillow
    python build.py

Номер версии - один, в installer.nsi; по нему названы и установщик, и архив.
С 2.0.0 внутри PySide6 (Qt), а не pygame. Qt большой, и PyInstaller кладёт в сборку
кое-что на всякий случай - лишнее убирается (EXCLUDE и trim), как в крестиках-ноликах.
"""
import argparse
import io
import os
import shutil
import subprocess
import sys
import zipfile

APP = "AiCar"
BUILD = "build"
DIST = "dist"
ICON = os.path.join(BUILD, f"{APP}.ico")
NSIS_SCRIPT = "installer.nsi"
NSIS_PATHS = (r"C:\Program Files (x86)\NSIS\makensis.exe", r"C:\Program Files\NSIS\makensis.exe")
ROOT = os.path.abspath(os.path.dirname(__file__))

# Игре не нужны: без них сборка меньше. PIL нужен только build.py - для иконки; torch - только
# обучению генератора трасс. Модули Qt, которых игра не импортирует, - на всякий случай тоже:
# чтобы случайный импорт в зависимости не притащил мегабайты.
EXCLUDE = ("torch", "torchvision", "scipy", "sklearn", "matplotlib", "PIL", "IPython", "notebook",
           "pandas", "pytest", "tkinter", "pygame",
           "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebChannel",
           "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.QtQuick3D",
           "PySide6.QtTest", "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtCharts",
           "PySide6.QtDataVisualization", "PySide6.QtGraphs", "PySide6.Qt3DCore", "PySide6.Qt3DRender",
           "PySide6.QtMultimediaWidgets", "PySide6.QtSql", "PySide6.QtBluetooth", "PySide6.QtPositioning",
           "PySide6.QtLocation", "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtDesigner",
           "PySide6.QtHelp", "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtSvg",
           "PySide6.QtXml", "PySide6.QtNetworkAuth", "PySide6.QtRemoteObjects", "PySide6.QtScxml",
           "PySide6.QtSpatialAudio", "PySide6.QtTextToSpeech", "PySide6.QtWebSockets", "PySide6.QtHttpServer",
           # ssl тянут urllib и http.client из numpy.lib._datasource; оба живут и без него
           # (import ssl у них в try), а это 12 МБ OpenSSL.
           "ssl", "_ssl", "charset_normalizer")

# Библиотеки и плагины Qt, которые PyInstaller кладёт за компанию, а окну игры они не нужны:
# экранная клавиатура тянет за собой Qt Quick и QML (15 МБ), SVG, TLS и сеть - для загрузок,
# direct2d и minimal - другие способы выводить окно (нужны windows и offscreen).
QT_UNUSED = ("Qt6VirtualKeyboard.dll", "Qt6Quick.dll", "Qt6Qml.dll", "Qt6QmlModels.dll", "Qt6QmlMeta.dll",
             "Qt6QmlWorkerScript.dll", "Qt6OpenGL.dll", "Qt6Svg.dll", "Qt6Pdf.dll", "opengl32sw.dll",
             os.path.join("plugins", "platforminputcontexts", "qtvirtualkeyboardplugin.dll"),
             os.path.join("plugins", "iconengines", "qsvgicon.dll"),
             os.path.join("plugins", "imageformats", "qsvg.dll"),
             os.path.join("plugins", "imageformats", "qpdf.dll"),
             os.path.join("plugins", "tls", "qcertonlybackend.dll"),
             os.path.join("plugins", "tls", "qopensslbackend.dll"),
             os.path.join("plugins", "tls", "qschannelbackend.dll"),
             os.path.join("plugins", "networkinformation", "qnetworklistmanager.dll"),
             os.path.join("plugins", "generic", "qtuiotouchplugin.dll"),
             os.path.join("plugins", "platforms", "qdirect2d.dll"),
             os.path.join("plugins", "platforms", "qminimal.dll"),
             os.path.join("plugins", "multimedia", "ffmpegmediaplugin.dll"))
# Файлы игрока, которые portable-версия пишет рядом с exe: после пробного запуска из dist
# они там есть, но в архив попадать не должны.
PLAYER_FILES = ("settings.json", "window.json", "window.json.tmp", "settings.json.tmp")

# OpenSSL рядом с python*.dll: после исключения ssl и плагинов TLS его никто не грузит.
OPENSSL_PREFIXES = ("libcrypto-", "libssl-")

# FFmpeg в сборке PySide6 - для QMediaPlayer (видео, mp3). Звук игры - свой синтез в
# QAudioSink, ему FFmpeg не нужен (проверено самопроверкой собранной игры: «звук: есть»).
FFMPEG_PREFIXES = ("avcodec-", "avformat-", "avutil-", "swresample-", "swscale-")


def version():
    with open(os.path.join(ROOT, NSIS_SCRIPT), encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("!define VERSION"):
                return line.split('"')[1]
    raise SystemExit("в installer.nsi нет !define VERSION")


def make_icon(path=ICON):
    """Иконка рисуется тем же кодом, что значок окна (render.icon_image), и собирается в .ico."""
    sys.path.insert(0, ROOT)
    import offscreen
    offscreen.setup()
    from PIL import Image
    from PySide6.QtCore import QBuffer, QIODevice

    import render

    offscreen.app()
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    render.icon_image(render.icon_car(), 256).save(buf, "PNG")
    image = Image.open(io.BytesIO(bytes(buf.data())))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    image.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return path


def run_pyinstaller(console=False):
    args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
            "--name", APP, "--icon", os.path.join(ROOT, ICON),
            "--add-data", f"{os.path.join(ROOT, 'models')}{os.pathsep}models",
            "--paths", ROOT,
            "--distpath", os.path.join(ROOT, DIST),
            "--workpath", os.path.join(ROOT, BUILD, "work"),
            "--specpath", os.path.join(ROOT, BUILD)]
    args += ["--console"] if console else ["--windowed"]
    for module in EXCLUDE:
        args += ["--exclude-module", module]
    args.append(os.path.join(ROOT, "main.py"))
    subprocess.run(args, check=True, cwd=ROOT)


def trim(folder):
    """Убрать из сборки то, что PyInstaller кладёт на всякий случай, а игре не нужно:
    программный OpenGL (окно рисуется без OpenGL), Qt Quick с экранной клавиатурой, SVG,
    TLS, переводы Qt кроме русского, чтение PDF, FFmpeg (видео для QMediaPlayer), OpenSSL.
    Проверка, что сборка после этого жива, - AiCar.exe --selftest (окно, шрифт, звук).
    Возвращает, сколько байт освободилось."""
    internal = os.path.join(folder, "_internal")
    qt = os.path.join(internal, "PySide6")
    doomed = [os.path.join(qt, name) for name in QT_UNUSED]
    if os.path.isdir(qt):
        doomed += [os.path.join(qt, n) for n in os.listdir(qt) if n.startswith(FFMPEG_PREFIXES)]
    if os.path.isdir(internal):
        doomed += [os.path.join(internal, n) for n in os.listdir(internal) if n.startswith(OPENSSL_PREFIXES)]
    translations = os.path.join(qt, "translations")
    if os.path.isdir(translations):
        doomed += [os.path.join(translations, n) for n in os.listdir(translations) if not n.endswith("_ru.qm")]
    freed = 0
    for path in doomed:
        if os.path.isfile(path):
            freed += os.path.getsize(path)
            os.remove(path)
    return freed


def make_portable(folder=None):
    folder = folder or os.path.join(ROOT, DIST, APP)
    with open(os.path.join(folder, "portable.txt"), "w", encoding="utf-8") as f:
        f.write("Пока этот файл лежит рядом с AiCar.exe, сохранения, статистика, настройки и место\n"
                "окна пишутся рядом, а не в профиль пользователя. Удалите его, чтобы вернуть\n"
                "обычное поведение.\n")
    # Интерфейс по-русски, и подробный рассказ тоже в README.ru.md; README.md - английская выжимка.
    for doc in ("README.md", "README.ru.md", "LICENSE"):
        shutil.copy(os.path.join(ROOT, doc), os.path.join(folder, doc))

    archive = os.path.join(DIST if os.path.isabs(DIST) else os.path.join(ROOT, DIST),
                           f"{APP}-{version()}-portable.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder):
            # Следы пробного запуска из dist (portable пишет рядом с exe) - не в архив.
            dirs[:] = [d for d in dirs if not (root == folder and d == "saves")]
            files = [n for n in files if not (root == folder and n in PLAYER_FILES)]
            for name in files:
                full = os.path.join(root, name)
                zf.write(full, os.path.join(APP, os.path.relpath(full, folder)))
    return archive


def find_nsis():
    found = shutil.which("makensis")
    if found:
        return found
    return next((p for p in NSIS_PATHS if os.path.exists(p)), None)


def make_installer():
    nsis = find_nsis()
    if nsis is None:
        print("NSIS не найден, установщик пропущен")
        return None
    subprocess.run([nsis, NSIS_SCRIPT], check=True, cwd=ROOT)
    return os.path.join(ROOT, DIST, f"{APP}-{version()}-setup.exe")


def size_of(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, files in os.walk(path) for f in files)


def main():
    ap = argparse.ArgumentParser(description="Сборка portable-версии и установщика")
    ap.add_argument("--console", action="store_true", help="оставить окно консоли")
    ap.add_argument("--skip-installer", action="store_true")
    args = ap.parse_args()

    print("иконка...")
    make_icon(os.path.join(ROOT, ICON))
    print("PyInstaller...")
    run_pyinstaller(args.console)
    freed = trim(os.path.join(ROOT, DIST, APP))
    print(f"убрано лишнее: {freed / 1e6:.0f} МБ")
    print("portable-архив...")
    archive = make_portable()

    folder = os.path.join(ROOT, DIST, APP)
    print()
    print(f"папка      {folder}  ({size_of(folder) / 1e6:.0f} МБ)")
    print(f"portable   {archive}  ({size_of(archive) / 1e6:.0f} МБ)")

    if not args.skip_installer:
        print("установщик...")
        setup = make_installer()
        if setup and os.path.exists(setup):
            print(f"установщик {setup}  ({size_of(setup) / 1e6:.0f} МБ)")


if __name__ == "__main__":
    main()
