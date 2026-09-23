import argparse
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

EXCLUDE = ("torch", "torchvision", "scipy", "sklearn", "matplotlib", "PIL",
           "IPython", "notebook", "pandas", "pytest")
HIDDEN = ("pkg_resources",)


def make_icon(path=ICON, size=512):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    import pygame
    from PIL import Image

    import car
    import cppn
    import config as cfg
    import render

    pygame.init()
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.rect(surface, (26, 29, 37), surface.get_rect(), border_radius=size // 6)

    genome = cppn.random_genome(cfg.CAR_CPPN_LAYERS, np.random.default_rng(11), cfg.CAR_INIT_SCALE)
    vehicle = car.generate(genome)
    vehicle.color = (108, 226, 168)
    scale = size * 0.62 / max(vehicle.length, vehicle.width)

    rotated = render.car_polygon(vehicle.stacked, np.array([size / 2, size / 2]), -np.pi / 2)
    pts = rotated.astype(np.int32).tolist()
    for (start, end), colour in zip(vehicle.slices, [render.WHEEL] * 4 + [vehicle.color, render.GLASS]):
        piece = [((x - size / 2) * scale + size / 2, (y - size / 2) * scale + size / 2)
                 for x, y in pts[start:end]]
        pygame.draw.polygon(surface, colour, piece)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    raw = pygame.image.tostring(surface, "RGBA")
    image = Image.frombytes("RGBA", (size, size), raw)
    image.save(path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return path


def run_pyinstaller(console=False):
    root = os.path.abspath(os.path.dirname(__file__))
    args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
            "--name", APP, "--icon", os.path.join(root, ICON),
            "--add-data", f"{os.path.join(root, 'models')}{os.pathsep}models",
            "--paths", root,
            "--distpath", os.path.join(root, DIST),
            "--workpath", os.path.join(root, BUILD, "work"),
            "--specpath", os.path.join(root, BUILD)]
    args += ["--console"] if console else ["--windowed"]
    for module in EXCLUDE:
        args += ["--exclude-module", module]
    for module in HIDDEN:
        args += ["--hidden-import", module]
    args.append(os.path.join(root, "main.py"))
    subprocess.run(args, check=True)


def make_portable(folder=None):
    folder = folder or os.path.join(DIST, APP)
    with open(os.path.join(folder, "portable.txt"), "w", encoding="utf-8") as f:
        f.write("Пока этот файл лежит рядом с AiCar.exe, сохранения пишутся в папку saves рядом,\n"
                "а не в профиль пользователя. Удалите его, чтобы вернуть обычное поведение.\n")
    # Интерфейс по-русски, и подробный рассказ тоже в README.ru.md; README.md - английская выжимка.
    for doc in ("README.md", "README.ru.md"):
        shutil.copy(doc, os.path.join(folder, doc))

    archive = os.path.join(DIST, f"{APP}-portable.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder):
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
    subprocess.run([nsis, NSIS_SCRIPT], check=True)
    return os.path.join(DIST, f"{APP}Setup.exe")


def size_of(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, files in os.walk(path) for f in files)


def main():
    ap = argparse.ArgumentParser(description="Сборка portable-версии и установщика")
    ap.add_argument("--console", action="store_true", help="оставить окно консоли")
    ap.add_argument("--skip-installer", action="store_true")
    args = ap.parse_args()

    print("иконка...")
    make_icon()
    print("PyInstaller...")
    run_pyinstaller(args.console)
    print("portable-архив...")
    archive = make_portable()

    folder = os.path.join(DIST, APP)
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
