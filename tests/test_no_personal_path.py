"""В файлах репозитория нет личного пути автора.

Репозиторий уходит на GitHub открытым. Однажды в нём оказался путь к папке автора на
его диске (в примерах, тестах и настройках) - а в нём имя. Скрипт публикации такой путь
заменяет и в истории, но проверка здесь ловит его раньше: на том же коммите, где он
появился, и в CI тоже.

Строка ищется в любом написании: с прямой или обратной косой чертой, с удвоенной
обратной (как в JSON и строках Python), в любом регистре. Сама она здесь не записана
целиком - иначе проверка нашла бы саму себя.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PERSONAL = re.compile(r"drive[\\/]+" + "alexey", re.IGNORECASE)


def _tracked_files():
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        # Без git (архив исходников) - все файлы, кроме служебных папок.
        skip = {".git", "build", "dist", "__pycache__", ".venv", ".ci-venv"}
        return [p for p in ROOT.rglob("*")
                if p.is_file() and not skip.intersection(p.relative_to(ROOT).parts)]
    return [ROOT / name for name in out.decode("utf-8").split("\0") if name]


def test_no_personal_path_in_files():
    found = []
    for path in _tracked_files():
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue  # двоичный файл
        text = data.decode("utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            if PERSONAL.search(line):
                found.append(f"{path.relative_to(ROOT).as_posix()}:{n}")
    assert not found, "личный путь автора в файлах: " + ", ".join(found)
