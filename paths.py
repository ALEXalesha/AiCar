import os
import sys

APP_NAME = "AiCar"


def frozen():
    return getattr(sys, "frozen", False)


def resource_dir():
    if frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def data_dir():
    if not frozen():
        return os.path.dirname(os.path.abspath(__file__))
    beside_exe = os.path.dirname(sys.executable)
    if os.path.exists(os.path.join(beside_exe, "portable.txt")):
        return beside_exe
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, APP_NAME)


def resource(*parts):
    return os.path.join(resource_dir(), *parts)


def user_file(*parts):
    path = os.path.join(data_dir(), *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
