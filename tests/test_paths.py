import os
import sys

import config as cfg
import paths


def test_not_frozen_when_running_from_source():
    assert not paths.frozen()


def test_source_run_keeps_everything_in_the_project():
    project = os.path.dirname(os.path.abspath(paths.__file__))
    assert paths.data_dir() == project
    assert paths.resource_dir() == project


def test_resource_joins_onto_the_resource_folder():
    assert paths.resource("models", "x.npz") == os.path.join(paths.resource_dir(), "models", "x.npz")


def test_user_file_creates_the_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "data_dir", lambda: str(tmp_path / "деep"))
    path = paths.user_file("saves", "brains.npz")
    assert os.path.isdir(os.path.dirname(path))


def test_frozen_without_marker_uses_the_user_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "AiCar.exe"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData"))
    assert paths.data_dir() == os.path.join(str(tmp_path / "AppData"), paths.APP_NAME)


def test_frozen_with_marker_writes_next_to_the_exe(tmp_path, monkeypatch):
    (tmp_path / "portable.txt").write_text("portable", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "AiCar.exe"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData"))
    assert paths.data_dir() == str(tmp_path)


def test_frozen_falls_back_to_home_without_localappdata(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "AiCar.exe"))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert paths.data_dir().startswith(os.path.expanduser("~"))


def test_frozen_resources_come_from_the_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    assert paths.resource_dir() == str(tmp_path / "bundle")


def test_config_paths_are_absolute():
    for path in (cfg.SAVE_DIR, cfg.BRAIN_FILE, cfg.STATS_FILE, cfg.DATASET_FILE, cfg.MODEL_FILE):
        assert os.path.isabs(path)


def test_saves_are_separate_from_bundled_resources():
    assert cfg.MODEL_FILE.startswith(paths.resource_dir())
    assert cfg.BRAIN_FILE.startswith(paths.data_dir())
