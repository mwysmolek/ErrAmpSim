"""Portable means: everything lives next to the exe, nothing in the registry or the user profile."""
from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

import pytest

from fibamp import paths

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def frozen_exe(tmp_path, monkeypatch):
    exe = tmp_path / "kayEi-AmpStudio" / "kayei_ampstudio.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.delenv("AMPSTUDIO_HOME", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(paths, "_HOME", None)
    yield exe
    paths._HOME = None


def test_packaged_app_keeps_data_next_to_the_exe(frozen_exe):
    assert paths.home_dir() == frozen_exe.parent / "data"
    assert (frozen_exe.parent / "data").is_dir()


def test_read_only_app_folder_falls_back_to_local_appdata(frozen_exe, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_writable", lambda p: False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    assert paths.home_dir() == tmp_path / "lad" / paths.APP_DIR_NAME


def test_env_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("AMPSTUDIO_HOME", str(tmp_path / "x"))
    assert paths.home_dir() == tmp_path / "x"


def test_batch_worker_reruns_the_exe_when_frozen(frozen_exe):
    from fibamp import batch
    assert batch.worker_command() == [str(frozen_exe), "--batch-worker"]


def test_batch_worker_uses_module_from_source():
    from fibamp import batch
    assert batch.worker_command()[1:] == ["-m", "fibamp.batch", "--worker"]


def test_library_and_batches_live_in_the_data_folder():
    from fibamp import batch, io
    home = str(paths.home_dir())
    assert io.fibre_dir().startswith(home)
    assert batch.batch_root().startswith(home)


def test_no_registry_and_no_profile_writes_in_the_code():
    """No QSettings (registry on Windows) and no hard-coded home-folder paths outside fibamp.paths."""
    offenders = []
    for f in list((ROOT / "amp_qt").rglob("*.py")) + list((ROOT / "fibamp").rglob("*.py")) + \
            list((ROOT / "packaging").glob("*.py")):
        text = f.read_text(encoding="utf-8")
        if f.name != "paths.py" and re.search(r"QSettings\(|import[^\n]*QSettings|expanduser\(|Path\.home\(\)|"
                                              r"winreg|environ[^\n]*APPDATA", text):
            offenders.append(f.name)
    assert not offenders, offenders


def test_chart_views_are_off_the_record():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    charts = importlib.import_module("amp_qt.charts")
    assert charts.profile().isOffTheRecord()
