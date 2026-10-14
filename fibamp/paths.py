"""Where fibamp keeps its files (fibre library, batches, stages, projects, settings).

Portable by design: the packaged app keeps everything in a ``data`` folder next to the exe, so
the whole app lives in one folder that can be copied, zipped or run from a USB stick, and
deleting the folder removes every trace of it. Order of precedence:

1. ``$AMPSTUDIO_HOME`` (tests, CI, or a deliberate override);
2. packaged app: ``<folder of the exe>\\data`` when that folder is writable, otherwise
   ``%LOCALAPPDATA%\\kayEi AmpStudio`` (e.g. unpacked under Program Files);
3. running from source: ``~/fibamp`` (where the original browser app kept its files).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "kayEi AmpStudio"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


_HOME: Path | None = None


def home_dir() -> Path:
    """The data folder (created on first use)."""
    global _HOME
    env = os.environ.get("AMPSTUDIO_HOME")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    if _HOME is not None:
        return _HOME
    if is_frozen():
        p = Path(sys.executable).resolve().parent / "data"
        if not _writable(p):
            base = os.environ.get("LOCALAPPDATA") or str(Path.home())
            p = Path(base) / APP_DIR_NAME
    else:
        p = Path.home() / "fibamp"
    p.mkdir(parents=True, exist_ok=True)
    _HOME = p
    return p


def sub(name: str) -> Path:
    p = home_dir() / name
    p.mkdir(parents=True, exist_ok=True)
    return p
