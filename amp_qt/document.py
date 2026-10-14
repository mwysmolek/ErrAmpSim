"""The open project, its file, autosave and app settings (all inside the portable data folder).

Settings live in ``settings.json`` next to the projects, never in the registry (no QSettings).
"""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from typing import Any, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from fibamp.paths import home_dir, sub
from fibamp.project import EXT, Project, atomic_write, slug


def settings_path() -> str:
    return str(home_dir() / "settings.json")


def load_settings() -> dict:
    try:
        with open(settings_path(), encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(d: dict) -> None:
    try:
        atomic_write(settings_path(), json.dumps(d, indent=1))
    except OSError:
        pass


def projects_dir() -> str:
    return str(sub("projects"))


def autosave_path() -> str:
    return str(home_dir() / f"autosave{EXT}")


class Document(QObject):
    """``changed(part)`` after any edit: part is fiber, pump, seed, sim, qswitch, batch, meta,
    xs or all (a new / opened project)."""
    changed = Signal(str)
    fileChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = Project.from_preset()
        self.path: Optional[str] = None
        self.dirty = False
        self.autosaved_at: Optional[datetime] = None
        self.settings = load_settings()
        self._auto = QTimer(self)
        self._auto.setSingleShot(True)
        self._auto.setInterval(1500)
        self._auto.timeout.connect(self.autosave)

    # ------------------------------------------------------------------ edits
    def update(self, part: str, value: Any) -> None:
        if getattr(self.project, part) == value:
            return
        setattr(self.project, part, copy.deepcopy(value))
        self._touch(part)

    def _touch(self, part: str) -> None:
        self.dirty = True
        self._auto.start()
        self.changed.emit(part)
        self.fileChanged.emit()

    def touch(self, part: str) -> None:
        self._touch(part)

    def replace(self, project: Project, path: Optional[str] = None, dirty: bool = False) -> None:
        self.project = project
        self.path = path
        self.dirty = dirty
        project.register_xs()
        self.changed.emit("all")
        self.fileChanged.emit()
        if path:
            self.add_recent(path)

    # ------------------------------------------------------------------- file
    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path or os.path.join(projects_dir(), slug(self.project.name) + EXT)
        self.project.save(path)
        self.path = path
        self.dirty = False
        self.add_recent(path)
        self.fileChanged.emit()
        return path

    def open(self, path: str) -> None:
        self.replace(Project.load(path), path)

    def autosave(self) -> None:
        try:
            d = json.loads(self.project.to_json())
            d["_path"] = self.path or ""
            d["_dirty"] = self.dirty
            atomic_write(autosave_path(), json.dumps(d, indent=1))
            self.autosaved_at = datetime.now()
        except (OSError, TypeError, ValueError):
            pass
        self.fileChanged.emit()

    def restore_autosave(self) -> bool:
        try:
            with open(autosave_path(), encoding="utf-8") as fh:
                d = json.load(fh)
            path, dirty = d.pop("_path", "") or None, bool(d.pop("_dirty", False))
            self.replace(Project.from_json(json.dumps(d)), path if path and os.path.exists(path) else None, dirty)
            return True
        except (OSError, ValueError, TypeError, KeyError):
            return False

    # --------------------------------------------------------------- settings
    def pref(self, key: str, default=None):
        return self.settings.get(key, default)

    def set_pref(self, key: str, value) -> None:
        self.settings[key] = value
        save_settings(self.settings)

    def add_recent(self, path: str) -> None:
        rec = [p for p in self.settings.get("recent", []) if p != path and os.path.exists(p)]
        self.set_pref("recent", [path] + rec[:7])

    def title(self) -> str:
        name = os.path.basename(self.path) if self.path else "not saved yet"
        return f"{self.project.name} — {name}{' •' if self.dirty else ''}"
