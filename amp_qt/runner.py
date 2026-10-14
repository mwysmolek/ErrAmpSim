"""Long computations off the UI thread, one at a time, with progress and Stop.

The engine reports progress through a ``progress(frac, message)`` callback; Stop makes that
callback raise ``Cancelled``, which unwinds the solver cleanly. A batch gets a threading.Event
instead (its cases run in worker processes that are killed).
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class Cancelled(Exception):
    pass


class _Signals(QObject):
    done = Signal(str, object)
    failed = Signal(str, str)
    progress = Signal(str, float, str)


class _Job(QRunnable):
    def __init__(self, task: str, fn: Callable, args: tuple, kwargs: dict, stop: threading.Event):
        super().__init__()
        self.task, self.fn, self.args, self.kwargs, self.stop = task, fn, args, kwargs, stop
        self.signals = _Signals()
        self.setAutoDelete(True)
        self._last = 0.0

    def _progress(self, frac, msg="") -> None:
        if self.stop.is_set():
            raise Cancelled()
        now = time.monotonic()
        if now - self._last > 0.1 or frac >= 1.0:          # at most ~10 updates a second
            self._last = now
            self.signals.progress.emit(self.task, float(frac), str(msg))

    def run(self) -> None:
        try:
            result = self.fn(*self.args, progress=self._progress, stop=self.stop, **self.kwargs)
            if self.stop.is_set() and result is None:
                raise Cancelled()
            self.signals.done.emit(self.task, result)
        except Cancelled:
            self.signals.failed.emit(self.task, "")
        except Exception as e:  # noqa: BLE001 - reported to the UI
            tb = traceback.format_exc(limit=4)
            self.signals.failed.emit(self.task, f"{type(e).__name__}: {e}\n\n{tb}")


class Runner(QObject):
    """``start(task, fn, *args)`` → ``done(task, result)`` or ``failed(task, error)``
    (error "" = stopped by the user). ``fn`` is called as ``fn(*args, progress=cb, stop=event)``."""
    busy = Signal(bool)
    done = Signal(str, object)
    failed = Signal(str, str)
    progress = Signal(str, float, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.task: Optional[str] = None
        self._stop = threading.Event()
        self._job: Optional[_Job] = None
        self.started_at = 0.0

    def is_busy(self) -> bool:
        return self.task is not None

    def start(self, task: str, fn: Callable, *args, **kwargs) -> bool:
        if self.task is not None:
            return False
        self.task = task
        self._stop = threading.Event()
        job = _Job(task, fn, args, kwargs, self._stop)
        job.signals.done.connect(self._done)
        job.signals.failed.connect(self._failed)
        job.signals.progress.connect(self.progress)
        self._job = job
        self.started_at = time.monotonic()
        self.busy.emit(True)
        self.pool.start(job)
        return True

    def stop(self) -> None:
        self._stop.set()

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at if self.task else 0.0

    @Slot(str, object)
    def _done(self, task: str, result) -> None:
        self.task, self._job = None, None
        self.busy.emit(False)
        self.done.emit(task, result)

    @Slot(str, str)
    def _failed(self, task: str, err: str) -> None:
        self.task, self._job = None, None
        self.busy.emit(False)
        self.failed.emit(task, err)

    def wait(self, ms: int = 5000) -> None:
        self._stop.set()
        self.pool.waitForDone(ms)
