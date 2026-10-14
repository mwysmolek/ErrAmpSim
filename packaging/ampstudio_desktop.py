"""Entry point of the packaged app (PyInstaller): kayei_ampstudio.exe.

    kayei_ampstudio [project.ampstudio] [--smoke] [--new-instance]      the window
    kayei_ampstudio run design.ampstudio --json out.json                 headless (see fibamp.cli)
    kayei_ampstudio --batch-worker IN OUT                                 one batch case (internal)

A windowed exe has no console, so output goes to ampstudio.log in the data folder (the
``data`` folder next to the exe: the app is portable).
"""
from __future__ import annotations

import os
import sys


def _log_to_file() -> None:
    if sys.stdout is not None and sys.stderr is not None and not getattr(sys, "frozen", False):
        return
    from fibamp.paths import home_dir
    log = home_dir() / "ampstudio.log"
    try:
        if log.exists() and log.stat().st_size > 5_000_000:      # keep the log small
            log.replace(log.with_suffix(".old.log"))
    except OSError:
        pass
    f = open(log, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
    sys.stdout = sys.stderr = f


def main() -> int:
    import multiprocessing
    multiprocessing.freeze_support()
    argv = sys.argv[1:]
    if argv[:1] == ["--batch-worker"]:            # a batch case in its own process: files in, file out
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from fibamp.batch import _worker_main
        _worker_main(argv[1:])
        return 0
    _log_to_file()
    try:
        from fibamp.cli import COMMANDS, main as cli_main
        if argv and argv[0] in COMMANDS:
            return cli_main(argv)
        from amp_qt.app import main as app_main
        return app_main(sys.argv)
    except SystemExit as e:
        return int(e.code or 0)
    except Exception:  # noqa: BLE001 - say what went wrong and stop, never hang
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
