"""kayEi AmpStudio desktop application (PySide6).

    python -m amp_qt.app [project.ampstudio] [--smoke] [--new-instance]

One running copy per data folder: starting it again (or dropping a project on the exe) hands
the file to the running window. ``--smoke`` is the CI self-test: open the window, simulate a
quick design, print SMOKE-QT OK and exit.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from fibamp.paths import home_dir

from . import theme
from .charts import clean_cache
from .document import Document
from .mainwindow import MainWindow

SMOKE_PRESET = "C-band pre-amplifier (Er80-4/125 class)"


def server_name() -> str:
    """Per user and per data folder: two portable copies in different folders run side by side."""
    user = os.environ.get("USERNAME") or os.environ.get("USER") or "user"
    tag = hashlib.sha1(str(home_dir()).encode("utf-8")).hexdigest()[:10]
    return f"kayei-ampstudio-{user}-{tag}"


def _hand_over(path: str) -> bool:
    sock = QLocalSocket()
    sock.connectToServer(server_name())
    if not sock.waitForConnected(300):
        return False
    sock.write((path or "").encode("utf-8"))
    sock.flush()
    sock.waitForBytesWritten(500)
    sock.disconnectFromServer()
    return True


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    flags = {a for a in argv[1:] if a.startswith("--")}
    files = [a for a in argv[1:] if not a.startswith("--")]
    path = str(Path(files[0]).resolve()) if files else None
    smoke = "--smoke" in flags
    if not smoke and "--new-instance" not in flags and _hand_over(path or ""):
        return 0

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    app.setApplicationName("kayEi AmpStudio")
    app.setOrganizationName("kayEi")
    theme.apply_theme(app)
    clean_cache()
    doc = Document()
    win = MainWindow(doc, smoke=smoke)

    server = QLocalServer()
    QLocalServer.removeServer(server_name())
    if not smoke and server.listen(server_name()):
        def on_conn():
            sock = server.nextPendingConnection()
            if sock is None:
                return
            sock.waitForReadyRead(500)
            data = bytes(sock.readAll()).decode("utf-8", "replace").strip()
            win.showNormal()
            win.raise_()
            win.activateWindow()
            if data:
                win.open_path(data)
        server.newConnection.connect(on_conn)

    win.show()
    if smoke:
        from fibamp.presets import numerics
        from fibamp.project import Project
        p = Project.from_preset(SMOKE_PRESET)
        p.sim = numerics("draft").to_dict()
        doc.replace(p, None)

        def check():
            res = win.pages["results"].res
            ok = res is not None and res.summary.get("output pulse energy (uJ)", 0) > 0
            print("SMOKE-QT OK" if ok else f"SMOKE-QT FAIL: no results {win.last_error}", flush=True)
            try:
                sys.stdout.flush()
            except (OSError, AttributeError):
                pass
            hard_exit(0 if ok else 3)

        win.runner.done.connect(lambda *_: QTimer.singleShot(2500, check))
        win.runner.failed.connect(lambda *_: QTimer.singleShot(500, check))
        QTimer.singleShot(200000, check)
        QTimer.singleShot(300, win.simulate)
    else:
        win.start(path)
    return app.exec()


def hard_exit(code: int) -> None:
    """End the process now with an exact exit code. On Windows os._exit goes through ExitProcess, whose
    DLL-detach round trips the Qt WebEngine / Chromium libraries while the app is live and can die with an
    access violation; TerminateProcess skips that and keeps the code the CI step reads."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.TerminateProcess(ctypes.c_void_p(-1), ctypes.c_uint(code))
        except Exception:  # noqa: BLE001
            pass
    os._exit(code)


if __name__ == "__main__":
    sys.exit(main())
