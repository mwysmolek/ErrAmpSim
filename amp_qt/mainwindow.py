"""The application window: project panel, pages, menus, status bar and the run loop."""
from __future__ import annotations

import os
import sys
import traceback

from PySide6.QtCore import QByteArray, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QPixmap
from PySide6.QtWidgets import (QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QTabWidget, QVBoxLayout,
                               QWidget)

from fibamp.config import SimConfig
from fibamp.paths import home_dir
from fibamp.presets import PRESETS, numerics
from fibamp.project import EXT, Project

from . import __version__, tasks
from . import theme as T
from .document import Document, projects_dir
from .pages.batch_page import BatchPage
from .pages.fibre_page import FibrePage
from .pages.launch_page import LaunchPage
from .pages.method_page import MethodPage
from .pages.numerics_page import NumericsPage, level_of
from .pages.qswitch_page import QSwitchPage
from .pages.results_page import ResultsPage
from .runner import Runner

APP = "kayEi AmpStudio"
PAGES = [("fibre", "🧵 Fibre"), ("launch", "⚡ Pump && seed"), ("numerics", "⚙ Numerics"), ("results", "📈 Results"),
         ("batch", "🧮 Batch"), ("qswitch", "🔦 Q-switch"), ("method", "📘 Method")]
TASK_LABEL = {"simulate": "Simulating the amplifier", "qswitch": "Designing the Q-switched pulse",
              "batch": "Running the batch"}


class ProjectPanel(QWidget):
    """Left dock: brand, project, notes, run controls."""

    def __init__(self, win: "MainWindow"):
        super().__init__(win)
        self.win, self.doc = win, win.doc
        self._loading = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)
        brand_row = QHBoxLayout()
        logo = QLabel()
        pm = QPixmap(str(T.ASSETS / "logo.png"))
        if not pm.isNull():
            logo.setPixmap(pm.scaled(44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            brand_row.addWidget(logo)
        brand_row.addWidget(QLabel(f'<span style="color:{T.ACCENT}; letter-spacing:2px; font-weight:700">kayEi</span>'
                                   '<br><span style="font-size:20px; font-weight:800">AmpStudio</span>'), 1)
        lay.addLayout(brand_row)
        sub = QLabel("Pulsed Er / Er:Yb fibre amplifiers, from datasheet to pulse")
        sub.setProperty("muted", True)
        sub.setWordWrap(True)
        lay.addWidget(sub)
        lay.addSpacing(6)
        lay.addWidget(QLabel("<b>📁 Project</b>"))
        self.name = QLineEdit()
        self.name.setToolTip("The project's name (file name when saved, report titles)")
        lay.addWidget(self.name)
        self.saved = QLabel()
        self.saved.setProperty("muted", True)
        self.saved.setWordWrap(True)
        lay.addWidget(self.saved)
        row = QHBoxLayout()
        b_save = QPushButton("Save")
        b_save.setToolTip("Save the project (Ctrl+S)")
        b_open = QPushButton("Open…")
        b_open.setToolTip("Open a project (Ctrl+O)")
        row.addWidget(b_save)
        row.addWidget(b_open)
        lay.addLayout(row)
        nrow = QHBoxLayout()
        self.preset = QComboBox()
        for k in PRESETS:
            self.preset.addItem(k, k)
        self.preset.setToolTip("Start a new project from a complete, working design")
        b_new = QPushButton("New")
        b_new.setToolTip("New project from the selected starting point")
        nrow.addWidget(self.preset, 1)
        nrow.addWidget(b_new)
        lay.addLayout(nrow)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("What is this design for? (saved with the project)")
        self.notes.setFixedHeight(80)
        lay.addWidget(self.notes)
        lay.addSpacing(8)
        lay.addWidget(QLabel("<b>▶ Run</b>"))
        arow = QHBoxLayout()
        arow.addWidget(QLabel("Accuracy"))
        self.acc = QComboBox()
        for k in ("draft", "standard", "fine"):
            self.acc.addItem(k.capitalize(), k)
        self.acc.addItem("Custom", "custom")
        self.acc.setToolTip("Grid and convergence settings (details on the Numerics page)")
        arow.addWidget(self.acc, 1)
        lay.addLayout(arow)
        rrow = QHBoxLayout()
        self.b_run = QPushButton("▶ Simulate")
        self.b_run.setProperty("primary", True)
        self.b_run.setToolTip("Run the amplifier (F5)")
        self.b_stop = QPushButton("■ Stop")
        self.b_stop.setProperty("danger", True)
        self.b_stop.setToolTip("Stop the running computation (Esc)")
        self.b_stop.setEnabled(False)
        rrow.addWidget(self.b_run, 2)
        rrow.addWidget(self.b_stop, 1)
        lay.addLayout(rrow)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.hide()
        lay.addWidget(self.bar)
        self.msg = QLabel()
        self.msg.setWordWrap(True)
        self.msg.setProperty("muted", True)
        lay.addWidget(self.msg)
        lay.addStretch(1)
        foot = QLabel(f'kayEi AmpStudio {__version__} · portable<br><a href="data" style="color:{T.MUTED}">'
                      f'data folder</a>')
        foot.setProperty("muted", True)
        foot.setWordWrap(True)
        foot.linkActivated.connect(lambda _h: win.open_data_folder())
        lay.addWidget(foot)

        b_save.clicked.connect(win.save)
        b_open.clicked.connect(win.open_dialog)
        b_new.clicked.connect(lambda: win.new_project(self.preset.currentData()))
        self.name.editingFinished.connect(self._renamed)
        self.notes.textChanged.connect(self._noted)
        self.acc.currentIndexChanged.connect(self._accuracy)
        self.b_run.clicked.connect(win.simulate)
        self.b_stop.clicked.connect(win.runner.stop)
        self.doc.changed.connect(self.reload)
        self.doc.fileChanged.connect(self.reload_file)
        self.reload("all")

    def reload(self, part: str = "all") -> None:
        self._loading = True
        try:
            p = self.doc.project
            if self.name.text() != p.name:
                self.name.setText(p.name)
            if self.notes.toPlainText() != (p.notes or ""):
                self.notes.setPlainText(p.notes or "")
            lvl = level_of(p.sim)
            self.acc.setCurrentIndex(max(0, self.acc.findData(lvl)))
        finally:
            self._loading = False
        self.reload_file()

    def reload_file(self) -> None:
        d = self.doc
        where = d.path or "not saved to a file yet"
        state = "● unsaved changes" if d.dirty else "✓ saved"
        auto = f" · autosaved {d.autosaved_at:%H:%M:%S}" if d.autosaved_at else ""
        self.saved.setText(f"{state}{auto}\n{where}")
        self.win.setWindowTitle(f"{d.title()} — {APP}")

    def _renamed(self) -> None:
        if not self._loading and self.name.text().strip():
            self.doc.update("name", self.name.text().strip())

    def _noted(self) -> None:
        if not self._loading:
            self.doc.update("notes", self.notes.toPlainText())

    def _accuracy(self) -> None:
        if self._loading:
            return
        lvl = self.acc.currentData()
        if lvl == "custom":
            self.win.go("numerics")
            return
        self.doc.update("sim", numerics(lvl, SimConfig.from_dict(self.doc.project.sim)).to_dict())

    def busy(self, on: bool) -> None:
        self.b_run.setEnabled(not on)
        self.b_stop.setEnabled(on)
        self.bar.setVisible(on)
        if on:
            self.bar.setValue(0)


class MainWindow(QMainWindow):
    def __init__(self, doc: Document, smoke: bool = False):
        super().__init__()
        self.doc = doc
        self.smoke = smoke
        self.runner = Runner(self)
        self.last_error = ""
        self.setWindowTitle(APP)
        self.resize(1480, 920)
        self.panel = ProjectPanel(self)
        dock = QDockWidget("PROJECT", self)
        dock.setObjectName("project")
        dock.setWidget(self.panel)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock.setMinimumWidth(270)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.pages = {}
        for key, label in PAGES:
            page = {"fibre": FibrePage, "launch": LaunchPage, "numerics": NumericsPage, "results": ResultsPage,
                    "batch": BatchPage, "qswitch": QSwitchPage, "method": MethodPage}[key](self)
            self.pages[key] = page
            self.tabs.addTab(page, label)
        self.setCentralWidget(self.tabs)
        self.status_lab = QLabel()
        self.statusBar().addWidget(self.status_lab, 1)
        self.elapsed = QLabel()
        self.statusBar().addPermanentWidget(self.elapsed)
        self._tick = QTimer(self)
        self._tick.setInterval(500)
        self._tick.timeout.connect(lambda: self.elapsed.setText(f"{self.runner.elapsed():.0f} s"))
        self._menus()

        self.runner.busy.connect(self._busy)
        self.runner.progress.connect(self._progress)
        self.runner.done.connect(self._done)
        self.runner.failed.connect(self._failed)
        geo = self.doc.pref("geometry")
        if geo and not smoke:
            self.restoreGeometry(QByteArray.fromBase64(geo.encode()))

    # -------------------------------------------------------------- menus
    def _menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        new = m.addMenu("New from")
        for k in PRESETS:
            new.addAction(k, lambda k=k: self.new_project(k))
        self._act(m, "&Open…", self.open_dialog, QKeySequence.Open)
        self.recent_menu = m.addMenu("Open recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        self._act(m, "&Save", self.save, QKeySequence.Save)
        self._act(m, "Save &as…", self.save_as, QKeySequence("Ctrl+Shift+S"))
        m.addSeparator()
        self._act(m, "Import run setup (fibamp JSON)…", self.import_setup)
        self._act(m, "Export run setup (fibamp JSON)…", self.export_setup)
        m.addSeparator()
        self._act(m, "Open the data folder", self.open_data_folder)
        m.addSeparator()
        self._act(m, "E&xit", self.close, QKeySequence.Quit)
        r = mb.addMenu("&Run")
        self._act(r, "▶ Simulate the amplifier", self.simulate, QKeySequence("F5"))
        self._act(r, "Design the Q-switched pulse", self.run_qswitch, QKeySequence("F6"))
        self._act(r, "Run the batch", self.run_batch, QKeySequence("F7"))
        self._act(r, "■ Stop", self.runner.stop, QKeySequence("Esc"))
        v = mb.addMenu("&View")
        for i, (key, label) in enumerate(PAGES):
            self._act(v, label, lambda k=key: self.go(k), QKeySequence(f"Ctrl+{i + 1}"))
        h = mb.addMenu("&Help")
        self._act(h, "Method and workflow", lambda: self.go("method"))
        self._act(h, "Open the log", self.open_log)
        self._act(h, "About", self.about)

    def _act(self, menu, text, fn, key=None) -> QAction:
        a = QAction(text, self)
        if key is not None:
            a.setShortcut(key)
        a.triggered.connect(fn)
        menu.addAction(a)
        self.addAction(a)
        return a

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        for p in self.doc.pref("recent", []):
            if os.path.exists(p):
                self.recent_menu.addAction(p, lambda p=p: self.open_path(p))
        if self.recent_menu.isEmpty():
            a = self.recent_menu.addAction("(none)")
            a.setEnabled(False)

    # ---------------------------------------------------------------- nav
    def go(self, key: str) -> None:
        self.tabs.setCurrentWidget(self.pages[key])

    def status(self, msg: str, ms: int = 8000) -> None:
        self.status_lab.setText(msg)
        print(msg, flush=True)
        if ms:
            QTimer.singleShot(ms, lambda m=msg: self.status_lab.text() == m and self.status_lab.setText(""))

    # -------------------------------------------------------------- files
    def start(self, path: str | None) -> None:
        if path:
            self.open_path(path)
        elif not self.smoke and self.doc.restore_autosave():
            self.status("Restored where you left off")
        else:
            self.doc.replace(Project.from_preset(), None)

    def _ok_to_drop(self) -> bool:
        if not self.doc.dirty or self.smoke:
            return True
        r = QMessageBox.question(self, APP, f"Save the changes to '{self.doc.project.name}' first?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

    def new_project(self, preset: str) -> None:
        if self._ok_to_drop():
            self.doc.replace(Project.from_preset(preset), None)
            self.go("fibre")
            self.status(f"New project from '{preset}'")

    def new_stage_project(self, stage_path: str, suffix: str = " – next stage") -> None:
        if not self._ok_to_drop():
            return
        p = Project.from_json(self.doc.project.to_json())
        p.name = self.doc.project.name + suffix
        p.seed = dict(p.seed, source="stage", rescale_energy=False)
        p.stage_file = stage_path
        self.doc.replace(p, None, dirty=True)
        self.go("launch")
        self.status("New stage seeded with the previous output — set its fibre and pump, then ▶ Simulate")

    def open_dialog(self) -> None:
        if not self._ok_to_drop():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", projects_dir(),
                                              f"AmpStudio projects (*{EXT});;fibamp run setups (*.json);;All files (*)")
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        try:
            self.doc.open(path)
            self.status(f"Opened {path}")
        except Exception as e:  # noqa: BLE001
            self._error("Open", f"Could not open {path}:\n{e}")

    def save(self) -> bool:
        if not self.doc.path:
            return self.save_as()
        try:
            self.doc.save()
            self.status(f"Saved {self.doc.path}")
            return True
        except OSError as e:
            self._error("Save", str(e))
            return False

    def save_as(self) -> bool:
        from fibamp.project import slug
        default = os.path.join(projects_dir(), slug(self.doc.project.name) + EXT)
        path, _ = QFileDialog.getSaveFileName(self, "Save project", default, f"AmpStudio projects (*{EXT})")
        if not path:
            return False
        if not path.endswith(EXT):
            path += EXT
        try:
            self.doc.save(path)
            self.status(f"Saved {path}")
            return True
        except OSError as e:
            self._error("Save", str(e))
            return False

    def import_setup(self) -> None:
        self.open_dialog()

    def export_setup(self) -> None:
        from fibamp.io import run_config_json
        path, _ = QFileDialog.getSaveFileName(self, "Export run setup", os.path.join(projects_dir(), "run_setup.json"),
                                              "JSON (*.json)")
        if path:
            ds, pump, seed, sim = self.doc.project.configs()
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(run_config_json(ds, pump, seed, sim))
            self.status(f"Run setup exported: {path}")

    def open_data_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(home_dir())))

    def open_log(self) -> None:
        log = home_dir() / "ampstudio.log"
        if log.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(log)))
        else:
            self.status("No log yet (the log is written by the packaged app)")

    def about(self) -> None:
        import numpy
        import plotly
        import PySide6
        import scipy
        QMessageBox.about(self, APP, f"<h3>kayEi AmpStudio {__version__}</h3>"
                          "<p>Pulsed Er / Er:Yb fibre amplifier and Q-switch designer (fibamp engine).</p>"
                          f"<p>Portable: all data in<br><code>{home_dir()}</code></p>"
                          f"<p style='color:{T.MUTED}'>Python {sys.version.split()[0]} · numpy {numpy.__version__} · "
                          f"scipy {scipy.__version__} · plotly {plotly.__version__} · Qt {PySide6.__version__}</p>")

    # ---------------------------------------------------------------- runs
    def _start(self, task: str, fn, *args) -> bool:
        if self.runner.is_busy():
            self.status(f"Busy: {TASK_LABEL.get(self.runner.task, self.runner.task)} — stop it first (Esc)")
            return False
        self.last_error = ""
        return self.runner.start(task, fn, *args)

    def simulate(self) -> None:
        snap = self.doc.project.to_json()
        self._inputs = self.pages["results"]._snapshot()
        if self._start("simulate", tasks.simulate, snap):
            self.go("results")

    def run_qswitch(self) -> None:
        if self._start("qswitch", tasks.qswitch, self.doc.project.to_json()):
            self.go("qswitch")

    def run_batch(self) -> None:
        try:
            spec = self.pages["batch"].spec_for_run()
        except Exception as e:  # noqa: BLE001
            self._error("Batch", str(e))
            return
        if spec.n_cases() > 200 and not self.smoke and QMessageBox.question(
                self, "Batch", f"{spec.n_cases()} cases: this can take a long time. Run it?") != QMessageBox.Yes:
            return
        if self._start("batch", tasks.batch, spec.to_dict(), int(self.doc.project.batch.get("workers") or 0)):
            self.go("batch")

    def _busy(self, on: bool) -> None:
        self.panel.busy(on)
        if on:
            self._tick.start()
            self.panel.msg.setText(TASK_LABEL.get(self.runner.task, "") + "…")
        else:
            self._tick.stop()
            self.elapsed.setText("")

    def _progress(self, task: str, frac: float, msg: str) -> None:
        self.panel.bar.setValue(int(frac * 1000))
        self.panel.msg.setText(f"{TASK_LABEL.get(task, task)}: {msg}" if msg else TASK_LABEL.get(task, task))

    def _done(self, task: str, result) -> None:
        self.panel.msg.setText("")
        if task == "simulate":
            self.pages["results"].show_result(result, self._inputs)
            s = result.summary
            self.status(f"Done in {result.timing.get('total_s', 0):.1f} s: {s['output pulse energy (uJ)']:.4g} µJ, "
                        f"{s['energy gain (dB)']:.2f} dB" + (f" — {len(result.warnings)} warning(s)" if result.warnings else ""))
        elif task == "qswitch":
            self.pages["qswitch"].show_result(result)
            self.status("Q-switched pulse designed")
        elif task == "batch":
            self.pages["batch"].show_result(result)
            self.status("Batch stopped — finished cases kept" if result.get("stopped") else "Batch finished")

    def _failed(self, task: str, err: str) -> None:
        self.panel.msg.setText("")
        if not err:
            self.status(f"{TASK_LABEL.get(task, task)}: stopped")
            return
        self.last_error = err
        first = err.strip().splitlines()[0] if err.strip() else err
        self._error(TASK_LABEL.get(task, task), f"{first}\n\nDetails:\n{err}")

    def _error(self, title: str, text: str) -> None:
        print(f"ERROR {title}: {text}", flush=True)
        if self.smoke:
            return
        box = QMessageBox(QMessageBox.Warning, title, text.split("\n\nDetails:")[0], parent=self)
        if "\n\nDetails:" in text:
            box.setDetailedText(text.split("\n\nDetails:\n", 1)[1])
        box.exec()

    # ---------------------------------------------------------------- close
    def closeEvent(self, ev) -> None:  # noqa: N802
        if self.runner.is_busy() and not self.smoke:
            if QMessageBox.question(self, APP, "A computation is running. Stop it and quit?") != QMessageBox.Yes:
                ev.ignore()
                return
        if not self._ok_to_drop():
            ev.ignore()
            return
        self.runner.wait(3000)
        try:
            self.doc.autosave()
            self.doc.set_pref("geometry", bytes(self.saveGeometry().toBase64()).decode())
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        ev.accept()
