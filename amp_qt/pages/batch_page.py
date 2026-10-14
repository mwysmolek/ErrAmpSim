"""Batch: sweep the design over parameter axes on every CPU core, rank, report."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSpinBox, QSplitter, QTabWidget, QVBoxLayout,
                               QWidget)

from fibamp.batch import AXES, METRICS, BatchSpec, _apply, list_batches, load_batch, parse_axis, passes, ranked
from fibamp.io import list_fibres

from ..charts import profile
from ..widgets import NO_WHEEL, Banner, Table, muted, section


def _combo(items, current=None) -> QComboBox:
    c = QComboBox()
    for k, lab in items:
        c.addItem(lab, k)
    if current is not None:
        c.setCurrentIndex(max(0, c.findData(current)))
    c.installEventFilter(NO_WHEEL)
    c.setFocusPolicy(Qt.StrongFocus)
    return c


AXIS_ITEMS = [(k, f"{lab} ({unit})" if unit else lab) for k, (lab, unit, _) in AXES.items()]
METRIC_ITEMS = [(k, f"{lab} ({unit})" if unit else lab) for k, (_, lab, unit) in METRICS.items()]


def status_of(r: dict, spec) -> str:
    """Why a finished case does (not) qualify, in words."""
    m = r["metrics"]
    if passes(r, spec.constraints):
        return "ok"
    if m.get("sbs_runaway"):
        return "SBS runaway"
    if not m.get("converged"):
        return "not converged"
    bad = []
    for key, (op, val) in (spec.constraints or {}).items():
        v = m.get(key)
        lab = METRICS[key][1] if key in METRICS else key
        if v is None:
            bad.append(f"{lab} n/a")
        elif (op == "<" and not v < val) or (op == ">" and not v > val):
            bad.append(f"{lab} {v:.3g} (needs {op} {val:g})")
    return "fails: " + "; ".join(bad) if bad else "fails a constraint"


class BatchPage(QWidget):
    def __init__(self, win):
        super().__init__(win)
        self.win, self.doc = win, win.doc
        self._loading = False
        self.spec = None
        self.rows = []
        self.folder = None
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)

        # ---------------------------------------------------------- set-up
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(section("🧮 Batch"))
        ll.addWidget(muted("The project as it is now, swept over the axes below. Each case runs in its own process on "
                           "every core but one; an interrupted batch keeps its finished cases and resumes."))
        g = QGridLayout()
        g.addWidget(QLabel("Name"), 0, 0)
        self.name = QLineEdit()
        g.addWidget(self.name, 0, 1, 1, 3)
        g.addWidget(QLabel("Accuracy"), 1, 0)
        self.acc = _combo([("draft", "draft (fast)"), ("standard", "standard"), ("fine", "fine"),
                           ("", "as on the Numerics page")])
        g.addWidget(self.acc, 1, 1)
        g.addWidget(QLabel("Workers"), 1, 2)
        self.workers = QSpinBox()
        self.workers.setRange(0, 256)
        self.workers.setSpecialValueText(f"auto ({max(1, (os.cpu_count() or 2) - 1)})")
        g.addWidget(self.workers, 1, 3)
        ll.addLayout(g)

        axes_box = QGroupBox("Sweep axes  ·  start:stop:step, a list a, b, c, or one value")
        self.axes_lay = QVBoxLayout(axes_box)
        self.axes_grid = QGridLayout()
        self.axes_lay.addLayout(self.axes_grid)
        b_axis = QPushButton("+ Axis")
        b_axis.clicked.connect(lambda: (self._add_axis("pump_co_W", "1.0"), self._edited()))
        self.axes_lay.addWidget(b_axis, 0, Qt.AlignLeft)
        self.n_cases = QLabel()
        self.axes_lay.addWidget(self.n_cases)
        ll.addWidget(axes_box)
        self.axis_rows: list[tuple] = []

        obj_box = QGroupBox("Best case")
        ol = QGridLayout(obj_box)
        ol.addWidget(QLabel("Objective"), 0, 0)
        self.objective = _combo(METRIC_ITEMS)
        ol.addWidget(self.objective, 0, 1)
        self.maximize = _combo([(True, "maximise"), (False, "minimise")])
        ol.addWidget(self.maximize, 0, 2)
        ol.addWidget(QLabel("Subject to"), 1, 0, Qt.AlignTop)
        self.cons_grid = QGridLayout()
        ol.addLayout(self.cons_grid, 1, 1, 1, 2)
        b_con = QPushButton("+ Constraint")
        b_con.clicked.connect(lambda: (self._add_con("B_rad", "<", 10.0), self._edited()))
        ol.addWidget(b_con, 2, 1, Qt.AlignLeft)
        self.keep = QCheckBox("Keep pulse and spectrum traces (for the report's overlays)")
        ol.addWidget(self.keep, 3, 0, 1, 3)
        ll.addWidget(obj_box)
        self.con_rows: list[tuple] = []

        self.err = Banner(kind="danger")
        ll.addWidget(self.err)
        row = QHBoxLayout()
        self.b_run = QPushButton("▶ Run batch")
        self.b_run.setProperty("primary", True)
        row.addWidget(self.b_run)
        row.addStretch(1)
        ll.addLayout(row)
        prev = QHBoxLayout()
        prev.addWidget(QLabel("Earlier batches"))
        self.prev = QComboBox()
        prev.addWidget(self.prev, 1)
        b_prev = QPushButton("Show")
        prev.addWidget(b_prev)
        ll.addLayout(prev)
        ll.addStretch(1)
        split.addWidget(left)

        # --------------------------------------------------------- results
        right = QWidget()
        rl = QVBoxLayout(right)
        head = QHBoxLayout()
        self.r_title = section("Results")
        head.addWidget(self.r_title)
        head.addStretch(1)
        self.b_load = QPushButton("Load selected case into the project")
        self.b_load.setToolTip("Writes that case's values into the project's fibre, pump and seed (double-click a row)")
        self.b_report = QPushButton("Open report in browser")
        self.b_folder = QPushButton("Open folder")
        for b in (self.b_load, self.b_report, self.b_folder):
            b.setEnabled(False)
            head.addWidget(b)
        rl.addLayout(head)
        self.best = Banner(kind="info")
        rl.addWidget(self.best)
        self.rtabs = QTabWidget()
        self.table = Table()
        self.rtabs.addTab(self.table, "Cases")
        self.report = QWebEngineView()
        self.report.setPage(QWebEnginePage(profile(), self.report))
        self.report.setHtml("<body style='background:#001a45;color:#9fb3c8;font-family:Segoe UI;padding:40px'>"
                            "The batch report (best case, sensitivities, maps) appears here.</body>")
        self.rtabs.addTab(self.report, "Report")
        rl.addWidget(self.rtabs, 1)
        split.addWidget(right)
        split.setSizes([520, 860])

        for w in (self.name,):
            w.editingFinished.connect(self._edited)
        for w in (self.acc, self.objective, self.maximize):
            w.currentIndexChanged.connect(lambda _i: self._edited())
        self.workers.valueChanged.connect(lambda _v: self._edited())
        self.keep.toggled.connect(lambda _b: self._edited())
        self.b_run.clicked.connect(win.run_batch)
        b_prev.clicked.connect(self.show_previous)
        self.b_load.clicked.connect(lambda: self.load_case(self.table.selected_row()))
        self.table.rowActivated.connect(self.load_case)
        self.b_report.clicked.connect(lambda: self.folder and QDesktopServices.openUrl(
            QUrl.fromLocalFile(os.path.join(self.folder, "report.html"))))
        self.b_folder.clicked.connect(lambda: self.folder and QDesktopServices.openUrl(QUrl.fromLocalFile(self.folder)))
        self.doc.changed.connect(lambda part: self.reload() if part in ("all", "batch") else None)
        self.reload()

    # ------------------------------------------------------------ editing
    def _add_axis(self, key: str, text: str) -> None:
        r = len(self.axis_rows)
        c = _combo(AXIS_ITEMS, key)
        e = QLineEdit(str(text))
        e.setPlaceholderText("2:6:1   or   1, 2, 5")
        n = QLabel()
        n.setMinimumWidth(70)
        n.setProperty("muted", True)
        rm = QPushButton("✕")
        rm.setFixedWidth(30)
        row = (c, e, n, rm)
        c.currentIndexChanged.connect(lambda _i: self._edited())
        e.editingFinished.connect(self._edited)
        rm.clicked.connect(lambda: self._remove(self.axis_rows, row))
        for i, w in enumerate(row):
            self.axes_grid.addWidget(w, r, i)
        self.axes_grid.setColumnStretch(1, 1)
        self.axis_rows.append(row)

    def _add_con(self, key: str, op: str, val: float) -> None:
        r = len(self.con_rows)
        c = _combo(METRIC_ITEMS, key)
        o = _combo([("<", "<"), (">", ">")], op)
        v = QDoubleSpinBox()
        v.setRange(-1e12, 1e12)
        v.setDecimals(4)
        v.setValue(float(val))
        v.setKeyboardTracking(False)
        v.installEventFilter(NO_WHEEL)
        rm = QPushButton("✕")
        rm.setFixedWidth(30)
        row = (c, o, v, rm)
        c.currentIndexChanged.connect(lambda _i: self._edited())
        o.currentIndexChanged.connect(lambda _i: self._edited())
        v.valueChanged.connect(lambda _x: self._edited())
        rm.clicked.connect(lambda: self._remove(self.con_rows, row))
        for i, w in enumerate(row):
            self.cons_grid.addWidget(w, r, i)
        self.con_rows.append(row)

    def _remove(self, rows: list, row: tuple) -> None:
        rows.remove(row)
        for w in row:
            w.deleteLater()
        self._edited()
        self.reload()

    def _clear(self, grid, rows) -> None:
        for row in rows:
            for w in row:
                grid.removeWidget(w)
                w.deleteLater()
        rows.clear()

    def draft(self) -> dict:
        axes = {}
        for c, e, _, _ in self.axis_rows:
            axes[c.currentData()] = e.text().strip()
        cons = {c.currentData(): [o.currentData(), v.value()] for c, o, v, _ in self.con_rows}
        return dict(name=self.name.text().strip() or "batch", axes=axes, objective=self.objective.currentData(),
                    maximize=bool(self.maximize.currentData()), constraints=cons, workers=self.workers.value(),
                    keep_traces=self.keep.isChecked(), accuracy=self.acc.currentData())

    def _edited(self) -> None:
        if self._loading:
            return
        self.doc.update("batch", self.draft())
        self._count()

    def reload(self) -> None:
        self._loading = True
        try:
            b = self.doc.project.batch
            self.name.setText(b.get("name", "batch"))
            self.acc.setCurrentIndex(max(0, self.acc.findData(b.get("accuracy", "draft") or "")))
            self.workers.setValue(int(b.get("workers") or 0))
            self.objective.setCurrentIndex(max(0, self.objective.findData(b.get("objective", "E_out_uJ"))))
            self.maximize.setCurrentIndex(0 if b.get("maximize", True) else 1)
            self.keep.setChecked(bool(b.get("keep_traces", True)))
            self._clear(self.axes_grid, self.axis_rows)
            for k, text in (b.get("axes") or {}).items():
                self._add_axis(k, text)
            self._clear(self.cons_grid, self.con_rows)
            for k, (op, v) in (b.get("constraints") or {}).items():
                self._add_con(k, op, v)
        finally:
            self._loading = False
        self._count()
        self.reload_previous()

    def _count(self) -> None:
        total, errs = 1, []
        for c, e, n, _ in self.axis_rows:
            try:
                vals = parse_axis(e.text())
                if c.currentData() == "fibre":
                    missing = [str(v) for v in vals if str(v) not in set(list_fibres())]
                    if missing:
                        raise ValueError(f"not in the fibre library: {', '.join(missing)}")
                n.setText(f"{len(vals)} value{'s' if len(vals) != 1 else ''}")
                total *= max(1, len(vals))
            except ValueError as ex:
                n.setText("✕")
                errs.append(f"{c.currentText()}: {ex}")
        keys = [c.currentData() for c, *_ in self.axis_rows]
        if len(keys) != len(set(keys)):
            errs.append("an axis appears twice")
        self.n_cases.setText(f"<b>{total}</b> case{'s' if total != 1 else ''}")
        self.err.set("<br>".join(errs), "danger")
        self.b_run.setEnabled(not errs)

    def reload_previous(self) -> None:
        cur = self.prev.currentText()
        self.prev.clear()
        self.prev.addItems(list_batches())
        if cur:
            self.prev.setCurrentText(cur)

    # ------------------------------------------------------------ results
    def show_previous(self) -> None:
        from fibamp.batch import batch_dir
        name = self.prev.currentText()
        if not name:
            return
        folder = batch_dir(name)
        try:
            spec, rows = load_batch(folder)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Batch", f"Could not load '{name}': {e}")
            return
        self.show_result(dict(spec=spec, rows=rows, folder=folder, stopped=False))

    def show_result(self, out: dict) -> None:
        from fibamp.report import write_reports
        self.spec, self.rows, self.folder = out["spec"], out["rows"], out["folder"]
        spec = self.spec
        n_ok = sum(1 for r in self.rows if r.get("ok"))
        n_pass = sum(1 for r in self.rows if passes(r, spec.constraints))
        self.r_title.setText(f"Results · {spec.name}")
        try:
            write_reports(spec, self.rows, self.folder)
            self.report.load(QUrl.fromLocalFile(os.path.join(self.folder, "report.html")))
        except Exception as e:  # noqa: BLE001
            print(f"batch report failed: {e}", flush=True)
        best = ranked(spec, self.rows)
        obj_lab = METRICS[spec.objective][1]
        unit = METRICS[spec.objective][2]
        msg = [f"{len(self.rows)} of {spec.n_cases()} cases run, {n_ok} finished, {n_pass} meet the constraints"
               + (" — <b>stopped</b>; run again to resume" if out.get("stopped") else "")]
        if best:
            b = best[0]
            vals = ", ".join(f"{AXES[k][0]} = {v:g} {AXES[k][1]}" if isinstance(v, (int, float)) else f"{AXES[k][0]} = {v}"
                             for k, v in b["values"].items())
            msg.append(f"<b>Best {obj_lab}: {b['metrics'][spec.objective]:.4g} {unit}</b> at {vals or 'the base design'}")
        else:
            msg.append("No case meets the constraints.")
        self.best.set("<br>".join(msg), "info" if best else "notice")
        keys = [k for k in spec.axes if len(spec.axes[k])]
        cols = keys + ["status", spec.objective] + [k for k in ("E_out_uJ", "gain_dB", "peak_kW", "pulse_frac_pct",
                                                                "sbs_pct", "B_rad", "eff_launched_pct")
                                                    if k != spec.objective] + ["seconds"]
        recs = []
        best_i = best[0]["index"] if best else None
        for r in self.rows:
            rec = {k: r["values"].get(k) for k in keys}
            rec["_row"] = r
            if r.get("ok"):
                rec.update(r["metrics"])
                rec["status"] = "★ best" if r["index"] == best_i else status_of(r, spec)
            else:
                rec["status"] = f"error: {r.get('error', '')}"
            rec["seconds"] = r.get("seconds")
            recs.append(rec)
        headers = {k: f"{AXES[k][0]} ({AXES[k][1]})" if AXES[k][1] else AXES[k][0] for k in keys}
        headers.update({k: f"{METRICS[k][1]} ({METRICS[k][2]})" for k in METRICS})
        colors = lambda rec, col: (("#22c55e" if rec["status"].startswith("★") else
                                    "#ef4444" if rec["status"].startswith("error") else
                                    "#f59e0b" if rec["status"] != "ok" else None) if col == "status" else None)  # noqa: E731
        self.table.set_rows(recs, cols, headers=headers, colors=colors)
        for b in (self.b_load, self.b_report, self.b_folder):
            b.setEnabled(True)
        self.reload_previous()

    def load_case(self, rec) -> None:
        if not rec or self.spec is None:
            return
        r = rec.get("_row") or {}
        values = r.get("values", {})
        try:
            fib, pump, seed = _apply(self.spec, values)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Batch", f"Could not apply this case: {e}")
            return
        if QMessageBox.question(self, "Load case", "Replace the project's fibre, pump and seed with this case?\n\n"
                                + "\n".join(f"{AXES[k][0]} = {v}" for k, v in values.items())) != QMessageBox.Yes:
            return
        self.doc.update("fiber", fib)
        self.doc.update("pump", pump)
        self.doc.update("seed", seed)
        self.win.status("Case loaded into the project — press ▶ Simulate for its full results")
        self.win.go("results")

    def spec_for_run(self) -> BatchSpec:
        from fibamp.cli import batch_spec
        p = self.doc.project
        p.batch = self.draft()
        return batch_spec(p)
