"""Results of the last amplifier run: verdict, KPIs, every chart, the numbers and the exports."""
from __future__ import annotations

import html
import json
import os
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

from fibamp import io as fio
from fibamp import plots
from fibamp.paths import sub
from fibamp.project import slug

from ..charts import ChartGrid
from ..widgets import Banner, KpiRow, Table, fmt_energy_uJ, fmt_power_kW, legend_below, section

KPIS = [("Output pulse energy", "Energy of the amplified pulse (the time window around it)"),
        ("Energy gain", "Output / seed pulse energy"),
        ("Peak power", "Peak of the output pulse"),
        ("Output FWHM", "Output pulse width (input width below)"),
        ("Pulse share of output", "Pulse energy / (pulse + forward ASE + Raman) per period"),
        ("Efficiency", "Signal power added / launched pump power")]

VIEWS = ["Pulse", "Spectrum", "Along the fibre", "Evolution", "Transverse & modes", "SBS & thermal",
         "Between pulses", "Numbers"]

INPUT_PARTS = ("fiber", "pump", "seed", "sim")


def _critical(w: str) -> bool:
    """The engine opens its show-stoppers with a capitalised headline ("SBS RUNAWAY", "NO PERIODIC
    STEADY STATE", "PARASITIC LASING / AT THRESHOLD", "PRE-LASING"); a short tag like "SBS:" is not one."""
    i = 0
    while i < len(w) and (w[i].isupper() or w[i] in " -/"):
        i += 1
    return len(w[:i].strip(" -/")) >= 8


class ResultsPage(QWidget):
    def __init__(self, win):
        super().__init__(win)
        self.win, self.doc = win, win.doc
        self.res = None
        self.inputs = None
        self._dirty_views: set[str] = set()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        head = QHBoxLayout()
        self.title = section("📈 Results")
        head.addWidget(self.title)
        head.addStretch(1)
        self.b_stage = QPushButton("Save stage…")
        self.b_stage.setToolTip("The output field + forward ASE as a .fibamp.npz file: the seed of a next stage")
        self.b_next = QPushButton("Next stage from this output…")
        self.b_next.setToolTip("Seed a following amplifier stage with this output")
        self.b_csv = QPushButton("Export CSV…")
        self.b_csv.setToolTip("Pulse, spectra and along-the-fibre profiles as CSV files")
        self.b_json = QPushButton("Export report…")
        self.b_json.setToolTip("Summary, power budget, warnings, fibre and full configuration as JSON")
        for b in (self.b_stage, self.b_next, self.b_csv, self.b_json):
            head.addWidget(b)
            b.setEnabled(False)
        lay.addLayout(head)
        self.kpis = KpiRow(KPIS)
        lay.addWidget(self.kpis)
        self.stale = Banner(kind="notice")
        lay.addWidget(self.stale)
        self.danger = Banner(kind="danger")
        lay.addWidget(self.danger)
        self.warn = Banner(kind="notice")
        lay.addWidget(self.warn)

        opts = QHBoxLayout()
        opts.addWidget(QLabel("OSA resolution"))
        self.rbw = QComboBox()
        for v in (0.01, 0.05, 0.1, 0.5, 1.0, 2.0):
            self.rbw.addItem(f"{v:g} nm", v)
        self.rbw.setCurrentIndex(2)
        opts.addWidget(self.rbw)
        self.logp = QCheckBox("Pulse on a log scale")
        opts.addWidget(self.logp)
        self.logevo = QCheckBox("Evolution maps in dB")
        opts.addWidget(self.logevo)
        opts.addStretch(1)
        self.timing = QLabel()
        self.timing.setProperty("muted", True)
        opts.addWidget(self.timing)
        lay.addLayout(opts)

        self.tabs = QTabWidget()
        self.grids: dict[str, ChartGrid] = {}
        for v in VIEWS:
            if v == "Numbers":
                w = QSplitter(Qt.Horizontal)
                self.t_summary, self.t_budget, self.t_fibre = Table(), Table(), Table()
                for t, lab in ((self.t_summary, "Summary"), (self.t_budget, "Average power budget"),
                               (self.t_fibre, "Fibre")):
                    box = QWidget()
                    bl = QVBoxLayout(box)
                    bl.setContentsMargins(4, 4, 4, 4)
                    bl.addWidget(section(lab))
                    bl.addWidget(t, 1)
                    w.addWidget(box)
                self.notes = Banner(kind="info")
                holder = QWidget()
                hl = QVBoxLayout(holder)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.addWidget(w, 1)
                hl.addWidget(self.notes)
                self.tabs.addTab(holder, v.replace("&", "&&"))
            else:
                g = ChartGrid(cols=2 if v in ("Pulse", "Spectrum", "Evolution", "Transverse & modes", "SBS & thermal") else 1,
                              empty="Press <b>▶ Simulate</b> (F5) to run the amplifier.")
                self.grids[v] = g
                self.tabs.addTab(g, v.replace("&", "&&"))
        lay.addWidget(self.tabs, 1)

        self.tabs.currentChanged.connect(lambda _i: self._render_current())
        self.rbw.currentIndexChanged.connect(lambda _i: self._invalidate("Spectrum"))
        self.logp.toggled.connect(lambda _b: self._invalidate("Pulse"))
        self.logevo.toggled.connect(lambda _b: self._invalidate("Evolution"))
        self.b_stage.clicked.connect(self.save_stage)
        self.b_next.clicked.connect(self.next_stage)
        self.b_csv.clicked.connect(self.export_csv)
        self.b_json.clicked.connect(self.export_json)
        self.doc.changed.connect(self._doc_changed)

    # --------------------------------------------------------------- state
    def _snapshot(self) -> str:
        p = self.doc.project
        return json.dumps({k: getattr(p, k) for k in INPUT_PARTS} | {"stage": p.stage_file, "csv": p.seed_csv},
                          sort_keys=True, default=str)

    def _doc_changed(self, part: str) -> None:
        if self.res is None:
            return
        if part == "all":
            self.clear()
            return
        if self._snapshot() != self.inputs:
            self.stale.set("Inputs changed since this run: the results below are for the previous settings. "
                           "Press <b>▶ Simulate</b> (F5) to update.", "notice")
        else:
            self.stale.set("")

    def clear(self) -> None:
        self.res = None
        self.kpis.clear()
        for b in (self.b_stage, self.b_next, self.b_csv, self.b_json):
            b.setEnabled(False)
        for w in (self.stale, self.danger, self.warn):
            w.set("")
        for g in self.grids.values():
            g.show_message("Press <b>▶ Simulate</b> (F5) to run the amplifier.")
        for t in (self.t_summary, self.t_budget, self.t_fibre):
            t.set_rows([])
        self.notes.set("")
        self.timing.setText("")

    def show_result(self, res, inputs: str) -> None:
        self.res, self.inputs = res, inputs
        s = res.summary
        self.kpis.set("Output pulse energy", fmt_energy_uJ(s["output pulse energy (uJ)"]),
                      f"seed {fmt_energy_uJ(s['seed energy (uJ)'])}")
        g = s["energy gain (dB)"]
        self.kpis.set("Energy gain", f"{g:.2f} dB", f"g₀L before the pulse {s['small-signal gain before pulse, g0L (dB)']:.1f} dB")
        self.kpis.set("Peak power", fmt_power_kW(s["output peak power (kW)"]),
                      f"{s['peak intensity (GW/cm2)']:.3g} GW/cm² · B = {s['B-integral (rad)']:.2g} rad")
        self.kpis.set("Output FWHM", f"{s['output FWHM (ns)']:.4g} ns", f"input {s['input FWHM (ns)']:.4g} ns")
        frac = s["pulse fraction of output energy (%)"]
        self.kpis.set("Pulse share of output", f"{frac:.1f} %",
                      f"fwd ASE {s['forward ASE, average (W)']*1e3:.3g} mW · bwd {s['backward ASE at input, average (W)']*1e3:.3g} mW",
                      color=None if frac > 50 else "#f59e0b")
        self.kpis.set("Efficiency", f"{s['optical efficiency vs launched pump (%)']:.1f} %",
                      f"pump absorbed {s['pump absorbed (%)']:.0f} %")
        crit = [w for w in res.warnings if _critical(w)]
        other = [w for w in res.warnings if not _critical(w)]
        self.danger.set("<br>".join(f"⛔ {html.escape(w)}" for w in crit), "danger")
        self.warn.set("<br>".join(f"⚠ {html.escape(w)}" for w in other), "notice")
        self.stale.set("")
        conv = res.diag.get("conv", {})
        self.timing.setText(f"{res.timing.get('total_s', 0):.1f} s · {conv.get('iterations', 0)} periods to steady state · "
                            f"photon ledger {res.diag.get('photon_balance_error', 0)*100:+.2f} % · "
                            f"model: {res.diag.get('nl_model', '')}")
        self.t_summary.set_kv(s)
        self.t_budget.set_kv(res.budget)
        self.t_fibre.set_kv(res.fiber.summary(res.seed.lam0))
        self.notes.set("<br>".join(f"• {html.escape(n)}" for n in res.notes), "info")
        for b in (self.b_stage, self.b_next, self.b_csv, self.b_json):
            b.setEnabled(True)
        self._dirty_views = set(self.grids)
        self._render_current()

    # -------------------------------------------------------------- charts
    def _invalidate(self, view: str) -> None:
        if self.res is not None:
            self._dirty_views.add(view)
            self._render_current()

    def _render_current(self) -> None:
        v = VIEWS[self.tabs.currentIndex()]
        if self.res is None or v not in self._dirty_views:
            return
        self._dirty_views.discard(v)
        figs = []
        for make, wide in self._figures(v):
            try:
                figs.append((make(), wide))
            except Exception as e:  # noqa: BLE001 - one broken chart must not hide the others
                print(f"chart failed in {v}: {type(e).__name__}: {e}", flush=True)
        if figs:
            self.grids[v].set_figures(figs)
        else:
            self.grids[v].show_message(self._empty_text(v))

    def _empty_text(self, v: str) -> str:
        if v == "Transverse & modes":
            return "Single ring and a single signal mode: nothing transverse to show.<br>Set radial rings > 1 or the " \
                   "multimode signal model on the Fibre page."
        return "Nothing to show for this run."

    def _figures(self, v: str):
        r = self.res
        rbw = float(self.rbw.currentData())
        if v == "Pulse":
            return [(lambda: plots.pulse_figure(r, log=self.logp.isChecked()), False),
                    (lambda: plots.pulse_figure(r, normalised=True), False)]
        if v == "Spectrum":
            return [(lambda: plots.osa_figure(r, rbw, "output"), False),
                    (lambda: plots.osa_figure(r, rbw, "input"), False),
                    (lambda: plots.spectrum_figure(r), True)]
        if v == "Along the fibre":
            return [(lambda: legend_below(plots.profile_figure(r), 760), True)]
        if v == "Evolution":
            log = self.logevo.isChecked()
            return [(lambda: plots.pulse_evolution_figure(r, log=log), False),
                    (lambda: plots.spectrum_evolution_figure(r), False),
                    (lambda: plots.ase_evolution_figure(r, +1), False),
                    (lambda: plots.ase_evolution_figure(r, -1), False)]
        if v == "Transverse & modes":
            out = []
            if r.fiber.nr > 1:
                out.append((lambda: plots.ring_inversion_figure(r), True))
            if len(r.pulse.get("mode_names", [])) > 1:
                out.append((lambda: plots.mode_energy_figure(r), True))
            return out
        if v == "SBS & thermal":
            out = [(lambda: plots.thermal_figure(r), r.sbs is None)]
            if r.sbs is not None:
                out.insert(0, (lambda: plots.sbs_figure(r), False))
            return out
        if v == "Between pulses":
            return [(lambda: plots.trace_figure(r), True)]
        return []

    # -------------------------------------------------------------- exports
    def _stage_name(self) -> str:
        return f"{slug(self.doc.project.name)}-{time.strftime('%Y%m%d-%H%M%S')}"

    def save_stage(self) -> None:
        if self.res is None:
            return
        default = str(sub("stages") / f"{self._stage_name()}.fibamp.npz")
        path, _ = QFileDialog.getSaveFileName(self, "Save stage", default, "Stage files (*.npz)")
        if path:
            fio.save_stage(self.res.to_stage(self.doc.project.name), path)
            self.win.status(f"Stage saved: {path}")

    def next_stage(self) -> None:
        if self.res is None:
            return
        path = str(sub("stages") / f"{self._stage_name()}.fibamp.npz")
        fio.save_stage(self.res.to_stage(self.doc.project.name), path)
        box = QMessageBox(self)
        box.setWindowTitle("Next stage")
        box.setText("The output is saved as a stage file. Where should the next stage be designed?")
        b_new = box.addButton("New project (copy of this one)", QMessageBox.AcceptRole)
        b_here = box.addButton("This project", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is b_new:
            self.win.new_stage_project(path)
        elif box.clickedButton() is b_here:
            self.win.pages["launch"].set_stage(path)
            self.win.go("launch")

    def export_csv(self) -> None:
        if self.res is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Export CSV files to", str(sub("runs")))
        if not folder:
            return
        base = self._stage_name()
        rbw = float(self.rbw.currentData())
        for name, text in ((f"{base}-pulse.csv", fio.pulse_csv(self.res)),
                           (f"{base}-spectrum.csv", fio.spectrum_csv(self.res, rbw)),
                           (f"{base}-profile.csv", fio.profile_csv(self.res))):
            with open(os.path.join(folder, name), "w", encoding="utf-8") as fh:
                fh.write(text)
        self.win.status(f"Exported pulse, spectrum and profile CSV files to {folder}")

    def export_json(self) -> None:
        if self.res is None:
            return
        default = str(sub("runs") / f"{self._stage_name()}-report.json")
        path, _ = QFileDialog.getSaveFileName(self, "Export report", default, "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(fio.report_json(self.res))
            self.win.status(f"Report saved: {path}")
