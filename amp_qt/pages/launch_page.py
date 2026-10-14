"""Pump and seed: what goes into the fibre, with a live preview of the seed pulse."""
from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSplitter, QVBoxLayout,
                               QWidget)

from fibamp.config import SeedConfig, SimConfig
from fibamp.plots import LAYOUT, C_OUT
from fibamp.project import parse_seed_csv

from ..charts import ChartGrid
from ..forms import F, Form
from ..widgets import Banner, muted, scroll, section

PULSED = lambda v: bool(v.get("pulsed"))                       # noqa: E731
ANALYTIC = lambda v: v.get("source") == "analytic"             # noqa: E731
STAGE = lambda v: v.get("source") == "stage"                   # noqa: E731


def pump_groups():
    src = lambda side, title: (title, [                         # noqa: E731
        F(f"{side}.power_W", "Power", unit="W", lo=0, hi=1000, dec=3, step=0.1),
        F(f"{side}.wavelength_nm", "Wavelength", unit="nm", lo=800, hi=1700, dec=1, step=1),
        F(f"{side}.fwhm_nm", "Diode linewidth (FWHM)", unit="nm", lo=0.01, hi=20, dec=2, step=0.1,
          tip="Wavelength-stabilised ~0.5 nm, free-running 3-5 nm"),
        F(f"{side}.n_sub", "Spectral sub-bands", "int", lo=1, hi=41, tip="Bands used to represent the diode line"),
    ])
    return [
        ("Pump launch", [
            F("geometry", "Geometry", "choice", choices=[("cladding", "cladding-pumped"), ("core", "core-pumped")]),
            F("pulsed", "Pulsed pump", "bool", tip="Pump on only for a window before each seed pulse"),
            F("on_us", "Pump pulse length", unit="µs", lo=0.1, hi=1e6, dec=1, step=10, when=PULSED),
            F("lead_us", "Pump ends before the seed by", unit="µs", lo=0, hi=1e6, dec=2, step=1, when=PULSED),
            F("off_fraction", "Simmer level (fraction of on)", lo=0, hi=1, dec=3, step=0.01, when=PULSED),
        ]),
        src("co", "Co-propagating pump"),
        src("counter", "Counter-propagating pump"),
    ]


def seed_groups():
    shape = lambda *s: (lambda v: v.get("source") == "analytic" and v.get("shape") in s)   # noqa: E731
    return [
        ("Seed", [
            F("source", "Source", "choice", choices=[("analytic", "analytic pulse shape"), ("csv", "measured pulse (CSV)"),
                                                     ("stage", "previous stage (.fibamp.npz)")]),
            F("wavelength_nm", "Wavelength", unit="nm", lo=900, hi=1700, dec=2, step=1),
            F("linewidth_nm", "Linewidth (FWHM)", unit="nm", lo=0, hi=50, dec=4, step=0.01,
              tip="0 = transform-limited; broadening raises the SBS threshold"),
            F("rep_rate_kHz", "Repetition rate", unit="kHz", lo=0, hi=1e5, dec=3, step=1,
              tip="Stage import: 0 keeps the previous stage's rate"),
            F("rng_seed", "Noise seed", "int", lo=0, hi=10 ** 6),
        ]),
        ("Pulse", [
            F("shape", "Shape", "choice", when=ANALYTIC,
              choices=[("gaussian", "Gaussian"), ("supergaussian", "super-Gaussian"), ("sech2", "sech²"),
                       ("square", "square"), ("asymmetric", "asymmetric (Q-switch-like)")]),
            F("fwhm_ns", "FWHM", unit="ns", lo=0.001, hi=1e5, dec=3, step=1,
              when=shape("gaussian", "supergaussian", "sech2", "square")),
            F("sg_order", "Super-Gaussian order", lo=1, hi=20, dec=1, when=shape("supergaussian")),
            F("edge_ns", "10-90 % edge", unit="ns", lo=0.001, hi=1e4, dec=3, when=shape("square")),
            F("rise_ns", "Rise (1/e)", unit="ns", lo=0.001, hi=1e4, dec=3, when=shape("asymmetric")),
            F("fall_ns", "Fall (1/e)", unit="ns", lo=0.001, hi=1e4, dec=3, when=shape("asymmetric")),
            F("specify", "Set by", "choice", choices=[("energy", "energy"), ("peak", "peak power")], when=ANALYTIC),
            F("energy_uJ", "Energy", unit="µJ", lo=0, hi=1e6, dec=6, step=0.1,
              when=lambda v: (v.get("source") == "analytic" and v.get("specify") == "energy") or v.get("rescale_energy")),
            F("peak_W", "Peak power", unit="W", lo=0, hi=1e9, dec=3,
              when=lambda v: v.get("source") == "analytic" and v.get("specify") == "peak"),
            F("rescale_energy", "Rescale to the energy above", "bool", when=lambda v: v.get("source") != "analytic"),
        ]),
        ("Between stages (previous-stage seed)", [
            F("interstage_loss_dB", "Isolator + splice loss", unit="dB", lo=0, hi=60, dec=2, step=0.1, when=STAGE),
            F("filter_center_nm", "Band-pass centre", unit="nm", lo=900, hi=1700, dec=2, when=STAGE),
            F("filter_fwhm_nm", "Band-pass FWHM", unit="nm", lo=0, hi=200, dec=2, when=STAGE, tip="0 = no filter"),
            F("filter_order", "Filter order", lo=1, hi=10, dec=1, when=STAGE, tip="1 Gaussian, >1 flatter top"),
        ]),
        ("Modulator leakage", [
            F("cw_leak_dB", "CW leakage vs average seed", "opt", unit="dB", lo=-120, hi=0, dec=1, default=-40.0,
              tip="Off: a perfect modulator. On: CW light at the seed wavelength between pulses"),
        ]),
    ]


class LaunchPage(QWidget):
    def __init__(self, win):
        super().__init__(win)
        self.win, self.doc = win, win.doc
        self._loading = False
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 8, 4)
        ll.addWidget(section("⚡ Pump and seed"))
        self.pump = Form(pump_groups(), cols=1)
        self.seed = Form(seed_groups(), cols=1)
        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        c1, c2 = QVBoxLayout(), QVBoxLayout()
        c1.addWidget(self.pump)
        c1.addStretch(1)
        # seed source files
        files = QWidget()
        fl = QVBoxLayout(files)
        fl.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.b_csv = QPushButton("Load pulse CSV…")
        self.b_csv.setToolTip("Columns: time (ns), power (W or normalised), optional phase (rad). Stored in the project.")
        self.b_stage = QPushButton("Choose stage file…")
        self.b_stage.setToolTip("The output of a previous stage (Results ▸ Save stage), or a Q-switched pulse")
        row.addWidget(self.b_csv)
        row.addWidget(self.b_stage)
        row.addStretch(1)
        fl.addLayout(row)
        self.src_info = muted("")
        fl.addWidget(self.src_info)
        c2.addWidget(files)
        c2.addWidget(self.seed)
        c2.addStretch(1)
        bl.addLayout(c1, 1)
        bl.addLayout(c2, 1)
        ll.addWidget(scroll(body), 1)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 4, 4, 4)
        rl.addWidget(section("Seed as launched"))
        self.err = Banner(kind="danger")
        rl.addWidget(self.err)
        self.info = Banner(kind="info")
        rl.addWidget(self.info)
        self.charts = ChartGrid(cols=1, empty="The seed pulse appears here.")
        rl.addWidget(self.charts, 1)
        split.addWidget(right)
        split.setSizes([820, 500])

        self._preview = QTimer(self)
        self._preview.setSingleShot(True)
        self._preview.setInterval(450)
        self._preview.timeout.connect(self.preview)
        self.pump.changed.connect(self._pump_edited)
        self.seed.changed.connect(self._seed_edited)
        self.b_csv.clicked.connect(self.load_csv)
        self.b_stage.clicked.connect(self.choose_stage)
        self.doc.changed.connect(self._doc_changed)
        self.reload()

    def _doc_changed(self, part: str) -> None:
        if part in ("all", "pump", "seed"):
            self.reload()
        elif part == "sim":
            self._preview.start()

    def reload(self) -> None:
        self._loading = True
        try:
            self.pump.set_values(self.doc.project.pump)
            self.seed.set_values(self.doc.project.seed)
        finally:
            self._loading = False
        self._source_info()
        self._preview.start()

    def _pump_edited(self) -> None:
        if not self._loading:
            self.doc.update("pump", self.pump.merged(self.doc.project.pump))

    def _seed_edited(self) -> None:
        if not self._loading:
            self.doc.update("seed", self.seed.merged(self.doc.project.seed))
            self._source_info()
            self._preview.start()

    def _source_info(self) -> None:
        p = self.doc.project
        src = p.seed.get("source")
        self.b_csv.setVisible(src == "csv")
        self.b_stage.setVisible(src == "stage")
        if src == "csv":
            c = p.seed_csv
            self.src_info.setText(f"Pulse file: {c.get('file') or 'loaded'} ({len(c['t_ns'])} points)" if c
                                  else "No pulse file loaded yet.")
        elif src == "stage":
            f = p.stage_file
            ok = bool(f) and os.path.exists(f)
            self.src_info.setText(f"Stage file: {f}" if ok else
                                  ("⚠ Stage file missing: " + f if f else "No stage file chosen yet."))
        else:
            self.src_info.setText("")

    def load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Seed pulse", "", "CSV / text (*.csv *.txt *.dat);;All files (*)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                data = parse_seed_csv(fh.read(), os.path.basename(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Seed pulse", f"Could not read the pulse: {e}")
            return
        self.doc.project.seed_csv = data
        self.doc.touch("seed")
        self._source_info()
        self._preview.start()

    def choose_stage(self) -> None:
        from fibamp.paths import sub
        path, _ = QFileDialog.getOpenFileName(self, "Previous stage", str(sub("stages")),
                                              "Stage files (*.npz);;All files (*)")
        if path:
            self.set_stage(path)

    def set_stage(self, path: str) -> None:
        self.doc.project.stage_file = path
        seed = dict(self.doc.project.seed, source="stage")
        self.doc.update("seed", seed)
        self.doc.touch("seed")
        self.reload()

    def preview(self) -> None:
        from fibamp.seed import build_seed, pulse_stats, spectrum
        p = self.doc.project
        try:
            cfg = SeedConfig.from_dict(p.seed)
            sim = SimConfig.from_dict(p.sim)
            csv_data, stage = p.seed_inputs()
            s = build_seed(cfg, sim, csv_data=csv_data, stage=stage)
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.err.set(f"<b>Seed:</b> {e}", "danger")
            self.info.set("")
            self.charts.show_message("No seed pulse to show.")
            return
        self.err.set("")
        st = pulse_stats(s.t, s.A)
        avg = st["E"] * s.f_rep
        lines = [f"<b>{st['E']*1e6:.4g} µJ</b>, peak <b>{st['P_peak']:.4g} W</b>, FWHM <b>{st['fwhm']*1e9:.4g} ns</b>, "
                 f"{s.f_rep/1e3:g} kHz → {avg*1e3:.4g} mW average; time grid {s.t.size} points over "
                 f"{(s.t[-1]-s.t[0])*1e9:.4g} ns"]
        lines += [f"• {n}" for n in s.notes]
        self.info.set("<br>".join(lines), "info")
        P = np.abs(s.A) ** 2
        k = np.where(P > 1e-4 * P.max())[0]
        a, b = (k[0], k[-1]) if k.size else (0, P.size - 1)
        pad = max(1, (b - a) // 5)
        sl = slice(max(0, a - pad), min(P.size, b + pad))
        step = max(1, (sl.stop - sl.start) // 4000)
        t = s.t[sl][::step] * 1e9
        f1 = go.Figure(go.Scatter(x=t, y=P[sl][::step], name="seed", line=dict(color=C_OUT)))
        f1.update_layout(**{**LAYOUT, "height": 320}, title="seed power", xaxis_title="time (ns)", yaxis_title="power (W)",
                         showlegend=False)
        lam, S = spectrum(s.t, s.A, s.lam0)
        m = S > S.max() * 1e-6
        lam, S = lam[m] * 1e9, S[m]
        step = max(1, lam.size // 4000)
        f2 = go.Figure(go.Scatter(x=lam[::step], y=10 * np.log10(S[::step] / S.max()), line=dict(color=C_OUT)))
        f2.update_layout(**{**LAYOUT, "height": 320}, title="seed spectrum (single shot)", xaxis_title="wavelength (nm)",
                         yaxis_title="dB rel. peak", showlegend=False)
        self.charts.set_figures([(f1, True), (f2, True)])
