"""Q-switched oscillator designer: the project's fibre and pump in an actively Q-switched cavity."""
from __future__ import annotations

import html
import time

import plotly.graph_objects as go
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSplitter, QVBoxLayout, QWidget

from fibamp import io as fio
from fibamp import plots
from fibamp.paths import sub
from fibamp.project import slug

from ..charts import ChartGrid
from ..forms import F, Form
from ..widgets import Banner, KpiRow, Table, fmt_energy_uJ, fmt_power_kW, legend_below, muted, scroll, section


def q_groups():
    return [
        ("Cavity", [
            F("lasing_nm", "Lasing wavelength", unit="nm", lo=1450, hi=1650, dec=2),
            F("free_space_m", "Free-space / component length (one way)", unit="m", lo=0, hi=100, dec=3, step=0.05),
            F("n_group", "Group index", lo=1, hi=2, dec=3),
            F("R_oc", "Output coupler reflectivity", lo=0.001, hi=0.999, dec=3, step=0.01),
            F("internal_loss_dB", "Internal loss (round trip)", unit="dB", lo=0, hi=30, dec=2, step=0.1),
        ]),
        ("Modulator (AOM)", [
            F("aom_open_loss_dB", "Insertion loss, open (round trip)", unit="dB", lo=0, hi=20, dec=2, step=0.1),
            F("aom_closed_dB", "Extra loss closed (round trip)", unit="dB", lo=0, hi=200, dec=1, step=1),
            F("aom_rise_ns", "Switching time", unit="ns", lo=0, hi=10000, dec=1, step=5),
            F("rep_rate_kHz", "Repetition rate", unit="kHz", lo=0.001, hi=1000, dec=3, step=1),
        ]),
        ("Seed export", [
            F("linewidth_nm", "Linewidth given to the exported pulse", unit="nm", lo=0, hi=10, dec=4),
            F("spont_seed", "Spontaneous photons per round trip", "sci", default=1e3),
        ]),
    ]


KPIS = [("Pulse energy", ""), ("Peak power", ""), ("FWHM", ""), ("Round trips in FWHM", "Below 3: a spiky train"),
        ("Build-up time", "Switch opening to pulse peak")]


class QSwitchPage(QWidget):
    def __init__(self, win):
        super().__init__(win)
        self.win, self.doc = win, win.doc
        self._loading = False
        self.res = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        split = QSplitter(Qt.Horizontal)
        lay.addWidget(split)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(section("🔦 Q-switched oscillator"))
        ll.addWidget(muted("The fibre (Fibre page) and pump (Pump & seed page) of this project in an actively "
                           "Q-switched cavity: the z-resolved hold-off inversion between pulses, then a point model of "
                           "the pulse. The pulse can seed an amplifier stage."))
        self.form = Form(q_groups(), cols=1)
        ll.addWidget(scroll(self.form), 1)
        row = QHBoxLayout()
        self.b_run = QPushButton("▶ Design the pulse")
        self.b_run.setProperty("primary", True)
        self.b_use = QPushButton("Use as amplifier seed…")
        self.b_use.setToolTip("Save the pulse as a stage file and seed a new amplifier project with it")
        self.b_use.setEnabled(False)
        row.addWidget(self.b_run)
        row.addWidget(self.b_use)
        ll.addLayout(row)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.kpis = KpiRow(KPIS)
        rl.addWidget(self.kpis)
        self.danger = Banner(kind="danger")
        rl.addWidget(self.danger)
        self.info = Banner(kind="info")
        rl.addWidget(self.info)
        rs = QSplitter(Qt.Vertical)
        self.charts = ChartGrid(cols=2, empty="Press <b>▶ Design the pulse</b>.")
        rs.addWidget(self.charts)
        self.table = Table()
        rs.addWidget(self.table)
        rs.setSizes([600, 220])
        rl.addWidget(rs, 1)
        split.addWidget(right)
        split.setSizes([470, 900])

        self.form.changed.connect(self._edited)
        self.b_run.clicked.connect(win.run_qswitch)
        self.b_use.clicked.connect(self.use_as_seed)
        self.doc.changed.connect(lambda part: self.reload() if part in ("all", "qswitch") else None)
        self.reload()

    def reload(self) -> None:
        self._loading = True
        try:
            self.form.set_values(self.doc.project.qswitch)
        finally:
            self._loading = False

    def _edited(self) -> None:
        if not self._loading:
            self.doc.update("qswitch", self.form.merged(self.doc.project.qswitch))

    def show_result(self, res) -> None:
        self.res = res
        s, p = res["summary"], res["pulse"]
        self.kpis.set("Pulse energy", fmt_energy_uJ(s["pulse energy out (uJ)"]),
                      f"extracted {fmt_energy_uJ(s['energy extracted from ions (uJ)'])}")
        self.kpis.set("Peak power", fmt_power_kW(s["peak power (kW)"]), f"{s['average output power (W)']:.3g} W average")
        self.kpis.set("FWHM", f"{s['pulse FWHM (ns)']:.4g} ns", f"round trip {s['round-trip time (ns)']:.3g} ns")
        n = s["round trips within FWHM"]
        self.kpis.set("Round trips in FWHM", f"{n:.1f}", "smooth pulse" if n >= 3 else "spiky train",
                      color=None if n >= 3 else "#f59e0b")
        self.kpis.set("Build-up time", f"{s['build-up time to peak (ns)']:.4g} ns",
                      f"inversion ratio r = {s['inversion ratio r']:.2f}")
        self.danger.set("<br>".join(f"⚠ {html.escape(w)}" for w in res["warnings"]), "danger")
        self.info.set("<br>".join(f"• {html.escape(n)}" for n in res["notes"]), "info")
        self.table.set_kv(s)
        f1 = plots.qswitch_pulse_figure(p)
        if p.get("ok") and p.get("fwhm"):
            f1.update_xaxes(range=[max(0.0, (p["t_peak"] - 4 * p["fwhm"]) * 1e9), (p["t_peak"] + 6 * p["fwhm"]) * 1e9])
        f1.update_layout(title="Q-switched pulse")
        Ls, Rs, FW, NR = res["map"]
        f2 = plots.qswitch_map_figure(Ls, Rs, FW, NR, mark=(res["L"], self.doc.project.qswitch["R_oc"]))
        f2.update_layout(title="FWHM map (Degnan); right of / below the red line: spikes")
        figs = [(f1, False), (legend_below(f2, 460), False)]
        z, nE = res["z"], res["nE"]
        f3 = go.Figure(go.Scatter(x=z, y=nE, name="Er", line=dict(color=plots.C_OUT)))
        if res.get("nY") is not None and len(res["nY"]) and max(res["nY"]) > 0:
            f3.add_trace(go.Scatter(x=z, y=res["nY"], name="Yb", line=dict(color=plots.C_AB)))
        f3.update_layout(**plots.LAYOUT, title="inversion when the switch opens", xaxis_title="z (m)",
                         yaxis_title="excited fraction")
        figs.append((f3, True))
        self.charts.set_figures(figs)
        self.b_use.setEnabled(res.get("stage") is not None)

    def use_as_seed(self) -> None:
        if not self.res or self.res.get("stage") is None:
            return
        path = str(sub("stages") / f"{slug(self.doc.project.name)}-qswitch-{time.strftime('%Y%m%d-%H%M%S')}.fibamp.npz")
        fio.save_stage(self.res["stage"], path)
        self.win.new_stage_project(path, suffix=" – amplifier after Q-switch")
