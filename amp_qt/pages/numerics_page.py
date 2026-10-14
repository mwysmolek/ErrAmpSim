"""Numerics: grids, convergence and physics switches, with one-click accuracy levels."""
from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from fibamp.config import SimConfig
from fibamp.presets import NUMERICS, numerics

from ..forms import F, Form
from ..widgets import muted, scroll, section


def sim_groups():
    return [
        ("Grids", [
            F("nz", "Fibre cells", "int", lo=10, hi=5000, step=10),
            F("ase_step_nm", "ASE band width", unit="nm", lo=0.05, hi=20, dec=2, step=0.25),
            F("er_ase_nm", "Er ASE range", "pair", unit="nm", lo=1400, hi=1700, dec=0),
            F("yb_ase_nm", "Yb ASE range", "pair", unit="nm", lo=950, hi=1200, dec=0),
            F("window_mult", "Time window", unit="× FWHM", lo=2, hi=100, dec=1, tip="≥ 8 recommended"),
            F("spectral_window_nm", "Spectral window", "opt", unit="nm", lo=0.1, hi=2000, dec=2, default=20.0,
              tip="Off: automatic"),
            F("min_spectral_nm", "Minimum automatic spectral window", unit="nm", lo=0.1, hi=2000, dec=1),
            F("nt_max", "Max time points", "int", lo=256, hi=2 ** 22, step=1024),
        ]),
        ("Convergence", [
            F("tol", "Periodic steady state tolerance", "sci", default=2e-4),
            F("max_periods", "Max periods", "int", lo=5, hi=100000, step=50),
            F("cfl", "Inversion time-step factor", lo=0.001, hi=1, dec=3, step=0.05),
            F("max_steps_per_period", "Steps per period (floor)", "int", lo=10, hi=100000, step=50),
            F("consistency_iters", "Pulse / inversion passes", "int", lo=0, hi=10),
        ]),
        ("Physics", [
            F("ase", "ASE", "bool"),
            F("quantum_noise", "Quantum noise on the seed", "bool"),
            F("spont_noise", "Spontaneous emission into the signal window", "bool"),
            F("spectral_gain", "Spectral gain shaping", "bool"),
            F("dispersion", "Dispersion", "bool"),
            F("spm", "Self-phase modulation", "bool"),
            F("raman", "Raman", "bool"),
            F("sbs", "Brillouin (backward Stokes wave)", "bool"),
        ]),
        ("Nonlinear model", [
            F("nl_model", "Model", "choice", choices=[("auto", "auto"), ("band", "band (ns: SPM + Raman bands)"),
                                                      ("gnlse", "GNLSE (ps: full field)")]),
            F("gnlse_below_ns", "Auto uses GNLSE below", unit="ns", lo=0.001, hi=100, dec=3),
            F("self_steepening", "Self-steepening (GNLSE)", "bool"),
            F("max_phase_step", "SPM per sub-step", unit="rad", lo=0.001, hi=1, dec=3),
            F("max_nl_substeps", "Max nonlinear sub-steps", "int", lo=10, hi=100000, step=100),
            F("raman_orders", "Raman Stokes orders", "int", lo=1, hi=6),
            F("raman_shift_THz", "Raman shift", unit="THz", lo=1, hi=30, dec=2),
            F("raman_width_THz", "Raman width", unit="THz", lo=0.1, hi=30, dec=2),
            F("raman_noise_THz", "Raman noise bandwidth", unit="THz", lo=0.1, hi=50, dec=2),
        ]),
        ("Brillouin", [
            F("sbs_dt_ns", "SBS time step", unit="ns", lo=0.001, hi=10, dec=3),
            F("sbs_shift_GHz", "Brillouin shift", unit="GHz", lo=1, hi=30, dec=2),
            F("sbs_gB_m_W", "Brillouin gain", "sci", unit="m/W", default=5e-11),
            F("sbs_dnuB_MHz", "Brillouin linewidth", unit="MHz", lo=1, hi=1000, dec=1),
            F("sbs_tau_ns", "Phonon lifetime", unit="ns", lo=0.1, hi=100, dec=2),
            F("fibre_temperature_K", "Fibre temperature (phonons)", unit="K", lo=50, hi=600, dec=1),
        ]),
        ("Damage reference", [
            F("damage_bulk_GW_cm2_10ns", "Bulk damage at 10 ns", unit="GW/cm²", lo=0.1, hi=1000, dec=2),
            F("damage_facet_GW_cm2_10ns", "Bare-facet damage at 10 ns", unit="GW/cm²", lo=0.01, hi=1000, dec=2),
        ]),
    ]


LEVEL_TEXT = {"draft": "Draft — coarse grids, ~3-5× faster. Quick looks and batches.",
              "standard": "Standard — the engine defaults.",
              "fine": "Fine — dense grids and tight convergence for final numbers (several × slower)."}


def level_of(sim: dict) -> str:
    """Which accuracy level the grid / convergence settings match (or 'custom')."""
    keys = set().union(*[set(v) for v in NUMERICS.values()])
    for lvl in ("draft", "standard", "fine"):
        ref = numerics(lvl).to_dict()
        if all(abs(float(sim.get(k, 0)) - float(ref[k])) <= 1e-12 * max(1.0, abs(float(ref[k]))) for k in keys):
            return lvl
    return "custom"


class NumericsPage(QWidget):
    def __init__(self, win):
        super().__init__(win)
        self.win, self.doc = win, win.doc
        self._loading = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.addWidget(section("⚙ Numerics"))
        lay.addWidget(muted("The solver checks itself (photon-number ledger, periodic convergence, window edges) and says "
                            "on the Results page when a setting here needs to change."))
        row = QHBoxLayout()
        row.addWidget(QLabel("Accuracy"))
        self.buttons = {}
        for lvl in ("draft", "standard", "fine"):
            b = QPushButton(lvl.capitalize())
            b.setCheckable(True)
            b.setToolTip(LEVEL_TEXT[lvl])
            b.clicked.connect(lambda _c=False, x=lvl: self.set_level(x))
            self.buttons[lvl] = b
            row.addWidget(b)
        self.level_lab = muted("")
        row.addWidget(self.level_lab, 1)
        lay.addLayout(row)
        self.form = Form(sim_groups(), cols=3)
        lay.addWidget(scroll(self.form), 1)
        self.form.changed.connect(self._edited)
        self.doc.changed.connect(lambda part: self.reload() if part in ("all", "sim") else None)
        self.reload()

    def reload(self) -> None:
        self._loading = True
        try:
            self.form.set_values(self.doc.project.sim)
        finally:
            self._loading = False
        self._show_level()

    def _show_level(self) -> None:
        lvl = level_of(self.doc.project.sim)
        for k, b in self.buttons.items():
            b.setChecked(k == lvl)
        self.level_lab.setText(LEVEL_TEXT.get(lvl, "Custom settings."))

    def _edited(self) -> None:
        if not self._loading:
            self.doc.update("sim", self.form.merged(self.doc.project.sim))
            self._show_level()

    def set_level(self, lvl: str) -> None:
        self.doc.update("sim", numerics(lvl, SimConfig.from_dict(self.doc.project.sim)).to_dict())
        self.reload()
