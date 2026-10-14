"""Fibre: the datasheet, the fibre library, cross-section tables and what follows from them."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
                               QInputDialog, QLabel, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget)

from fibamp import io as fio
from fibamp import xsections as xs
from fibamp.fiber import Fiber, FiberDatasheet

from ..charts import ChartGrid
from ..forms import F, Form
from ..widgets import NO_WHEEL, Banner, Table, muted, scroll, section

ER_YB = lambda v: v.get("dopant") == "Er:Yb"            # noqa: E731
MM = lambda v: v.get("signal_mode") == "multimode"      # noqa: E731


def er_tables():
    return [(k, k) for k in xs.LIBRARY if not k.startswith("ESA") and "Yb" not in k]


def yb_tables():
    return [(k, k) for k in xs.LIBRARY if "Yb" in k]


def fibre_groups():
    return [
        ("Fibre", [
            F("name", "Name", "text", tip="Used in the library and in reports"),
            F("dopant", "Dopant", "choice", choices=[("Er", "Er"), ("Er:Yb", "Er:Yb co-doped")]),
            F("er_table", "Er cross sections", "choice", choices=er_tables(),
              tip="Shipped tables are representative, not vendor data: import your vendor's spectra for real work"),
            F("yb_table", "Yb cross sections", "choice", choices=yb_tables(), when=ER_YB),
            F("length_m", "Length", unit="m", lo=0.01, hi=500, dec=3, step=0.1),
        ]),
        ("Geometry", [
            F("d_core_um", "Core diameter", unit="µm", lo=1, hi=200, dec=2, step=0.5),
            F("d_clad_um", "Cladding diameter", unit="µm", lo=50, hi=1000, dec=1, step=5),
            F("clad_shape", "Cladding shape", "choice", choices=[("round", "round"), ("octagonal", "octagonal (flat to flat)")]),
            F("na_core", "Core NA", lo=0.01, hi=0.5, dec=3, step=0.005),
            F("na_clad", "Cladding NA", lo=0.05, hi=0.7, dec=3, step=0.01),
            F("d_doped_um", "Doped diameter", "opt", unit="µm", lo=1, hi=200, dec=2, default=20.0,
              tip="Off: the whole core is doped"),
            F("mfd_um", "MFD at 1550 nm", "opt", unit="µm", lo=1, hi=100, dec=2, default=10.0,
              tip="Off: Marcuse estimate from core size and NA"),
        ]),
        ("Losses and reflections", [
            F("bg_loss_dB_per_km", "Core background loss", unit="dB/km", lo=0, hi=10000, dec=1, step=5),
            F("clad_loss_dB_per_km", "Pump cladding loss", unit="dB/km", lo=0, hi=10000, dec=1, step=5),
            F("R_in_dB", "Input-end reflectance", unit="dB", lo=-120, hi=0, dec=1, step=1,
              tip="Seen by the backward ASE; with the output reflectance it can close a parasitic cavity"),
            F("R_out_dB", "Output-end reflectance", unit="dB", lo=-120, hi=0, dec=1, step=1),
            F("bend_radius_cm", "Coil radius", unit="cm", lo=0, hi=100, dec=2, step=0.5,
              tip="0 = straight: no bend loss, every guided mode carries ASE"),
        ]),
        ("Modes", [
            F("signal_mode", "Signal mode model", "choice",
              choices=[("auto", "auto"), ("LP01", "LP01 (Gaussian)"), ("core-filling", "core-filling"),
                       ("multimode", "multimode (LP modes)")]),
            F("n_rings", "Radial rings", "int", lo=0, hi=64, tip="0 = auto (1, or 8 for multimode)"),
            F("ase_modes", "Modes carrying ASE", "opt", lo=1, hi=10000, dec=1, default=2.0,
              tip="Off: V²/2 (minus the modes the coil strips)"),
            F("lp01_launch", "Seed fraction in LP01", lo=0, hi=1, dec=3, step=0.01, when=MM),
            F("max_modes", "Signal modes carried", "int", lo=1, hi=60, when=MM),
        ]),
        ("Er:Yb", [
            F("yb_er_ratio", "N_Yb / N_Er", lo=0.1, hi=100, dec=2, when=ER_YB,
              tip="Used only when the absorption figures cannot separate Er and Yb"),
            F("tau_yb_ms", "Yb lifetime", unit="ms", lo=0.01, hi=10, dec=3, when=ER_YB),
            F("k_tr_m3_s", "Yb→Er transfer", "sci", unit="m³/s", default=1.2e-21, when=ER_YB),
            F("k_bt_m3_s", "Er→Yb back-transfer", "sci", unit="m³/s", default=2e-22, when=ER_YB),
            F("tau_er32_us", "Er ⁴I₁₁/₂ lifetime", unit="µs", lo=0.001, hi=100, dec=3, when=ER_YB),
        ]),
        ("Er physics", [
            F("tau_er_ms", "Er lifetime", unit="ms", lo=0.1, hi=20, dec=2),
            F("c_up_m3_s", "Upconversion coefficient", "sci", unit="m³/s", default=3e-24),
            F("pair_fraction", "Er ions in pairs", lo=0, hi=1, dec=3, step=0.01,
              tip="Clustered ions (pair-induced quenching); heavily doped fibres often have 5-20 %"),
            F("esa", "Signal-band ESA", "bool", tip="4I13/2 → 4I9/2 excited-state absorption (matters in the L band)"),
            F("temperature_K", "Fibre temperature", unit="K", lo=150, hi=500, dec=1,
              tip="Cross sections are rescaled from 300 K by McCumber"),
        ]),
        ("Nonlinear and dispersion", [
            F("n2_m2_W", "n₂", "sci", unit="m²/W", default=2.6e-20),
            F("gR_m_W", "Raman gain peak", "sci", unit="m/W", default=0.65e-13),
            F("beta2_ps2_km", "β₂", unit="ps²/km", lo=-200, hi=200, dec=2),
            F("beta3_ps3_km", "β₃", unit="ps³/km", lo=-10, hi=10, dec=3),
        ]),
        ("Thermal", [
            F("coating_diam_um", "Coating diameter", unit="µm", lo=0, hi=2000, dec=0, step=10,
              tip="0 = cladding + 200 µm"),
            F("h_conv_W_m2K", "Heat transfer from coating", unit="W/m²K", lo=1, hi=100000, dec=0, step=50,
              tip="~10 still air, ~500 coiled on a plate / in a groove, >2000 potted on a water-cooled plate"),
        ]),
    ]


class AbsorptionEditor(QGroupBox):
    """The datasheet absorption figures (1 to 3) that fix the doping."""
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Datasheet absorption (fixes the doping)", parent)
        self.lay = QVBoxLayout(self)
        self.grid = QGridLayout()
        self.lay.addLayout(self.grid)
        self.add = QPushButton("+ Add a figure")
        self.add.setToolTip("Er:Yb: a cladding figure at ~976 nm plus a core figure at ~1535 nm separate Er and Yb")
        self.add.clicked.connect(lambda: (self._add_row(dict(value_dB_per_m=10.0, wavelength_nm=1530.0, kind="core")),
                                          self.changed.emit()))
        self.lay.addWidget(self.add)
        self.rows: list[tuple] = []

    def _add_row(self, a: dict) -> None:
        r = len(self.rows)
        v = QDoubleSpinBox()
        v.setRange(0.001, 10000)
        v.setDecimals(3)
        v.setSuffix(" dB/m")
        v.setValue(float(a.get("value_dB_per_m", 1.0)))
        w = QDoubleSpinBox()
        w.setRange(800, 1700)
        w.setDecimals(1)
        w.setSuffix(" nm")
        w.setValue(float(a.get("wavelength_nm", 976.0)))
        k = QComboBox()
        k.addItem("cladding (pump)", "cladding")
        k.addItem("core", "core")
        k.setCurrentIndex(max(0, k.findData(a.get("kind", "cladding"))))
        rm = QPushButton("✕")
        rm.setFixedWidth(30)
        rm.setToolTip("Remove this figure")
        for x in (v, w, k):
            x.installEventFilter(NO_WHEEL)
            x.setFocusPolicy(Qt.StrongFocus)
        v.setKeyboardTracking(False)
        w.setKeyboardTracking(False)
        v.valueChanged.connect(lambda _x: self.changed.emit())
        w.valueChanged.connect(lambda _x: self.changed.emit())
        k.currentIndexChanged.connect(lambda _x: self.changed.emit())
        row = (v, w, k, rm)
        rm.clicked.connect(lambda: self._remove(row))
        for c, x in enumerate((v, QLabel("at"), w, k, rm)):
            self.grid.addWidget(x, r, c)
        self.rows.append(row)
        self._enable()

    def _remove(self, row) -> None:
        vals = self.values()
        i = self.rows.index(row)
        del vals[i]
        self.set_values(vals)
        self.changed.emit()

    def _enable(self) -> None:
        for r in self.rows:
            r[3].setEnabled(len(self.rows) > 1)
        self.add.setEnabled(len(self.rows) < 3)

    def values(self) -> list[dict]:
        return [dict(value_dB_per_m=v.value(), wavelength_nm=w.value(), kind=k.currentData()) for v, w, k, _ in self.rows]

    def set_values(self, absorption: list) -> None:
        while self.grid.count():
            it = self.grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.rows = []
        for a in absorption or [dict(value_dB_per_m=5.0, wavelength_nm=976.0, kind="cladding")]:
            self.blockSignals(True)
            self._add_row(a if isinstance(a, dict) else a.__dict__)
            self.blockSignals(False)
        self._enable()


class FibrePage(QWidget):
    def __init__(self, win):
        super().__init__(win)
        self.win, self.doc = win, win.doc
        self._loading = False
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)

        # ---- left: library + datasheet
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 8, 4)
        ll.addWidget(section("🧵 Fibre datasheet"))
        ll.addWidget(muted("Enter what the datasheet gives; the doping, overlaps, mode areas and saturation follow. "
                           "Hover a field for help."))
        lib = QHBoxLayout()
        self.lib = QComboBox()
        self.lib.setToolTip("Fibres saved in the library (the data folder next to the app)")
        self.lib.setMinimumWidth(200)
        b_load = QPushButton("Load")
        b_load.setToolTip("Replace the datasheet with the selected library fibre (the length is kept)")
        b_save = QPushButton("Save to library")
        b_del = QPushButton("Delete")
        b_del.setProperty("danger", True)
        b_xs = QPushButton("Import cross sections…")
        b_xs.setToolTip("A CSV of wavelength, σa, σe (your vendor's spectra); it is stored inside the project")
        for b in (b_load, b_save, b_del):
            b.setFixedHeight(28)
        lib.addWidget(QLabel("Library"))
        lib.addWidget(self.lib, 1)
        lib.addWidget(b_load)
        lib.addWidget(b_save)
        lib.addWidget(b_del)
        ll.addLayout(lib)
        ll.addWidget(b_xs, 0, Qt.AlignLeft)
        self.absorb = AbsorptionEditor()
        self.form = Form(fibre_groups(), cols=2)
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.absorb)
        bl.addWidget(self.form)
        bl.addStretch(1)
        ll.addWidget(scroll(body), 1)
        split.addWidget(left)

        # ---- right: derived quantities
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 4, 4, 4)
        rl.addWidget(section("What follows from it"))
        self.err = Banner(kind="danger")
        rl.addWidget(self.err)
        self.msgs = Banner(kind="info")
        rl.addWidget(self.msgs)
        rs = QSplitter(Qt.Vertical)
        self.table = Table()
        rs.addWidget(self.table)
        self.charts = ChartGrid(cols=1, empty="The cross sections and gain spectrum appear here.")
        rs.addWidget(self.charts)
        rs.setSizes([300, 500])
        rl.addWidget(rs, 1)
        split.addWidget(right)
        split.setSizes([760, 560])

        self._derive = QTimer(self)
        self._derive.setSingleShot(True)
        self._derive.setInterval(400)
        self._derive.timeout.connect(self.derive)
        self.form.changed.connect(self._edited)
        self.absorb.changed.connect(self._edited)
        b_load.clicked.connect(self.load_library)
        b_save.clicked.connect(self.save_library)
        b_del.clicked.connect(self.delete_library)
        b_xs.clicked.connect(self.import_xs)
        self.doc.changed.connect(self._doc_changed)
        self.reload()

    # ------------------------------------------------------------------ data
    def _doc_changed(self, part: str) -> None:
        if part in ("all", "fiber", "xs"):
            self.reload()
        elif part in ("seed", "pump"):
            self._derive.start()

    def reload(self) -> None:
        self._loading = True
        try:
            self.form.set_choices("er_table", er_tables())
            self.form.set_choices("yb_table", yb_tables())
            self.form.set_values(self.doc.project.fiber)
            self.absorb.set_values(self.doc.project.fiber.get("absorption", []))
        finally:
            self._loading = False
        self.reload_library()
        self._derive.start()

    def reload_library(self) -> None:
        cur = self.lib.currentText()
        self.lib.clear()
        self.lib.addItems(fio.list_fibres())
        i = self.lib.findText(cur)
        if i >= 0:
            self.lib.setCurrentIndex(i)

    def _edited(self) -> None:
        if self._loading:
            return
        d = self.form.merged(self.doc.project.fiber)
        d["absorption"] = self.absorb.values()
        self.doc.update("fiber", d)
        self._derive.start()

    def derive(self) -> None:
        p = self.doc.project
        try:
            p.register_xs()
            fib = Fiber(FiberDatasheet.from_dict(p.fiber))
            lam_s = float(p.seed.get("wavelength_nm", 1550.0)) * 1e-9
            lam_p = float(p.pump.get("co", {}).get("wavelength_nm", 976.0)) * 1e-9
            summ = fib.summary(lam_s, lam_p)
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.err.set(f"<b>This datasheet does not make a fibre:</b> {e}", "danger")
            self.table.set_rows([])
            self.charts.show_message("Fix the datasheet to see the fibre.")
            return
        self.err.set("")
        msgs = [f"⚠ {w}" for w in fib.warnings] + [f"• {n}" for n in fib.notes]
        n_er = summ.get("N_Er (m^-3)", 0)
        if n_er > 1.2e26:
            msgs.insert(0, f"⚠ N_Er = {n_er:.2g} m⁻³ is far above any real fibre (~0.5-8e25): check the absorption "
                           "figure and its kind (cladding vs core)")
        self.msgs.set("<br>".join(msgs), "notice" if any(m.startswith("⚠") for m in msgs) else "info")
        self.table.set_kv(summ)
        from fibamp.plots import gain_spectrum_figure, xs_figure
        f1 = xs_figure(fib)
        f1.update_layout(title="cross sections", height=330)
        f2 = gain_spectrum_figure(fib)
        f2.update_layout(title=f"small-signal gain of {fib.L:g} m at fixed inversion", height=360)
        self.charts.set_figures([(f1, True), (f2, True)])

    # --------------------------------------------------------------- library
    def load_library(self) -> None:
        name = self.lib.currentText()
        if not name:
            return
        try:
            ds = fio.load_fibre(name).to_dict()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Library", f"Could not load '{name}': {e}")
            return
        ds["length_m"] = self.doc.project.fiber.get("length_m", ds.get("length_m"))
        self.doc.update("fiber", ds)
        self.win.status(f"Loaded '{name}' from the library (length kept at {ds['length_m']:g} m)")

    def save_library(self) -> None:
        name, ok = QInputDialog.getText(self, "Save to library", "Fibre name:",
                                        text=self.doc.project.fiber.get("name", "fibre"))
        if not ok or not name.strip():
            return
        d = dict(self.doc.project.fiber, name=name.strip())
        self.doc.update("fiber", d)
        path = fio.save_fibre(FiberDatasheet.from_dict(d))
        self.reload_library()
        self.lib.setCurrentText(name.strip())
        self.win.status(f"Saved to the fibre library: {path}")

    def delete_library(self) -> None:
        name = self.lib.currentText()
        if name and QMessageBox.question(self, "Library", f"Delete '{name}' from the fibre library?") == QMessageBox.Yes:
            fio.delete_fibre(name)
            self.reload_library()

    def import_xs(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Cross-section table", "",
                                              "CSV / text (*.csv *.txt *.dat);;All files (*)")
        if not path:
            return
        unit, ok = QInputDialog.getItem(self, "Units", "Cross sections in the file are in:", list(xs.UNIT_SCALE), 0, False)
        if not ok:
            return
        import os
        name, ok = QInputDialog.getText(self, "Table name", "Name for this table:",
                                        text=os.path.splitext(os.path.basename(path))[0])
        if not ok or not name.strip():
            return
        role, ok = QInputDialog.getItem(self, "Use as", "Use the table for:", ["Er", "Yb"], 0, False)
        if not ok:
            return
        try:
            t = xs.load_table(path, name.strip(), unit)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Cross sections", f"Could not read the table: {e}")
            return
        lo, hi = t.range_nm
        if role == "Yb" and "Yb" not in t.name:
            t.name = f"Yb: {t.name}"
        self.doc.project.add_xs_table(t)
        d = dict(self.doc.project.fiber)
        d["yb_table" if role == "Yb" else "er_table"] = t.name
        if role == "Yb":
            d["dopant"] = "Er:Yb"
        self.doc.touch("xs")
        self.doc.update("fiber", d)
        self.win.status(f"Imported '{t.name}' ({len(t.lam)} rows, {lo:.0f}-{hi:.0f} nm); stored in the project")
