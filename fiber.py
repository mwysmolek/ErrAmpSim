"""Fibre description: datasheet inputs -> every derived quantity the solver needs.

The user supplies what a datasheet gives (geometry, NA, absorption at one or two
wavelengths, background loss, length). Dopant concentrations are solved from the
absorption figures and the cross-section library; overlaps, mode areas, mode
counts, saturation energies and gain ceilings follow.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from . import modes as lp
from . import xsections as xs

H, C0 = xs.H, xs.C0
DB = 10.0 / np.log(10.0)               # 4.3429 dB per neper
N_A = 6.02214076e23
RHO_SILICA = 2200.0                     # kg/m^3
T_REF = 300.0                           # temperature of the shipped cross-section tables
LAM_ZERO = {"Er": 1537.7e-9, "Yb": 975.0e-9}   # McCumber zero-line wavelengths
K_SILICA, K_COATING = 1.38, 0.2         # W/m/K
M_ER, M_YB = 0.16726, 0.17304           # kg/mol


@dataclass
class AbsorptionSpec:
    """One datasheet absorption figure."""
    value_dB_per_m: float
    wavelength_nm: float
    kind: str = "cladding"               # "cladding" (pump absorption) or "core"


@dataclass
class FiberDatasheet:
    name: str = "fibre"
    dopant: str = "Er"                   # "Er" or "Er:Yb"
    er_table: str = "Er (aluminosilicate)"
    yb_table: str = "Yb (phosphosilicate)"
    d_core_um: float = 25.0
    d_clad_um: float = 250.0
    na_core: float = 0.09
    na_clad: float = 0.46
    d_doped_um: Optional[float] = None   # default: whole core doped
    clad_shape: str = "round"            # "round" or "octagonal" (flat-to-flat = d_clad)
    absorption: list = field(default_factory=lambda: [AbsorptionSpec(5.0, 976.0, "cladding")])
    yb_er_ratio: float = 15.0            # used only when absorption data cannot separate Er and Yb
    length_m: float = 2.0
    bg_loss_dB_per_km: float = 30.0      # core, at signal wavelengths
    clad_loss_dB_per_km: float = 10.0    # pump cladding
    mfd_um: Optional[float] = None       # LP01 MFD at 1550 nm, overrides the Marcuse estimate
    signal_mode: str = "auto"            # "auto", "LP01", "core-filling"
    ase_modes: Optional[float] = None    # guided modes carrying ASE; default V^2/2
    tau_er_ms: float = 10.0
    tau_yb_ms: float = 1.0
    k_tr_m3_s: float = 1.2e-21           # Yb(2F5/2) -> Er(4I11/2) transfer coefficient
    c_up_m3_s: float = 3.0e-24           # Er homogeneous upconversion
    n2_m2_W: float = 2.6e-20
    beta2_ps2_km: float = -22.0
    beta3_ps3_km: float = 0.12
    gR_m_W: float = 0.65e-13             # peak Raman gain coefficient at 1.55 um
    R_in_dB: float = -50.0               # input-end reflectance seen by backward ASE
    R_out_dB: float = -50.0              # output-end reflectance seen by forward ASE
    # --- transverse / modes
    n_rings: int = 0                     # radial rings for the inversion; 0 = auto (1, or 8 for multimode)
    bend_radius_cm: float = 0.0          # coil radius; 0 = straight (no bend loss, all modes carry ASE)
    lp01_launch: float = 0.95            # multimode: fraction of the seed launched into LP01
    max_modes: int = 12                  # multimode: signal modes carried by the pulse solver
    # --- Er physics beyond the two-level ion
    pair_fraction: float = 0.0           # fraction of Er ions in clusters/pairs (pair-induced quenching)
    esa: bool = True                     # signal-band excited-state absorption 4I13/2 -> 4I9/2
    temperature_K: float = 300.0         # fibre temperature for the cross sections (McCumber scaling)
    tau_er32_us: float = 1.0             # Er 4I11/2 lifetime (Er:Yb transfer bottleneck)
    k_bt_m3_s: float = 2e-22             # Er(4I11/2) -> Yb back-transfer coefficient
    # --- thermal estimate
    coating_diam_um: float = 0.0         # 0 = cladding + 200 um
    h_conv_W_m2K: float = 500.0          # heat transfer from the coating: ~10 still air, ~500 coiled on a
                                         # plate / in a groove, >2000 potted on a water-cooled plate

    def to_dict(self):
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "FiberDatasheet":
        d = dict(d)
        d["absorption"] = [a if isinstance(a, AbsorptionSpec) else AbsorptionSpec(**a)
                           for a in d.get("absorption", [])]
        known = FiberDatasheet.__dataclass_fields__
        return FiberDatasheet(**{k: v for k, v in d.items() if k in known})


class Fiber:
    """Derived fibre model. All quantities SI.

    Transverse picture: the doped region is split into ``nr`` equal-width rings,
    each with its own inversion. Every light channel (signal mode, ASE band, pump)
    has an overlap vector over the rings; with one ring this reduces to the usual
    scalar overlap factor Gamma.

    Er ions are split into isolated ions (fraction 1 - k) and pairs (fraction k,
    density k N / 2 pairs). A pair holds at most one excitation: a second one is
    quenched by upconversion faster than any other process (Delevaque et al. 1993).
    """

    def __init__(self, ds: FiberDatasheet):
        self.ds = ds
        self.notes: list[str] = []
        self.warnings: list[str] = []
        if ds.dopant not in ("Er", "Er:Yb"):
            raise ValueError("dopant must be 'Er' or 'Er:Yb'")
        self.has_yb = ds.dopant == "Er:Yb"
        self.xs_er = xs.library(ds.er_table)
        self.xs_yb = xs.library(ds.yb_table) if self.has_yb else None
        self.xs_esa = xs.library("ESA: Er 4I13/2") if ds.esa else None
        self.T = float(ds.temperature_K)

        self.a = ds.d_core_um * 1e-6 / 2
        self.a_d = (ds.d_doped_um or ds.d_core_um) * 1e-6 / 2
        if self.a_d > self.a * 1.0001:
            raise ValueError("doped diameter cannot exceed core diameter")
        self.A_core = np.pi * self.a ** 2
        self.A_d = np.pi * self.a_d ** 2
        dc = ds.d_clad_um * 1e-6
        self.A_clad = dc ** 2 * (2 * (np.sqrt(2) - 1) if ds.clad_shape == "octagonal" else np.pi / 4)
        self.L = ds.length_m
        self.alpha_bg = ds.bg_loss_dB_per_km / DB / 1e3
        self.alpha_clad = ds.clad_loss_dB_per_km / DB / 1e3
        self.tau_er = ds.tau_er_ms * 1e-3
        self.tau_yb = ds.tau_yb_ms * 1e-3
        self.tau_32 = ds.tau_er32_us * 1e-6
        self.k_tr = ds.k_tr_m3_s
        self.k_bt = ds.k_bt_m3_s
        self.c_up = ds.c_up_m3_s
        self.R_in = 10 ** (ds.R_in_dB / 10)
        self.R_out = 10 ** (ds.R_out_dB / 10)
        self.beta2 = ds.beta2_ps2_km * 1e-27
        self.beta3 = ds.beta3_ps3_km * 1e-39
        self.R_bend = ds.bend_radius_cm * 1e-2
        self.pair_k = float(min(max(ds.pair_fraction, 0.0), 1.0))

        v1550 = self.V(1550e-9)
        mode = ds.signal_mode
        if mode == "auto":
            mode = "LP01" if v1550 <= 8.0 else "core-filling"
            self.notes.append(f"signal mode: {mode} (auto, V = {v1550:.2f} at 1550 nm)")
        if mode not in ("LP01", "core-filling", "multimode"):
            raise ValueError("signal mode must be auto, LP01, core-filling or multimode")
        self.signal_mode = mode
        if mode == "LP01" and v1550 > 2.405:
            self.notes.append(f"V = {v1550:.2f}: fibre is multimode; LP01 operation assumes "
                              "mode control by coiling / tapered launch")
        # rings (equal width over the doped radius)
        nr = int(ds.n_rings) if ds.n_rings and ds.n_rings > 0 else (8 if mode == "multimode" else 1)
        self.nr = nr
        self.r_edges = np.linspace(0.0, self.a_d, nr + 1)
        self.A_r = np.pi * np.diff(self.r_edges ** 2)
        self._mode_cache = {}
        self._bend_factor = None
        self._solve_concentrations()
        self.N_s = (1 - self.pair_k) * self.N_er          # isolated Er ions
        self.N_p = 0.5 * self.pair_k * self.N_er          # Er pairs (each holds <= 1 excitation)
        if self.pair_k > 0:
            self.notes.append(f"{self.pair_k*100:.1f} % of Er ions in pairs: at most "
                              f"{(1 - self.pair_k/2)*100:.1f} % of the Er can be excited")
        if self.T != T_REF:
            self.notes.append(f"cross sections rescaled from {T_REF:.0f} K to {self.T:.0f} K (McCumber)")

    # ---------------------------------------------------------------- modes
    def V(self, lam):
        return 2 * np.pi * self.a * self.ds.na_core / np.asarray(lam, float)

    def w_lp01(self, lam):
        """LP01 mode-field radius (Marcuse), or scaled from a given MFD at 1550 nm."""
        lam = np.asarray(lam, float)
        if self.ds.mfd_um:
            w0 = self.ds.mfd_um * 1e-6 / 2
            ref = self.a * (0.65 + 1.619 * self.V(1550e-9) ** -1.5 + 2.879 * self.V(1550e-9) ** -6)
            return w0 * self.a * (0.65 + 1.619 * self.V(lam) ** -1.5 + 2.879 * self.V(lam) ** -6) / ref
        V = np.maximum(self.V(lam), 0.8)
        return self.a * (0.65 + 1.619 * V ** -1.5 + 2.879 * V ** -6)

    def _gauss_rings(self, lam):
        """Ring overlaps of a Gaussian LP01 profile: (n_lambda, nr)."""
        w = np.atleast_1d(self.w_lp01(lam))[:, None]
        e = np.exp(-2 * self.r_edges[None, :] ** 2 / w ** 2)
        return e[:, :-1] - e[:, 1:]

    def ring_gamma_core_light(self, lam):
        """Ring overlaps (n_lambda, nr) of core-guided light other than the signal (ASE, core pump)."""
        lam = np.atleast_1d(np.asarray(lam, float))
        if self.signal_mode == "LP01":
            return self._gauss_rings(lam)
        return np.broadcast_to(self.A_r / self.A_core, (lam.size, self.nr)).copy()

    def ring_gamma_clad(self):
        return self.A_r / self.A_clad

    def signal_modes(self, lam):
        """Signal channels at wavelength lam: names, ring overlaps (nm, nr), A_eff, bend loss, launch weights."""
        key = round(float(lam) * 1e12)
        if key in self._mode_cache:
            return self._mode_cache[key]
        lam = float(lam)
        if self.signal_mode == "LP01":
            G = self._gauss_rings(lam)
            md = self._lp01(lam)
            ab = np.array([lp.bend_loss(md, self.a, lam, self.R_bend)]) if md else np.zeros(1)
            out = dict(names=["LP01"], G=G, aeff=np.array([np.pi * float(self.w_lp01(lam)) ** 2]),
                       alpha_bend=ab, w=np.ones(1), degeneracy=np.array([2]))
        elif self.signal_mode == "core-filling":
            out = dict(names=["core-filling"], G=(self.A_r / self.A_core)[None, :],
                       aeff=np.array([self.A_core]), alpha_bend=np.zeros(1), w=np.ones(1),
                       degeneracy=np.array([2]))
        else:
            V = float(self.V(lam))
            mds = lp.solve_modes(V)
            r_max = self.ds.d_clad_um * 1e-6 / 2
            keep = []
            for md in mds:
                ab = lp.bend_loss(md, self.a, lam, self.R_bend)
                if ab < 1000.0 or md.l == 0 and md.m == 1:
                    keep.append((md, ab))
                if len(keep) >= max(1, int(self.ds.max_modes)):
                    break
            G = np.array([lp.ring_overlaps(md, self.a, self.r_edges, r_max) for md, _ in keep])
            aeffs = np.array([lp.aeff(md, self.a, r_max) for md, _ in keep])
            deg = np.array([md.degeneracy for md, _ in keep])
            w = np.zeros(len(keep))
            w[0] = 1.0
            if len(keep) > 1:
                p = min(max(self.ds.lp01_launch, 0.0), 1.0)
                w[0] = p
                w[1:] = (1 - p) * deg[1:] / deg[1:].sum()
            out = dict(names=[md.name for md, _ in keep], G=G, aeff=aeffs,
                       alpha_bend=np.array([ab for _, ab in keep]), w=w, degeneracy=deg)
        out["gamma"] = out["G"].sum(axis=1)
        self._mode_cache[key] = out
        return out

    def _lp01(self, lam):
        mds = lp.solve_modes(float(self.V(lam)), max_modes=1)
        return mds[0] if mds else None

    def gamma_core(self, lam):
        """Overlap of the signal with the doped region (launch-weighted for multimode)."""
        lam = np.asarray(lam, float)
        if self.signal_mode == "core-filling":
            return np.full(lam.shape, (self.a_d / self.a) ** 2)
        if self.signal_mode == "multimode":
            sm = self.signal_modes(1550e-9)
            return np.full(lam.shape, float(np.dot(sm["w"], sm["gamma"])))
        w = self.w_lp01(lam)
        return 1.0 - np.exp(-2 * self.a_d ** 2 / w ** 2)

    def aeff(self, lam):
        lam = np.asarray(lam, float)
        if self.signal_mode == "core-filling":
            return np.full(lam.shape, self.A_core)
        if self.signal_mode == "multimode":
            return np.full(lam.shape, float(self.signal_modes(1550e-9)["aeff"][0]))
        return np.pi * self.w_lp01(lam) ** 2

    def gamma_clad(self):
        return self.A_d / self.A_clad

    def bend_mode_factor(self):
        """Fraction of guided modes (degeneracy-weighted) that survive the coil, exp(-alpha L / 2)."""
        if self._bend_factor is None:
            if not self.R_bend:
                self._bend_factor = 1.0
            else:
                mds = lp.solve_modes(float(self.V(1550e-9)))
                g = np.array([m.degeneracy for m in mds], float)
                s = np.array([np.exp(-min(lp.bend_loss(m, self.a, 1550e-9, self.R_bend) * self.L / 2, 700))
                              for m in mds])
                self._bend_factor = float(max((g * s).sum(), 2.0) / max(g.sum(), 2.0))
        return self._bend_factor

    def n_modes(self, lam):
        """Guided modes (both polarisations) available to ASE (bend-stripped modes removed)."""
        if self.ds.ase_modes:
            return np.full(np.shape(lam), float(self.ds.ase_modes))
        V = self.V(lam)
        return np.maximum(2.0, V ** 2 / 2 * self.bend_mode_factor())

    # ------------------------------------------------------ cross sections
    def _mccumber_T(self, lam, se, species):
        if self.T == T_REF:
            return se
        dE = H * C0 * (1 / LAM_ZERO[species] - 1 / np.asarray(lam, float))
        return se * np.exp(dE / xs.KB * (1 / self.T - 1 / T_REF))

    def xs_er_at(self, lam):
        sa, se = self.xs_er(lam)
        return sa, self._mccumber_T(lam, se, "Er")

    def xs_yb_at(self, lam):
        if not self.has_yb:
            z = np.zeros(np.shape(lam))
            return z, z
        sa, se = self.xs_yb(lam)
        return sa, self._mccumber_T(lam, se, "Yb")

    def esa_at(self, lam):
        if self.xs_esa is None:
            return np.zeros(np.shape(lam))
        return self.xs_esa(lam)[0]

    # ---------------------------------------------------- concentrations
    def _solve_concentrations(self):
        specs = list(self.ds.absorption)
        if not specs:
            raise ValueError("at least one absorption figure is needed to fix the doping")
        rows, rhs = [], []
        for s in specs:
            lam = s.wavelength_nm * 1e-9
            gam = self.gamma_clad() if s.kind == "cladding" else float(self._gamma_datasheet(lam))
            saE, _ = self.xs_er(lam)
            saY = self.xs_yb(lam)[0] if self.has_yb else 0.0
            rows.append([gam * float(saE), gam * float(saY)])
            rhs.append(s.value_dB_per_m / DB)
            if float(saE) + float(saY) <= 0:
                raise ValueError(f"no absorption in the cross-section tables at {s.wavelength_nm} nm")
        Amat, b = np.array(rows), np.array(rhs)
        if not self.has_yb:
            col = Amat[:, 0]
            self.N_er = float(np.dot(col, b) / np.dot(col, col))
            self.N_yb = 0.0
            if len(specs) > 1:
                resid = np.abs(col * self.N_er - b) / b
                if resid.max() > 0.15:
                    self.warnings.append(f"absorption figures are mutually inconsistent by up to "
                                         f"{resid.max()*100:.0f} % with the Er cross-section table")
        else:
            solved = False
            if len(specs) >= 2:
                cond = np.linalg.cond(Amat)
                sol, *_ = np.linalg.lstsq(Amat, b, rcond=None)
                if cond < 1e3 and np.all(sol > 0):
                    self.N_er, self.N_yb = map(float, sol)
                    solved = True
                    self.notes.append("Er and Yb concentrations solved from two absorption figures")
            if not solved:
                r = self.ds.yb_er_ratio
                col = Amat[:, 0] + r * Amat[:, 1]
                self.N_er = float(np.dot(col, b) / np.dot(col, col))
                self.N_yb = r * self.N_er
                self.notes.append(f"Er/Yb split not determined by the absorption data; assumed "
                                  f"N_Yb/N_Er = {r:g}. Add a core absorption figure at ~1535 nm to fix it.")
        self.N_er_ppm = self.N_er * M_ER / (RHO_SILICA * N_A) * 1e6
        self.N_yb_ppm = self.N_yb * M_YB / (RHO_SILICA * N_A) * 1e6

    def _gamma_datasheet(self, lam):
        """Overlap assumed by a datasheet core-absorption figure (measured with the fundamental mode
        or a core-filling launch -- the same convention as the signal model, LP01 for multimode)."""
        if self.signal_mode == "core-filling":
            return (self.a_d / self.a) ** 2
        w = self.w_lp01(lam)
        return 1.0 - np.exp(-2 * self.a_d ** 2 / w ** 2)

    # ------------------------------------------------------------ derived
    def small_signal_coeffs(self, lam, n2_er=1.0, n2_yb=0.0, core=True):
        """Gain coefficient (1/m, excluding background loss) for core light at inversion n2.

        n2_er is the excited fraction of isolated ions and of pairs (a pair holds at most one
        excitation, so n2_er = 1 means every pair singly excited)."""
        lam = np.asarray(lam, float)
        gam = self.gamma_core(lam) if core else self.gamma_clad()
        saE, seE = self.xs_er_at(lam)
        saY, seY = self.xs_yb_at(lam)
        esa = self.esa_at(lam)
        n, x = n2_er, n2_er
        gE = (self.N_s * (seE * n - saE * (1 - n)) + self.N_p * (x * (seE - saE) - 2 * saE * (1 - x))
              - esa * (self.N_s * n + self.N_p * x))
        return gam * (gE + self.N_yb * (seY * n2_yb - saY * (1 - n2_yb)))

    def e_sat(self, lam):
        lam = np.asarray(lam, float)
        saE, seE = self.xs_er_at(lam)
        return H * C0 / lam * self.A_d / (self.gamma_core(lam) * (saE + seE))

    def transparency(self, lam):
        saE, seE = self.xs_er_at(lam)
        return saE / (saE + seE)

    def thermal_resistance(self):
        """Core temperature rise per W/m of heat deposited in the core (K m / W)."""
        rc, rcl = self.a, self.ds.d_clad_um * 1e-6 / 2
        rco = (self.ds.coating_diam_um or (self.ds.d_clad_um + 200.0)) * 1e-6 / 2
        return (1 / (4 * np.pi * K_SILICA) + np.log(rcl / rc) / (2 * np.pi * K_SILICA)
                + np.log(rco / rcl) / (2 * np.pi * K_COATING) + 1 / (2 * np.pi * rco * self.ds.h_conv_W_m2K))

    def summary(self, lam_s=1550e-9, lam_p=976e-9) -> dict:
        g_full = float(self.small_signal_coeffs(lam_s, 1.0, 1.0))
        n_tr = float(self.transparency(lam_s))
        E_store = (self.N_s * (1 - n_tr) + self.N_p * max(0.0, 1 - n_tr)) * self.A_d * self.L * H * C0 / lam_s
        L40 = 40.0 / DB / g_full if g_full > 0 else np.inf
        abs_clad = float(self.gamma_clad() * (self.N_er * self.xs_er_at(lam_p)[0]
                                              + self.N_yb * self.xs_yb_at(lam_p)[0]) * DB)
        abs_core = float(-self.small_signal_coeffs(lam_s, 0.0, 0.0) * DB)
        out = {
            "V at signal": float(self.V(lam_s)),
            "V at pump": float(self.V(lam_p)),
            "guided modes at signal (V^2/2)": float(self.V(lam_s) ** 2 / 2),
            "ASE mode count used": float(self.n_modes(lam_s)),
            "signal mode model": self.signal_mode,
            "radial rings": self.nr,
            "A_eff at signal (um^2)": float(self.aeff(lam_s)) * 1e12,
            "doped area (um^2)": self.A_d * 1e12,
            "signal overlap Gamma_s": float(self.gamma_core(lam_s)),
            "cladding pump overlap Gamma_p": self.gamma_clad(),
            "N_Er (m^-3)": self.N_er,
            "N_Er (wt-ppm, approx.)": self.N_er_ppm,
            "Er ions in pairs (%)": self.pair_k * 100,
            "N_Yb (m^-3)": self.N_yb,
            "N_Yb (wt-ppm, approx.)": self.N_yb_ppm,
            "cladding absorption at pump (dB/m)": abs_clad,
            "core absorption at signal (dB/m)": abs_core,
            "transparency inversion at signal": n_tr,
            "ESA cross section at signal (m^2)": float(self.esa_at(lam_s)),
            "max gain at full inversion (dB/m)": g_full * DB,
            "max single-pass gain, whole fibre (dB)": g_full * DB * self.L,
            "E_sat at signal (uJ)": float(self.e_sat(lam_s)) * 1e6,
            "stored energy at full inversion (mJ)": E_store * 1e3,
            "40-dB length at full inversion (m)": L40,
            "BPP if the core fills (mm.mrad)": self.a * self.ds.na_core * 1e6,
            "gamma_nl at signal (1/W/km)": 2 * np.pi * self.ds.n2_m2_W / (lam_s * float(self.aeff(lam_s))) * 1e3,
            "pump cladding etendue NA": self.ds.na_clad,
            "core temperature rise per W/m of heat (K)": float(self.thermal_resistance()),
            "cross-section temperature (K)": self.T,
        }
        if self.R_bend:
            out["fraction of modes surviving the coil"] = self.bend_mode_factor()
        if self.signal_mode == "multimode" or self.R_bend:
            sm = self.signal_modes(lam_s)
            for nm_, ab, g_, ae, w_ in zip(sm["names"], sm["alpha_bend"], sm["gamma"], sm["aeff"], sm["w"]):
                out[f"{nm_}: bend loss (dB/m) / overlap / A_eff (um^2) / launch"] = (
                    f"{ab*DB:.3g} / {g_:.3f} / {ae*1e12:.0f} / {w_:.3f}")
        return out
