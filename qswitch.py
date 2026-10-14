"""Actively Q-switched fibre oscillator designer.

Hold-off (between pulses): the z-resolved inter-pulse solver of the amplifier
(pump, spectrally resolved ASE, parasitic reflections) pumps the fibre with the
modulator closed.

Pulse: a point model of the cavity -- photon number Phi and single-pass gain g
(nepers at the lasing wavelength):

    dPhi/dt = [(2 g - eps(t)) Phi + S] / t_RT
    dg/dt   = -2 g Phi / (t_RT Phi_sat),     Phi_sat = A_d / (Gamma (sigma_a + sigma_e))

with eps(t) the round-trip loss (output coupler + internal + modulator, whose
extra loss falls linearly to zero over the switching time). The same equation
for a three-level medium drives g towards transparency, not below. A uniform
photon density depletes every fibre cell's excess inversion by the same factor,
so the point-model result maps back onto the z-resolved inversion exactly
within the model's assumption.

Validity: the point model describes the pulse envelope. When the pulse spans
fewer than ~3 cavity round trips the real output is a train of round-trip-spaced
spikes, and the tool flags it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy.integrate import solve_ivp

from .config import PumpConfig, SimConfig
from .fiber import Fiber, FiberDatasheet, DB
from .inversion import SlowModel, Measurement, State
from .seed import fwhm, phase_noise
from .xsections import H, C0


@dataclass
class QSwitchConfig:
    lasing_nm: float = 1550.0
    free_space_m: float = 0.20          # one-way free-space / component length
    n_group: float = 1.45
    R_oc: float = 0.9
    internal_loss_dB: float = 0.5       # round trip, excluding OC and the modulator's open loss
    aom_open_loss_dB: float = 0.4       # round trip, modulator open
    aom_closed_dB: float = 40.0         # round-trip extra loss when closed
    aom_rise_ns: float = 20.0           # 0 -> 100 % switching time
    rep_rate_kHz: float = 1.0
    linewidth_nm: float = 0.0           # for the exported seed field only
    spont_seed: float = 1e3             # photons coupled per round trip from spontaneous emission

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return QSwitchConfig(**{k: v for k, v in dict(d).items() if k in QSwitchConfig.__dataclass_fields__})


def degnan_F(r):
    """tau_p / tau_c and extraction efficiency from Degnan's theory."""
    r = max(float(r), 1.0001)
    if r > 25:
        eta = 1.0 - math.exp(-r)
    else:
        from scipy.optimize import brentq
        eta = 1.0 - brentq(lambda x: x - math.exp(-r * (1 - x)), 0.0, 1 - 1e-12)
    return r * eta / (r - 1 - math.log(r)), eta


class QSwitchModel:
    def __init__(self, ds: FiberDatasheet, pump: PumpConfig, q: QSwitchConfig, sim: SimConfig):
        self.fiber = Fiber(ds)
        self.pump, self.q, self.sim = pump, q, sim
        f = self.fiber
        self.lam = q.lasing_nm * 1e-9
        self.hnu = H * C0 / self.lam
        self.t_RT = 2 * (q.n_group * f.L + q.free_space_m) / C0
        self.eps_open = -math.log(q.R_oc) + (q.internal_loss_dB + q.aom_open_loss_dB) / DB
        self.eps_closed = q.aom_closed_dB / DB
        self.gam = float(f.gamma_core(self.lam))
        sa, se = (float(v) for v in f.xs_er_at(self.lam))
        self.sa, self.se = sa, se
        self.Phi_sat = f.A_d / (self.gam * (sa + se))
        self.n_tr = sa / (sa + se)
        self.slow = SlowModel(f, pump, sim, self.lam, q.rep_rate_kHz * 1e3)
        self.dz = self.slow.dz

    def _ring_g(self, st: State):
        """Per-cell, per-ring Er gain density at the lasing wavelength (isolated ions and pairs)."""
        f = self.fiber
        esa = float(f.esa_at(self.lam))
        return (f.N_s * (self.se * st.n - self.sa * (1 - st.n)) + f.N_p * (self.se * st.x - 2 * self.sa * (1 - st.x))
                - f.N_p * self.sa * st.x - esa * (f.N_s * st.n + f.N_p * st.x))

    def g_of(self, st: State):
        """Single-pass gain (nepers) of the lasing mode."""
        G = self.fiber.signal_modes(self.lam)["G"][0]
        return float((self._ring_g(st) @ G).sum() * self.dz)

    def eps(self, t):
        return self.eps_open + self.eps_closed * max(0.0, 1.0 - t / max(self.q.aom_rise_ns * 1e-9, 1e-15))

    def pulse(self, g0, n_out=20000):
        """Integrate one Q-switched pulse starting from single-pass gain g0 (nepers)."""
        tRT, S, Ps = self.t_RT, self.q.spont_seed, self.Phi_sat
        if 2 * g0 <= self.eps_open:
            return dict(ok=False, g_final=g0, t=np.zeros(2), P=np.zeros(2), E=0.0, reason="below threshold")

        def rhs(t, y):
            u, g = y
            Phi = math.exp(min(u, 200.0))
            return [((2 * g - self.eps(t)) + S / Phi) / tRT, -2 * g * Phi / (tRT * Ps)]

        r = 2 * g0 / self.eps_open
        tau_c = tRT / self.eps_open
        t_build = tau_c * math.log(max(Ps * g0 / S, 10.0)) / max(r - 1, 1e-3)
        t_end = self.q.aom_rise_ns * 1e-9 + 20 * t_build + 200 * tRT
        sol = solve_ivp(rhs, (0.0, t_end), [math.log(S), g0], method="LSODA", rtol=1e-8, atol=1e-10,
                        max_step=max(tRT / 4, 1e-12), dense_output=True)
        tt = np.linspace(0.0, sol.t[-1], n_out)
        u, g = sol.sol(tt)
        Phi = np.exp(np.minimum(u, 200.0))
        P = self.hnu * Phi * (-math.log(self.q.R_oc)) / tRT
        E = float(np.trapezoid(P, tt))
        return dict(ok=True, t=tt, P=P, g=g, g_final=float(g[-1]), E=E, fwhm=fwhm(tt, P),
                    t_peak=float(tt[np.argmax(P)]), r=r, t_build_est=t_build)

    def solve(self, progress=None):
        q, f = self.q, self.fiber
        last = {}

        def deplete(st):
            g0 = self.g_of(st)
            p = self.pulse(g0, n_out=4000)
            last["p"] = p
            if not p["ok"] or g0 <= 0:
                return st
            # uniform photon density: every cell's excess over transparency shrinks by the same factor
            frac = p["g_final"] / g0
            xtr = min(1.0, 2 * self.n_tr)
            n = np.where(st.n > self.n_tr, self.n_tr + (st.n - self.n_tr) * frac, st.n)
            x = np.where(st.x > xtr, xtr + (st.x - xtr) * frac, st.x)
            return State(np.clip(n, 0, 1), np.clip(x, 0, 1), st.y)

        st, conv = self.slow.solve_periodic(deplete, progress=progress, label="hold-off inversion")
        g0 = self.g_of(st)
        p = self.pulse(g0)
        meas = Measurement(self.slow)
        post = deplete(st)
        self.slow.evolve(post, 0.0, self.slow.T, measure=meas)
        avg = meas.averages()
        warn, notes = [], []
        # hold-off checks
        if 2 * g0 - (self.eps_open + self.eps_closed) > 0:
            warn.append(f"PRE-LASING: round-trip gain with the modulator closed is "
                        f"{(2*g0-self.eps_open-self.eps_closed)*DB:+.1f} dB. Increase modulator "
                        "extinction or reduce hold-off gain (pump, length, rep rate).")
        n_rt = p.get("fwhm", 0.0) / self.t_RT if p["ok"] else 0.0
        if p["ok"] and n_rt < 3:
            warn.append(f"pulse spans only {n_rt:.1f} cavity round trips: expect a train of spikes spaced "
                        f"by {self.t_RT*1e9:.2f} ns, not a smooth pulse. Lower the output coupling, "
                        "reduce internal loss or lengthen the cavity.")
        if p["ok"]:
            t_open = q.aom_rise_ns * 1e-9
            if t_open > 0.5 * p["t_peak"]:
                warn.append("the modulator is still opening when the pulse builds up: energy is dumped in "
                            "the modulator. Use a faster switch (tighter focus in the AOM).")
        if meas.max_RT > 0.5:
            warn.append(f"parasitic cavity from end reflections: round-trip ASE gain {meas.max_RT:.2f}")
        b = self.slow.b
        ase = b.kind == "ase"
        E_ext = (g0 - p["g_final"]) * self.Phi_sat * self.hnu if p["ok"] else 0.0
        summary = {
            "round-trip time (ns)": self.t_RT * 1e9,
            "single-pass gain at switch-open (dB)": g0 * DB,
            "round-trip loss, open (dB)": self.eps_open * DB,
            "inversion ratio r": 2 * g0 / self.eps_open,
            "pulse energy out (uJ)": p["E"] * 1e6 if p["ok"] else 0.0,
            "energy extracted from ions (uJ)": E_ext * 1e6,
            "output-coupling efficiency (%)": (p["E"] / E_ext * 100) if E_ext > 0 else 0.0,
            "pulse FWHM (ns)": p.get("fwhm", 0.0) * 1e9,
            "round trips within FWHM": n_rt,
            "peak power (kW)": float(p["P"].max()) / 1e3 if p["ok"] else 0.0,
            "build-up time to peak (ns)": p.get("t_peak", 0.0) * 1e9,
            "average output power (W)": p["E"] * q.rep_rate_kHz * 1e3 if p["ok"] else 0.0,
            "hold-off forward ASE (W)": float(avg["P_out"][ase & (b.direction > 0)].sum()),
            "hold-off backward ASE (W)": float(avg["P_out"][ase & (b.direction < 0)].sum()),
            "pump absorbed (W)": float(-avg["ion_x"][b.is_pump].sum()),
            "periods to steady state": conv.get("iterations", 0),
        }
        if not conv.get("converged", False):
            warn.append("periodic steady state not reached (the train may be period-doubling)")
        from .inversion import er_excited, yb_excited
        return dict(pulse=p, summary=summary, warnings=warn, notes=notes + list(f.notes), nE=er_excited(f, st),
                    nY=yb_excited(f, st), z=self.slow.z_cells, g0=g0)

    def design_map(self, g_per_m, L_range=(0.1, 3.0), R_range=(0.05, 0.99), n=50):
        """Degnan-theory FWHM and round-trip count over fibre length and OC reflectivity."""
        Ls = np.linspace(*L_range, n)
        Rs = np.linspace(*R_range, n)
        FW = np.full((n, n), np.nan)
        NR = np.full((n, n), np.nan)
        loss_int = (self.q.internal_loss_dB + self.q.aom_open_loss_dB) / DB
        for i, R in enumerate(Rs):
            eps = -math.log(R) + loss_int
            for j, L in enumerate(Ls):
                tRT = 2 * (self.q.n_group * L + self.q.free_space_m) / C0
                r = 2 * g_per_m * L / eps
                if r <= 1.02:
                    continue
                F, _ = degnan_F(r)
                tp = F * tRT / eps * 0.767          # E/P_peak width -> FWHM, calibrated on the ODE
                FW[i, j] = tp
                NR[i, j] = tp / tRT
        return Ls, Rs, FW, NR

    def export_seed(self, res, window_mult=8.0, nt=2 ** 16, rng_seed=1):
        """Stage dict holding the Q-switched pulse, ready to seed an amplifier."""
        p = res["pulse"]
        c = float(np.sum(p["t"] * p["P"]) / np.sum(p["P"]))
        Tw = max(window_mult * p["fwhm"], 1.2 * np.ptp(p["t"]))
        t = (np.arange(nt) - nt // 2) * Tw / nt
        P = np.interp(t, p["t"] - c, p["P"], left=0.0, right=0.0)
        A = np.sqrt(P) * phase_noise(t, self.q.linewidth_nm, self.lam, np.random.default_rng(rng_seed))
        return dict(name="Q-switched oscillator", t=t, A=A, lam0=self.lam,
                    f_rep=self.q.rep_rate_kHz * 1e3, ase_lam=None, ase_psd=None,
                    summary=res["summary"], config=dict(qswitch=self.q.to_dict()))
