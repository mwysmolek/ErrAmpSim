"""Inter-pulse (slow-axis) solver: pump + spectrally resolved ASE + rate equations.

Between signal pulses the populations evolve under the pump, forward and
backward ASE in spectral bands (with end-face reflections that can close a
parasitic cavity), spontaneous decay, Er upconversion and Yb->Er transfer.
Light transit (~ns) is instantaneous on the population timescale, so each time
step propagates every band through a frozen population profile exactly (the
propagation is linear once populations are fixed) and then advances the
populations with an exact exponential update at frozen photon fluxes.

Populations live on a (z cell, radial ring) grid and come in three species
(``State``):

* ``n``: excited fraction of isolated Er ions (density N_s = (1 - k) N_Er);
* ``x``: fraction of Er pairs holding one excitation (density N_p = k N_Er / 2).
  A pair absorbs from its ground state with 2 sigma_a; a singly excited pair can
  emit (sigma_e) or absorb (sigma_a), and the doubly excited pair is quenched by
  upconversion at once, so that absorption is a pure loss (pair-induced quenching);
* ``y``: excited fraction of Yb (Er:Yb only).

Signal-band excited-state absorption (4I13/2 -> 4I9/2, returning non-radiatively)
is a population-neutral loss proportional to the excited Er density. Yb->Er
transfer reaches Er 4I11/2; a fraction tau_32 k_bt N_Yb (1 - y) / (1 + ...) is
back-transferred to Yb before relaxing to 4I13/2 (adiabatic elimination of 4I11/2).

The signal pulse is applied as an instantaneous depletion at t = 0 of every
repetition period; the periodic steady state of that cycle is found by
iterating periods. At high repetition rates several periods are lumped into one
macro-step with their depletion summed, then the last periods are resolved
individually.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import PumpConfig, SimConfig
from .fiber import Fiber, DB
from .xsections import H, C0

LAM_ER_LEVEL = 1535e-9     # photon-energy equivalent of Er 4I13/2, for power budgets
LAM_YB_LEVEL = 975e-9


@dataclass
class State:
    """Populations on the (nz, nr) grid: isolated Er, Er pairs, Yb."""
    n: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @staticmethod
    def zeros(nz, nr):
        return State(np.zeros((nz, nr)), np.zeros((nz, nr)), np.zeros((nz, nr)))

    def copy(self):
        return State(self.n.copy(), self.x.copy(), self.y.copy())

    def clip(self):
        return State(np.clip(self.n, 0, 1), np.clip(self.x, 0, 1), np.clip(self.y, 0, 1))

    def lerp_from(self, other: "State", M: float) -> "State":
        """self - M (self - other): M-fold extrapolation of a per-period change."""
        return State(self.n - M * (self.n - other.n), self.x - M * (self.x - other.x),
                     self.y - M * (self.y - other.y)).clip()

    def max_diff(self, other: "State") -> float:
        return float(max(np.max(np.abs(self.n - other.n)), np.max(np.abs(self.x - other.x))))


def er_excited(fiber: Fiber, st: State):
    """Excited fraction of all Er ions per cell (area-weighted over rings)."""
    A = fiber.A_r
    return (fiber.N_s * (st.n @ A) + fiber.N_p * (st.x @ A)) / (fiber.N_er * fiber.A_d)


def yb_excited(fiber: Fiber, st: State):
    return st.y @ fiber.A_r / fiber.A_d


def _f1(x):
    """(e^x - 1) / x, branch-free."""
    small = np.abs(x) <= 1e-6
    xs = np.where(small, 1.0, x)
    return np.where(small, 1 + x / 2 + x * x / 6, np.expm1(xs) / xs)


def _f2(x):
    """((e^x - 1) / x - 1) / x, branch-free."""
    small = np.abs(x) <= 1e-4
    xs = np.where(small, 1.0, x)
    return np.where(small, 0.5 + x / 6 + x * x / 24, (np.expm1(xs) / xs - 1) / xs)


def sweep(g, s, dz):
    """Exact propagation of dP/dz = g P + s through cells in propagation order.

    g, s: (nz, m). Returns nodes of the particular solution (P0 = 0), nodes of the
    homogeneous solution (P0 = 1), and the corresponding cell means.
    """
    x = g * dz
    F1 = _f1(x)
    q = s * dz * F1
    nz, m = g.shape
    C = np.vstack([np.zeros((1, m)), np.cumsum(x, axis=0)])
    if m == 0:
        z = np.zeros((nz + 1, 0))
        return z, z.copy(), np.zeros((nz, 0)), np.zeros((nz, 0))
    if np.abs(C).max() < 600:
        eC = np.exp(C)
        part = eC * np.vstack([np.zeros((1, m)), np.cumsum(q * np.exp(-C[1:]), axis=0)])
    else:
        e = np.exp(np.clip(x, -700, 700))
        part = np.zeros((nz + 1, m))
        eC = np.ones((nz + 1, m))
        for k in range(nz):
            part[k + 1] = part[k] * e[k] + q[k]
            eC[k + 1] = eC[k] * e[k]
    mean_hom = eC[:-1] * F1
    mean_part = part[:-1] * F1 + s * dz * _f2(x)
    return part, eC, mean_part, mean_hom


@dataclass
class Bands:
    lam: np.ndarray
    direction: np.ndarray        # +1 forward (enters at z=0), -1 backward (enters at z=L)
    kind: np.ndarray             # "ase", "pump"
    G: np.ndarray                # ring overlaps (nb, nr)
    saE: np.ndarray
    seE: np.ndarray
    esa: np.ndarray
    saY: np.ndarray
    seY: np.ndarray
    alpha: np.ndarray
    spont: np.ndarray            # M h nu dnu  (W per unit of Gamma*sigma_e*N2 per metre)
    P_in: np.ndarray             # power injected at the entry end (pump: on-level)
    is_pump: np.ndarray
    dlam: np.ndarray
    pair_f: np.ndarray = field(default_factory=lambda: np.zeros(0, int))   # forward ASE band indices
    pair_b: np.ndarray = field(default_factory=lambda: np.zeros(0, int))   # matching backward indices

    @property
    def gamma(self):
        return self.G.sum(axis=1)

    @property
    def nu(self):
        return C0 / self.lam

    def __len__(self):
        return self.lam.size


def build_bands(fiber: Fiber, pump: PumpConfig, sim: SimConfig, lam_s, input_spec=None,
                cw_lines=()) -> tuple[Bands, list]:
    notes = []
    lam_list, dirs, kinds, dl, Pin, pump_flag = [], [], [], [], [], []
    # ASE grid
    ranges = [sim.er_ase_nm] + ([sim.yb_ase_nm] if fiber.has_yb else [])
    ase_lams = []
    for lo, hi in ranges:
        n = max(2, int(round((hi - lo) / sim.ase_step_nm)))
        edges = np.linspace(lo, hi, n + 1) * 1e-9
        ase_lams.append((0.5 * (edges[1:] + edges[:-1]), np.diff(edges)))
    lamA = np.concatenate([a for a, _ in ase_lams])
    dlA = np.concatenate([d for _, d in ase_lams])
    PinA = np.zeros(lamA.size)
    if input_spec is not None:
        il, ip = input_spec
        tot = np.trapezoid(ip, il) if il.size > 1 else 0.0
        captured = 0.0
        for i, (lc, d) in enumerate(zip(lamA, dlA)):
            xx = np.linspace(lc - d / 2, lc + d / 2, 9)
            PinA[i] = np.trapezoid(np.interp(xx, il, ip, left=0, right=0), xx)
            captured += PinA[i]
        if tot > 0 and captured < 0.99 * tot:
            notes.append(f"{(1-captured/tot)*100:.1f} % of the imported input spectrum lies outside the "
                         "ASE band grid and is not propagated")
    for lc, P in cw_lines:
        i = int(np.argmin(np.abs(lamA - lc)))
        PinA[i] += P
    if not sim.ase:
        keep = PinA > 0
        lamA, dlA, PinA = lamA[keep], dlA[keep], PinA[keep]
    nA = lamA.size
    GA = fiber.ring_gamma_core_light(lamA) if nA else np.zeros((0, fiber.nr))
    Gs = []
    # forward then backward ASE
    for d in (+1, -1):
        lam_list.append(lamA); dirs.append(np.full(nA, d)); kinds += ["ase"] * nA
        dl.append(dlA); Pin.append(PinA if d > 0 else np.zeros(nA)); pump_flag.append(np.zeros(nA, bool))
        Gs.append(GA)
    # pump sub-bands
    for src, d in ((pump.co, +1), (pump.counter, -1)):
        if src.power_W <= 0:
            continue
        k = max(1, int(src.n_sub))
        if k == 1 or src.fwhm_nm <= 0:
            x, w = np.array([0.0]), np.array([1.0])
        else:
            x = np.linspace(-1.5, 1.5, k)
            w = np.exp(-4 * np.log(2) * x ** 2)
            w /= w.sum()
        lp = (src.wavelength_nm + x * src.fwhm_nm) * 1e-9
        lam_list.append(lp); dirs.append(np.full(lp.size, d)); kinds += ["pump"] * lp.size
        dl.append(np.zeros(lp.size)); Pin.append(w * src.power_W); pump_flag.append(np.ones(lp.size, bool))
        if pump.geometry == "cladding":
            Gs.append(np.broadcast_to(fiber.ring_gamma_clad(), (lp.size, fiber.nr)).copy())
        else:
            Gs.append(fiber.ring_gamma_core_light(lp))
    lam = np.concatenate(lam_list)
    direction = np.concatenate(dirs)
    is_pump = np.concatenate(pump_flag)
    G = np.vstack(Gs)
    saE, seE = fiber.xs_er_at(lam)
    saY, seY = fiber.xs_yb_at(lam)
    alpha = np.full(lam.size, fiber.alpha_bg)
    if pump.geometry == "cladding":
        alpha = np.where(is_pump, fiber.alpha_clad, alpha)
    dlam = np.concatenate(dl)
    dnu = C0 * dlam / lam ** 2
    spont = np.where(is_pump, 0.0, fiber.n_modes(lam) * H * C0 / lam * dnu) if sim.ase else np.zeros(lam.size)
    b = Bands(lam=lam, direction=direction, kind=np.array(kinds), G=G, saE=saE, seE=seE,
              esa=fiber.esa_at(lam), saY=saY, seY=seY, alpha=alpha, spont=spont,
              P_in=np.concatenate(Pin), is_pump=is_pump, dlam=dlam)
    b.pair_f = np.arange(nA)
    b.pair_b = np.arange(nA, 2 * nA)
    return b, notes


class SlowModel:
    RT_CLAMP = 0.95
    IMPLICIT_ITERS = 8
    IMPLICIT_TOL = 1e-4

    def __init__(self, fiber: Fiber, pump: PumpConfig, sim: SimConfig, lam_s, f_rep,
                 input_spec=None, cw_lines=()):
        self.f = fiber
        self.pump = pump
        self.sim = sim
        self.nz = sim.nz
        self.nr = fiber.nr
        self.dz = fiber.L / sim.nz
        self.z_nodes = np.linspace(0, fiber.L, sim.nz + 1)
        self.z_cells = 0.5 * (self.z_nodes[1:] + self.z_nodes[:-1])
        self.T = 1.0 / f_rep
        self.b, self.notes = build_bands(fiber, pump, sim, lam_s, input_spec, cw_lines)
        self.lasing_seen = False
        b = self.b
        self.F = np.where(b.direction > 0)[0]
        self.B = np.where(b.direction < 0)[0]
        # rate per ion in ring r per watt in band b per unit cross section
        self.W = b.G / (H * b.nu[:, None] * fiber.A_r[None, :])
        self.hnuE = H * C0 / LAM_ER_LEVEL
        self.hnuY = H * C0 / LAM_YB_LEVEL
        # pump schedule
        if pump.pulsed:
            on = min(pump.on_us * 1e-6, self.T)
            end = (self.T - pump.lead_us * 1e-6) % self.T or self.T
            start = end - on
            self._on = (start, end)
            self.edges = sorted({e % self.T for e in (start, end) if 0 < e % self.T < self.T})
        else:
            self._on = None
            self.edges = []

    def zero_state(self):
        return State.zeros(self.nz, self.nr)

    # --------------------------------------------------------------- pump
    def pump_scale(self, t0, t1):
        """Mean pump scale factor over [t0, t1] within a period."""
        if self._on is None:
            return 1.0
        s, e = self._on
        off = self.pump.off_fraction

        def on_overlap(a, b):
            tot = 0.0
            for k in (-1, 0, 1):
                lo, hi = s + k * self.T, e + k * self.T
                tot += max(0.0, min(b, hi) - max(a, lo))
            return tot
        ov = on_overlap(t0, t1)
        return (ov + off * (t1 - t0 - ov)) / max(t1 - t0, 1e-30)

    def duty(self):
        if self._on is None:
            return 1.0
        s, e = self._on
        d = (e - s) / self.T
        return d + self.pump.off_fraction * (1 - d)

    # -------------------------------------------------------- propagation
    def coefficients(self, st: State, idx=None):
        """Gain coefficients (nz, nb): total, Er population-changing, Er loss (pair quench + ESA), Yb."""
        f, b = self.f, self.b
        sl = slice(None) if idx is None else idx
        G = b.G[sl]
        G1 = G.sum(axis=1)[None, :]
        Gn, Gx = st.n @ G.T, st.x @ G.T
        saE, seE, esa = b.saE[sl][None, :], b.seE[sl][None, :], b.esa[sl][None, :]
        gpop = f.N_s * (seE * Gn - saE * (G1 - Gn)) + f.N_p * (seE * Gx - 2 * saE * (G1 - Gx))
        gloss = -f.N_p * saE * Gx - esa * (f.N_s * Gn + f.N_p * Gx)
        if f.has_yb:
            Gy = st.y @ G.T
            gY = f.N_yb * (b.seY[sl][None, :] * Gy - b.saY[sl][None, :] * (G1 - Gy))
            sY = f.N_yb * b.seY[sl][None, :] * Gy
        else:
            gY = sY = 0.0
        s = b.spont[sl][None, :] * (seE * (f.N_s * Gn + f.N_p * Gx) + sY)
        return dict(gion=gpop + gloss + gY, gpop=gpop, gloss=gloss, gY=gY, s=s)

    def propagate(self, st: State, pscale=1.0):
        b, dz = self.b, self.dz
        co = self.coefficients(st)
        gion, s = co["gion"], co["s"]
        g = gion - b.alpha[None, :]
        nb = len(b)
        Pin = b.P_in * np.where(b.is_pump, pscale, 1.0)
        F, B = self.F, self.B
        pF, hF, mpF, mhF = sweep(g[:, F], s[:, F], dz)
        pB, hB, mpB, mhB = sweep(g[::-1][:, B], s[::-1][:, B], dz)
        P0F = Pin[F].copy()
        PLB = Pin[B].copy()
        nA = self.b.pair_f.size
        RT = np.zeros(nA)
        if nA:
            # forward ASE bands are F[:nA], backward ASE bands are B[:nA] (same wavelengths, see build_bands)
            Tf, Sf = hF[-1, :nA], pF[-1, :nA]
            Tb, Sb = hB[-1, :nA], pB[-1, :nA]
            R1, R2 = self.f.R_in, self.f.R_out
            RT = R1 * R2 * Tf * Tb
            den = np.maximum(1 - RT, 1 - self.RT_CLAMP)
            Pif, Pib = Pin[F][:nA], Pin[B][:nA]
            P0F[:nA] = (Pif + R1 * Tb * Pib + R1 * Tb * R2 * Sf + R1 * Sb) / den
            PfL = Tf * P0F[:nA] + Sf
            PLB[:nA] = Pib + R2 * PfL
        Pn = np.zeros((self.nz + 1, nb))
        Pm = np.zeros((self.nz, nb))
        Pn[:, F] = pF + hF * P0F[None, :]
        Pm[:, F] = mpF + mhF * P0F[None, :]
        Pn[:, B] = (pB + hB * PLB[None, :])[::-1]
        Pm[:, B] = (mpB + mhB * PLB[None, :])[::-1]
        return dict(Pn=Pn, Pm=Pm, RT=RT, Pin=Pin, **co)

    def transfer_eta(self, st: State):
        """Fraction of Yb->Er transfers that relax to 4I13/2 instead of returning to Yb."""
        f = self.f
        return 1.0 / (1.0 + f.tau_32 * f.k_bt * f.N_yb * (1 - st.y))

    def rates(self, st: State, pr):
        """Exponential-update coefficients (A, B) per species: dp/dt = A (1 - p) - B p."""
        f, b = self.f, self.b
        Pm = pr["Pm"]
        RaE = Pm @ (self.W * b.saE[:, None])
        ReE = Pm @ (self.W * b.seE[:, None])
        if f.has_yb:
            eta = self.transfer_eta(st)
            T = f.k_tr * f.N_yb * st.y * eta
            RaY = Pm @ (self.W * b.saY[:, None])
            ReY = Pm @ (self.W * b.seY[:, None])
            ground = f.N_s * (1 - st.n) + f.N_p * (2 * (1 - st.x) + st.x)
            AY, BY = RaY, ReY + 1 / f.tau_yb + f.k_tr * eta * ground
        else:
            T = 0.0
            AY = BY = np.zeros_like(st.n)
        An = RaE + T
        Bn = ReE + 1 / f.tau_er + f.c_up * f.N_s * st.n
        Ax = 2 * (RaE + T)
        Bx = ReE + 1 / f.tau_er
        return (An, Bn), (Ax, Bx), (AY, BY)

    def clamp(self, st: State):
        """Gain clamping above parasitic-lasing threshold.

        The round-trip gain of the cavity closed by the end reflections cannot exceed ~1:
        above threshold the excess excitation is emitted as parasitic laser light. ln(RT)
        is linear in the populations, so scaling every population of one species towards its
        transparency value at the lasing wavelength by kappa is solved exactly for the largest
        kappa that keeps every band at or below RT_CLAMP. Returns the clamped state, the removed
        excitations as (species, wavelength, ions) tuples and the ions removed per cell.
        """
        f, b, dz = self.f, self.b, self.dz
        nA = b.pair_f.size
        removed = []
        per_cell = np.zeros(self.nz)
        if nA == 0 or f.R_in * f.R_out <= 0:
            return st, removed, per_cell
        idx = b.pair_f
        lnR = math.log(f.R_in * f.R_out)
        c = math.log(self.RT_CLAMP)
        loss = 2 * b.alpha[idx] * f.L
        for _ in range(6):
            co = self.coefficients(st, idx)
            G1 = co["gion"].sum(axis=0) * dz
            ln1 = lnR + 2 * G1 - loss
            k = int(np.argmax(ln1))
            if ln1[k] <= c + 1e-9:
                break
            # clamp the species that provides the gain in the worst band
            gE_k = float((co["gpop"] + co["gloss"])[:, k].sum())
            gY_k = float(np.sum(co["gY"][:, k])) if f.has_yb else 0.0
            species = "Yb" if (f.has_yb and gY_k > gE_k) else "Er"
            if species == "Er":
                sa, se = b.saE[idx][k], b.seE[idx][k]
                if sa + se <= 0:
                    break
                ntr = sa / (sa + se)
                xtr = min(1.0, 2 * sa / (sa + se))
                tr = State(np.minimum(st.n, ntr), np.minimum(st.x, xtr), st.y)
            else:
                sa, se = b.saY[idx][k], b.seY[idx][k]
                if sa + se <= 0:
                    break
                tr = State(st.n, st.x, np.minimum(st.y, sa / (sa + se)))
            G0 = self.coefficients(tr, idx)["gion"].sum(axis=0) * dz
            ln0 = lnR + 2 * G0 - loss
            # largest kappa that brings every band this species can fix down to the limit; bands that stay
            # above it even with this species at transparency are left for the other species
            kappa = 1.0
            for l0, l1 in zip(ln0, ln1):
                if l1 > c and l0 < c:
                    kappa = min(kappa, (c - l0) / (l1 - l0))
            kappa = max(0.0, kappa)
            if kappa >= 1.0 - 1e-12:
                break
            new = State(tr.n + kappa * (st.n - tr.n), tr.x + kappa * (st.x - tr.x),
                        tr.y + kappa * (st.y - tr.y))
            if species == "Er":
                cell = ((st.n - new.n) @ f.A_r * f.N_s + (st.x - new.x) @ f.A_r * f.N_p) * dz
            else:
                cell = (st.y - new.y) @ f.A_r * f.N_yb * dz
            if cell.sum() <= 0:
                break
            removed.append((species, float(b.lam[idx][k]), float(cell.sum())))
            per_cell += np.maximum(cell, 0) * H * C0 / b.lam[idx][k]
            st = new
        return st, removed, per_cell

    @staticmethod
    def _expupdate(p, A, Bc, dt):
        k = A + Bc
        peq = np.where(k > 0, A / np.where(k > 0, k, 1), p)
        return peq + (p - peq) * np.exp(-k * dt)

    @staticmethod
    def _expmean(p, A, Bc, dt):
        """Time-average over dt of the frozen-rate solution p(t) (see _expupdate)."""
        k = A + Bc
        peq = np.where(k > 0, A / np.where(k > 0, k, 1), p)
        x = k * dt
        f = np.where(x > 1e-8, -np.expm1(-x) / np.where(x > 1e-8, x, 1), 1 - 0.5 * x)
        return peq + (p - peq) * f

    def _mean_over_step(self, st: State, R, dt):
        (An, Bn), (Ax, Bx), (AY, BY) = R
        return State(self._expmean(st.n, An, Bn, dt),
                     self._expmean(st.x, Ax, Bx, dt) if self.f.N_p > 0 else st.x,
                     self._expmean(st.y, AY, BY, dt) if self.f.has_yb else st.y)

    def _update(self, st: State, R, dt):
        (An, Bn), (Ax, Bx), (AY, BY) = R
        return State(self._expupdate(st.n, An, Bn, dt),
                     self._expupdate(st.x, Ax, Bx, dt) if self.f.N_p > 0 else st.x,
                     self._expupdate(st.y, AY, BY, dt) if self.f.has_yb else st.y)

    # ----------------------------------------------------------- evolve
    def evolve(self, st: State, t_start, duration, averaged_pump=False, measure=None):
        """Advance populations from t_start (within the period) for `duration` seconds."""
        t, t_end = t_start, t_start + duration
        rmax_seen, steps = 0.0, 0
        while t < t_end * (1 - 1e-12):
            # do not step across pump edges
            nxt = t_end
            if not averaged_pump:
                tp = t % self.T
                base = t - tp
                for e in self.edges:
                    for kk in (0, 1):
                        ee = base + e + kk * self.T
                        if t + 1e-15 < ee < nxt:
                            nxt = ee
            ps_probe = self.duty() if averaged_pump else self.pump_scale(t % self.T, t % self.T + 1e-12)
            # predictor: fluxes at the start of the step fix the step size and a half-step state
            pr0 = self.propagate(st, ps_probe)
            R0 = self.rates(st, pr0)
            r = float(max(np.max(a + bb) for a, bb in R0))
            rmax_seen = max(rmax_seen, r)
            # accuracy-limited step, but never below 1/steps_per_period of a period: under
            # parasitic lasing the rates become enormous and the exponential midpoint update
            # (unconditionally stable) then tracks the clamped quasi-equilibrium instead
            dt_floor = min(self.T, duration) / max(self.sim.max_steps_per_period, 1)
            dt = min(nxt - t, max(self.sim.cfl / max(r, 1e-30), dt_floor))
            # implicit step: find the time-averaged populations over the step that are consistent with
            # the light they generate. Fluxes are evaluated at that mean state, the populations follow the
            # exact frozen-rate solution, and its time average must return the same mean state. Near the
            # parasitic-lasing threshold the ASE depends exponentially on the inversion and an explicit
            # (frozen-flux) step overshoots; the self-consistent mean lets the ASE hold the inversion back
            # within the step, and both the ion ledger and the optical energy balance close exactly.
            # (the mean state is kept at or below the lasing threshold: above it the cavity has no
            # steady state and the light it carries is meaningless)
            sta = self.clamp(self._mean_over_step(st, R0, dt))[0]
            shp = [sta.n.shape, sta.x.shape, sta.y.shape]
            flat = lambda q: np.concatenate([q.n.ravel(), q.x.ravel(), q.y.ravel()])

            def unflat(v):
                a1, a2 = shp[0][0] * shp[0][1], shp[1][0] * shp[1][1]
                return State(v[:a1].reshape(shp[0]), v[a1:a1 + a2].reshape(shp[1]), v[a1 + a2:].reshape(shp[2])).clip()
            X, Fh = [], []                                   # Anderson history: iterates and residuals
            for it in range(self.IMPLICIT_ITERS):
                pr = self.propagate(sta, ps_probe)
                R = self.rates(sta, pr)
                x = flat(sta)
                res = flat(self._mean_over_step(st, R, dt)) - x
                err = float(np.max(np.abs(res))) if res.size else 0.0
                if err < self.IMPLICIT_TOL or it == self.IMPLICIT_ITERS - 1:
                    break
                X.append(x); Fh.append(res)
                X, Fh = X[-4:], Fh[-4:]
                beta = 0.7
                if len(X) > 1:
                    dX = np.array([X[i + 1] - X[i] for i in range(len(X) - 1)]).T
                    dF = np.array([Fh[i + 1] - Fh[i] for i in range(len(Fh) - 1)]).T
                    gam, *_ = np.linalg.lstsq(dF, res, rcond=None)
                    xn = x + beta * res - (dX + beta * dF) @ gam
                else:
                    xn = x + beta * res
                sta = self.clamp(unflat(xn))[0]
            st_new = self._update(st, R, dt).clip()
            if measure is not None:
                measure.accumulate(self, sta, pr, t, dt)
            st, removed, per_cell = self.clamp(st_new)
            if removed:
                self.lasing_seen = True
                if measure is not None:
                    measure.add_lasing(removed, per_cell)
            t += dt
            steps += 1
            if steps > 200000:
                raise RuntimeError("inter-pulse integration exceeded 200000 steps")
        return st, rmax_seen, steps

    # --------------------------------------------------------- periodic
    def solve_periodic(self, deplete, st: State | None = None, progress=None, label="inversion"):
        """Find the periodic steady state of (pulse depletion at t=0, evolution over T).

        Convergence is judged on the estimated *distance* to the periodic state, not on the change
        per period: when the period is much shorter than the population relaxation time the state
        changes very little per period even far from steady state. The per-period change contracts
        by a factor q (estimated from successive changes), so the remaining distance is
        change * q / (1 - q); the same estimate extrapolates the state ahead (vector Aitken step).
        Periods are lumped into averaged macro-steps while the state is far from the orbit."""
        st = self.zero_state() if st is None else st.copy()
        M, calm, history = 1, 0, []
        rmax = 1.0 / self.f.tau_er
        prev = None                       # previous per-period change vector (M = 1 only)
        since_extrap = 0
        lump_ok = True                    # lumped macro-steps only for the initial approach
        # slowest possible relaxation: spontaneous decay alone
        q_floor = math.exp(-self.T / self.f.tau_er)
        for it in range(self.sim.max_periods):
            pre = st.copy()
            post = deplete(st)
            dn_max = st.max_diff(post)
            if M > 1:
                post = st.lerp_from(post, M)
            st, rmax, _ = self.evolve(post, 0.0, M * self.T, averaged_pump=M > 1)
            d = np.concatenate([(st.n - pre.n).ravel(), (st.x - pre.x).ravel()]) / M
            change = float(np.max(np.abs(d)))
            if M == 1 and prev is not None and float(prev @ prev) > 0:
                q = float(np.clip((d @ prev) / (prev @ prev), 0.0, 0.9999))
            else:
                q = q_floor if M == 1 else 0.0
            dist = change * q / (1 - q) + change if M == 1 else change * M
            history.append(dict(iter=it, M=M, change=change, q=q, dist=dist))
            if progress:
                progress(min(0.99, it / 60), f"{label}: period {it+1}, distance {dist:.1e}, lumping {M}")
            if M == 1 and dist < self.sim.tol:
                calm += 1
                if calm >= 2:
                    return st, dict(converged=True, iterations=it + 1, history=history)
            else:
                calm = 0
            # stagnation: single periods that stop getting closer mean there is no periodic orbit
            # (e.g. self-pulsing of a parasitic cavity); stop instead of running to max_periods
            dh = [h["dist"] for h in history if h["M"] == 1]
            if len(dh) >= 24 and min(dh[-12:]) > 0.5 * min(dh[-24:-12]):
                return st, dict(converged=False, iterations=it + 1, history=history, stagnated=True)
            since_extrap += 1
            if M == 1 and prev is not None and 0.3 < q < 0.9999 and since_extrap >= 2 and dist > self.sim.tol:
                # geometric tail of the slow mode: jump to its limit
                k = q / (1 - q)
                st = State(st.n + k * (st.n - pre.n), st.x + k * (st.x - pre.x), st.y).clip()
                since_extrap = 0
                prev = None
                continue
            prev = d if M == 1 else None
            # choose lumping for the next macro-step
            lim_r = 0.3 / max(rmax * self.T, 1e-30)
            lim_d = 0.02 / max(dn_max, 1e-12)
            M_new = int(max(1, min(lim_r, lim_d, 2000))) if lump_ok else 1
            if dist < 20 * self.sim.tol:
                M_new = 1
            if M > 1 and M_new == 1:
                lump_ok = False           # from here on: single periods + Aitken extrapolation
            if M_new != M:
                prev = None
            M = M_new
        return st, dict(converged=False, iterations=self.sim.max_periods, history=history)


class Measurement:
    """Time-averaged powers, ion-exchange budget, heat and traces over one resolved period."""

    def __init__(self, model: SlowModel):
        self.m = model
        nb = len(model.b)
        nz = model.nz
        self.T = 0.0
        self.P_out = np.zeros(nb)          # energy leaving each band at its exit end
        self.P_inj = np.zeros(nb)
        self.ion_x = np.zeros(nb)          # energy given by ions to each band (stimulated, net, incl. losses)
        self.ion_pop_E = np.zeros(nb)      # Er population-changing share (emission - absorption)
        self.ion_loss_E = np.zeros(nb)     # Er pair-quench absorption + ESA (<= 0, becomes heat)
        self.ion_Y = np.zeros(nb)          # Yb share
        self.spont_in = np.zeros(nb)       # spontaneous energy emitted into guided band
        self.bg = np.zeros(nb)
        self.decay_er = self.upc = self.decay_yb = 0.0
        self.transfer = self.transfer_wasted = 0.0   # useful (to 4I13/2) and into quenched pairs, ions/s
        self.max_RT = 0.0
        self.lasing = {}
        self.lasing_z = np.zeros(nz)
        self.heat_z = np.zeros(nz)          # net optical power absorbed per cell minus radiative decay
        self.trace = dict(t=[], nE_mean=[], ase_f=[], ase_b=[], pump_res=[], pump_abs=[], pscale=[])
        self.P_z_mean = np.zeros((nz + 1, nb))

    def add_lasing(self, removed, per_cell):
        for species, lam, ions in removed:
            key = (species, round(lam * 1e9, 1))
            self.lasing[key] = self.lasing.get(key, 0.0) + ions
        self.lasing_z += per_cell

    def lasing_power(self):
        """Parasitic laser power by (species, wavelength nm), W, and photons/s by species."""
        T = max(self.T, 1e-30)
        P = {k: v * H * C0 / (k[1] * 1e-9) / T for k, v in self.lasing.items()}
        ph = {"Er": 0.0, "Yb": 0.0}
        for (sp, _), v in self.lasing.items():
            ph[sp] += v / T
        return P, ph

    def accumulate(self, m: SlowModel, st: State, pr, t, dt):
        b, f, dz = m.b, m.f, m.dz
        Pn, Pm = pr["Pn"], pr["Pm"]
        exit_P = np.where(b.direction > 0, Pn[-1], Pn[0])
        self.P_out += exit_P * dt
        self.P_inj += pr["Pin"] * dt
        self.ion_x += (pr["gion"] * Pm).sum(axis=0) * dz * dt
        self.ion_pop_E += (pr["gpop"] * Pm).sum(axis=0) * dz * dt
        self.ion_loss_E += (pr["gloss"] * Pm).sum(axis=0) * dz * dt
        if f.has_yb:
            self.ion_Y += (pr["gY"] * Pm).sum(axis=0) * dz * dt
        self.spont_in += pr["s"].sum(axis=0) * dz * dt
        self.bg += (b.alpha[None, :] * Pm).sum(axis=0) * dz * dt
        A = f.A_r
        exc_er = (f.N_s * st.n + f.N_p * st.x) @ A                     # excited Er per metre, per cell
        dec_er = exc_er / f.tau_er * m.hnuE
        upc = (f.c_up * f.N_s ** 2 * st.n ** 2) @ A * m.hnuE
        self.decay_er += dec_er.sum() * dz * dt
        self.upc += upc.sum() * dz * dt
        dec_yb = 0.0
        if f.has_yb:
            eta = m.transfer_eta(st)
            dec_yb = (f.N_yb * st.y / f.tau_yb) @ A * m.hnuY
            self.decay_yb += dec_yb.sum() * dz * dt
            base = f.k_tr * f.N_yb * st.y * eta
            useful = (base * (f.N_s * (1 - st.n) + 2 * f.N_p * (1 - st.x))) @ A
            wasted = (base * f.N_p * st.x) @ A
            self.transfer += useful.sum() * dz * dt
            self.transfer_wasted += wasted.sum() * dz * dt
        # local heat per metre: optical power absorbed by the ions and the background minus radiative decay
        absorbed = -(pr["gion"] * Pm).sum(axis=1) + (b.alpha[None, :] * Pm).sum(axis=1)
        self.heat_z += (absorbed - dec_er - dec_yb) * dt
        self.max_RT = max(self.max_RT, float(pr["RT"].max()) if pr["RT"].size else 0.0)
        self.P_z_mean += Pn * dt
        self.T += dt
        tr = self.trace
        ase = b.kind == "ase"
        tr["t"].append(t)
        tr["nE_mean"].append(float(er_excited(f, st).mean()))
        tr["ase_f"].append(float(exit_P[ase & (b.direction > 0)].sum()))
        tr["ase_b"].append(float(exit_P[ase & (b.direction < 0)].sum()))
        pp = b.is_pump
        tr["pump_res"].append(float(exit_P[pp].sum()))
        tr["pump_abs"].append(float(-(pr["gion"][:, pp] * Pm[:, pp]).sum() * dz))
        tr["pscale"].append(float(pr["Pin"][pp].sum() / max(b.P_in[pp].sum(), 1e-30)) if pp.any() else 0.0)

    def averages(self):
        T = max(self.T, 1e-30)
        return dict(P_out=self.P_out / T, P_inj=self.P_inj / T, ion_x=self.ion_x / T,
                    ion_pop_E=self.ion_pop_E / T, ion_loss_E=self.ion_loss_E / T, ion_Y=self.ion_Y / T,
                    spont_in=self.spont_in / T, bg=self.bg / T, decay_er=self.decay_er / T,
                    upc=self.upc / T, decay_yb=self.decay_yb / T, transfer_rate=self.transfer / T,
                    transfer_wasted_rate=self.transfer_wasted / T, heat_z=self.heat_z / T,
                    lasing_z=self.lasing_z / T, P_z=self.P_z_mean / T)
