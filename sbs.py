"""Transient stimulated Brillouin scattering: the backward Stokes wave, simulated.

Three-wave model in the slowly varying envelope approximation, lab frame,
Stokes counter-propagating (towards z = 0):

    dA_p/dz  = -kappa A_s rho            + (forward-solver evolution of the pump)
    dA_s/dz' =  kappa A_p rho*  +  (g_Er - alpha) A_s / 2           (z' = -z)
    d rho/dt =  Gamma (A_p A_s* - rho) + f(z, t)

kappa = g_B / (2 A_eff), Gamma = pi dnu_B (acoustic amplitude damping). In steady
state rho = A_p A_s* and the Stokes grows as g_B P_p / A_eff while the pump loses
the same number of photons. A pump linewidth above dnu_B, or a pulse shorter than
1 / Gamma, drives the acoustic wave less efficiently -- both fall out of the
equations. The Langevin source f (thermal phonons, occupation n_th + 1) is
normalised so that a CW pump in a passive fibre gives the spontaneous Brillouin
power per unit length (n_th + 1) h nu (g_B P_p / A_eff) (pi dnu_B / 2).

The pump enters with the seed field and is carried through the SBS grid using
the forward solver's own cell-to-cell evolution (gain saturation, reshaping,
SPM) as a transfer function, and it is depleted by the Stokes wave. The forward
solver itself does not see that depletion, so when the reflected fraction is
more than ~1 % the forward result is an upper bound and the depleted output
energy computed here is the better estimate.

Characteristic grid: dz = v_g dt, so both waves advance one cell per step; the
grid is refined until the Brillouin gain per cell is small.
"""
from __future__ import annotations

import math

import numpy as np

from .xsections import H, C0, KB

N_G = 1.468
G_RUNAWAY = 150.0          # steady-state gain exponent beyond which the pulse is destroyed outright


def _interp_c(x, xp, fp):
    return np.interp(x, xp, fp.real, left=0, right=0) + 1j * np.interp(x, xp, fp.imag, left=0, right=0)


def simulate_sbs(fiber, res, sim, gain_z, lam0, rng=None, max_cells=3000, max_work=4e7):
    """Backward Stokes wave for each stored mode. gain_z: Er net gain (1/m) of the signal per solver cell."""
    store = res["sbs_store"]
    if store is None:
        return None
    rng = rng or np.random.default_rng(7)
    A_all = store["A"]                                  # (n_stored, n_modes, M) complex, retarded time
    nst, nmode, M = A_all.shape
    dts = store["dt"]
    vg = C0 / N_G
    L = fiber.L
    zi = np.asarray(store.get("z_idx", np.arange(nst)), float)
    z_st = zi / zi[-1] * L
    zc_solver = (np.arange(gain_z.size) + 0.5) * L / gain_z.size
    dnuB = sim.sbs_dnuB_MHz * 1e6
    Gam = math.pi * dnuB
    nuB = sim.sbs_shift_GHz * 1e9
    n_th = 1.0 / math.expm1(H * nuB / (KB * sim.fibre_temperature_K))
    hnu = H * C0 / lam0
    tau_st = (np.arange(M) - M // 2) * dts
    out = []
    for im in range(nmode):
        aeff = float(store["aeff"][im])
        kap = sim.sbs_gB_m_W / (2 * aeff)
        P_st = np.abs(A_all[:, im, :]) ** 2
        # steady-state gain exponent seen at each retarded time (the usual threshold measure, ~21)
        G_ss = float(np.max(np.trapezoid(P_st, z_st, axis=0))) * 2 * kap
        # the same, reduced for the pump linewidth (Lorentzian gain / broad pump) and pulse length:
        # only used to decide whether the pulse is destroyed outright and to size the grid
        Sp = np.abs(np.fft.fft(A_all[-1, im, :])) ** 2
        fq = np.fft.fftfreq(M, dts)
        mu = (fq * Sp).sum() / max(Sp.sum(), 1e-300)
        dnu_p = 2.355 * math.sqrt(max(((fq - mu) ** 2 * Sp).sum() / max(Sp.sum(), 1e-300), 0.0))
        P_last = P_st[-1]
        above = P_last > 0.5 * P_last.max()
        tau_p = max(above.sum() * dts, dts)
        # Lorentzian Brillouin line convolved with a Gaussian pump line: Voigt peak / Lorentz peak
        # -> sqrt(pi ln2) dnu_B / dnu_p for a broad pump
        G_eff = G_ss / (1.0 + dnu_p / (1.476 * dnuB)) * min(1.0, tau_p * Gam / 2)
        rec = dict(aeff=aeff, G_B=G_ss, G_eff=G_eff, dnu_pump=dnu_p, runaway=G_eff > G_RUNAWAY)
        if G_ss < 1e-3 or rec["runaway"]:
            rec.update(t=np.zeros(2), P=np.zeros(2), E=0.0 if G_ss < 1e-3 else float("nan"),
                       E_pump_out=float("nan"))
            out.append(rec)
            continue
        # grid: resolve the stored pump in time and keep the Brillouin gain per cell below ~0.2
        Ns = int(max(20, math.ceil(L / (vg * dts)), math.ceil(G_eff / 0.2)))
        Ns = min(Ns, max_cells)
        dz = L / Ns
        dtl = dz / vg
        Ml = int(math.ceil(M * dts / dtl))
        if (Ns + 1) * Ml > max_work:
            Ml = int(max_work / (Ns + 1))
            dtl_eff = M * dts / Ml
        else:
            dtl_eff = dtl
        tau_l = tau_st[0] + np.arange(Ml) * dtl_eff
        zb = np.arange(Ns + 1) * dz
        # pump at the SBS boundaries on the lab-step retarded-time grid
        Ab = np.empty((Ns + 1, Ml), complex)
        Ast = np.array([_interp_c(tau_l, tau_st, A_all[q, im, :]) for q in range(nst)])
        wz = np.interp(zb, z_st, np.arange(nst))
        i0 = np.minimum(wz.astype(int), nst - 2)
        fr = (wz - i0)[:, None]
        Ab[:] = (1 - fr) * Ast[i0] + fr * Ast[i0 + 1]
        mag = np.abs(Ab)
        ok = mag[:-1] > 1e-6 * max(mag.max(), 1e-300)
        R = np.where(ok, Ab[1:] / np.where(ok, Ab[:-1], 1), 1.0)       # forward-solver cell transfer
        zc = (np.arange(Ns) + 0.5) * dz
        g_c = np.interp(zc, zc_solver, gain_z) - fiber.alpha_bg
        gainexp = np.exp(0.5 * g_c * dz)
        ex = math.exp(-Gam * dtl)
        Q = 2 * (n_th + 1) * hnu * Gam ** 2 / kap
        noise_sd = math.sqrt(Q * (1 - ex ** 2) / (2 * Gam) / dz / 2)
        n_steps = Ml + Ns
        S = np.zeros(Ns + 1, complex)                   # Stokes at boundaries, S[Ns] = 0 (far end)
        Pp = np.zeros(Ns + 1, complex)                  # pump at boundaries (entering the cell to the right)
        rho = np.zeros(Ns, complex)
        P_back = np.zeros(n_steps)
        P_fwd = np.zeros(n_steps)
        kk = np.arange(Ns)
        for j in range(n_steps):
            Pp[0] = Ab[0, j] if j < Ml else 0.0
            Ap = Pp[:-1]
            Smid = 0.5 * (S[:-1] + S[1:])
            rho = rho * ex + Ap * np.conj(Smid) * (1 - ex) \
                + noise_sd * (rng.normal(size=Ns) + 1j * rng.normal(size=Ns))
            # pump photons handed to the Stokes in this cell; never more than the pump holds (strong
            # SBS with an explicit step could otherwise overshoot) -- the same factor scales both waves
            dP = kap * Smid * rho * dz
            lim = np.minimum(1.0, 0.5 * np.abs(Ap) / np.maximum(np.abs(dP), 1e-300))
            Snew = np.empty_like(S)
            Snew[Ns] = 0.0
            Snew[:Ns] = (S[1:] + kap * Ap * np.conj(rho) * dz * lim) * gainexp
            idx = j - kk
            valid = (idx >= 0) & (idx < Ml)
            Rk = np.ones(Ns, complex)
            Rk[valid] = R[kk[valid], idx[valid]]
            Pnew = np.empty_like(Pp)
            Pnew[0] = 0.0
            Pnew[1:] = (Ap - dP * lim) * Rk
            S, Pp = Snew, Pnew
            P_back[j] = abs(S[0]) ** 2
            P_fwd[j] = abs(Pp[Ns]) ** 2
        rec.update(t=tau_st[0] + np.arange(n_steps) * dtl, P=P_back, E=float(P_back.sum() * dtl),
                   E_pump_out=float(P_fwd.sum() * dtl), E_pump_ref=float((np.abs(Ab[-1]) ** 2).sum() * dtl_eff),
                   n_cells=Ns, dt=dtl)
        out.append(rec)
    runaway = any(o["runaway"] for o in out)
    sim_out = [o for o in out if not o["runaway"] and o["P"].size > 2]
    E_tot = float("nan") if runaway else sum(o["E"] for o in out)
    if sim_out:
        t = sim_out[0]["t"]
        P = np.sum([np.interp(t, o["t"], o["P"]) for o in sim_out], axis=0)
    else:
        t, P = np.zeros(2), np.zeros(2)
    depl = [1 - o["E_pump_out"] / o["E_pump_ref"] for o in sim_out if o.get("E_pump_ref", 0) > 0]
    return dict(t=t, P=P, E=E_tot, per_mode=out, G_B=max(o["G_B"] for o in out),
                G_eff=max(o.get("G_eff", 0.0) for o in out), runaway=runaway,
                pump_depletion=float(max(depl)) if depl else 0.0, n_th=n_th,
                lam=C0 / (C0 / lam0 - nuB), res_limited=store.get("res_limited", False))
