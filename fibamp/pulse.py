"""Intra-pulse (fast-axis) propagation of the complex signal envelope(s).

The signal is carried by one complex envelope per guided mode (one for the LP01
and core-filling models, up to ``max_modes`` LP modes for the multimode model).
Modes add incoherently in power; they share the ring-resolved inversion, so a
mode that overlaps the depleted centre of the core sees less gain (transverse
spatial hole burning). Per fibre cell:

  1. saturated gain, time-resolved: each ring's populations follow the exact
     solution for a fixed photon fluence (isolated ions towards transparency,
     pairs towards their own equilibrium, ESA population-neutral), evaluated with
     the cell-mean fluence (predictor-corrector); every photon a mode gains is
     taken from a specific ring and species, so the ledger closes exactly;
  2. spectral gain correction exp[(g(nu) - g(nu0)) dz / 2] from the cross sections
     at the cell's mean populations, booked photon by photon against the rings;
  3. dispersion (beta2, beta3) and, with the noise floor on, spontaneous emission
     into the mode (Langevin source with the emission spectrum and local N2);
  4. nonlinearity, either
     * ``band`` model (ns pulses): SPM + XPM, and a cascade of Raman Stokes orders
       as time-resolved power bands seeded by spontaneous scattering (photon
       conserving; Er gain/absorption applied to each order), or
     * ``gnlse`` model (ps pulses): generalised NLSE in the interaction picture
       (RK4IP) with the delayed Raman response (f_R = 0.18) and self-steepening;
       Raman then appears in the field spectrum itself.

Conventions: carrier exp(+i w0 t), envelope component at FFT frequency f is the
optical frequency nu0 + f; propagation phase exp(-i beta z).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.fft as sfft

from .fiber import Fiber
from .inversion import State
from .seed import SeedField, pulse_stats, spectral_widths
from .xsections import H, C0

N_SNAP_T = 1024
N_SNAP_F = 1024
N_SNAP_Z = 100


def _fft(a):
    return sfft.fft(a, axis=-1, workers=-1)


def _ifft(a):
    return sfft.ifft(a, axis=-1, workers=-1)


def _expm1_over(x):
    """(e^x - 1) / x without branches (series near 0)."""
    small = np.abs(x) < 1e-6
    xs = np.where(small, 1.0, x)
    return np.where(small, 1 + 0.5 * x, np.expm1(xs) / xs)


def resolve_nl_model(sim, fwhm_s):
    if sim.nl_model in ("band", "gnlse"):
        return sim.nl_model
    return "gnlse" if fwhm_s < sim.gnlse_below_ns * 1e-9 else "band"


def raman_response(nt, dt, tau1=12.2e-15, tau2=32e-15):
    """FFT of the causal Raman response h_R sampled on the grid (normalised to unit area)."""
    t = np.arange(nt) * dt
    h = (tau1 ** 2 + tau2 ** 2) / (tau1 * tau2 ** 2) * np.exp(-t / tau2) * np.sin(t / tau1)
    h[t > 40 * tau2] = 0.0
    s = h.sum() * dt
    if s <= 0:
        return np.ones(nt, complex)
    return np.fft.fft(h / s) * dt


def _species_gain(fib, n, p, sa, se, esa):
    """Per-ring gain density (1/m per unit overlap) during a pulse, split into (isolated, paired, ESA loss).

    Within a ns pulse the upconversion that quenches a doubly excited pair (~us) has no time to act,
    so the 2 N_p paired ions behave as ordinary ions with excited fraction p; the pair constraint is
    applied after the pulse by `_pair_quench`."""
    gs = fib.N_s * (se * n - sa * (1 - n))
    gp = 2 * fib.N_p * (se * p - sa * (1 - p))
    gl = -esa * (fib.N_s * n + 2 * fib.N_p * p)
    return gs, gp, gl


def _pair_quench(x0, p_new, ntr):
    """Pair state after the pulse and the doubly excited pairs that quench (per pair).

    Before the pulse a fraction x0 of pairs holds one excitation. Each ion then follows the same affine
    map p -> ntr + (p - ntr) e under the pulse, independently, so a ground pair ends doubly excited
    with probability a^2 and a singly excited pair with a b (a, b: the images of 0 and 1). e is chosen
    to reproduce the photon-counted excited fraction p_new; excitations are conserved apart from the
    quenched ones."""
    p0 = 0.5 * x0
    den = p0 - ntr
    e = np.clip(np.where(np.abs(den) > 1e-12, (p_new - ntr) / np.where(np.abs(den) > 1e-12, den, 1), 1.0), 0, 1)
    a = ntr * (1 - e)
    b = ntr + (1 - ntr) * e
    P11 = (1 - x0) * a * a + x0 * a * b
    return np.clip(2 * p_new - P11, 0.0, 1.0), P11


def fn_deplete(fiber: Fiber, st: State, E_in, lam_s, dz):
    """Energy-only saturation pass (launch-weighted modes, no time resolution): post-pulse state and E(z).

    Per cell and ring the populations follow the exact saturation law for the cell-mean photon fluence,
    n -> ntr + (n - ntr) exp(-(sigma_a + sigma_e) F), and the photons the modes gain or lose are counted
    from that population change, so the result can never overshoot transparency and conserves photons
    (a Frantz-Nodvik pass generalised to rings, modes and paired ions). Used to converge the pulse train
    cheaply; the full solver's post-pulse state then corrects it."""
    sm = fiber.signal_modes(lam_s)
    G, w = sm["G"], sm["w"]
    saE, seE = (float(v) for v in fiber.xs_er_at(lam_s))
    saY, seY = (float(v) for v in fiber.xs_yb_at(lam_s))
    hnu = H * C0 / lam_s
    A_r = fiber.A_r
    Ns, Np = fiber.N_s, fiber.N_p
    ntr, cs = saE / (saE + seE), saE + seE
    post = st.copy()
    nz = st.n.shape[0]
    Ez = np.empty(nz + 1)
    Em = E_in * w
    Ez[0] = Em.sum()
    for j in range(nz):
        n0, x0 = st.n[j], st.x[j]
        p0 = 0.5 * x0
        Emean = Em
        for _ in range(3):
            F = (G.T @ Emean) / (hnu * A_r)                          # photons per m^2 in each ring
            e_ = np.exp(-cs * F)
            n1 = ntr + (n0 - ntr) * e_
            p1 = ntr + (p0 - ntr) * e_
            ph = ((n0 - n1) * Ns + (p0 - p1) * 2 * Np) * A_r * dz     # photons released per ring
            share = G * Emean[:, None]                                # (nm, nr): who carries the light in ring r
            tot = share.sum(axis=0)
            share = np.divide(share, tot, out=np.zeros_like(share), where=tot > 0)
            Eout = np.maximum(Em + hnu * (share @ ph), 0.0)
            Emean = 0.5 * (Em + Eout)
        post.n[j] = n1
        if Np > 0:
            post.x[j] = _pair_quench(x0, p1, ntr)[0]
        Em = Eout
        if fiber.has_yb:
            gY = G @ (fiber.N_yb * (seY * st.y[j] - saY * (1 - st.y[j]))) * dz
            Em = Em * np.exp(gY)
        Em = Em * np.exp(-(fiber.alpha_bg + sm["alpha_bend"]) * dz)
        Ez[j + 1] = Em.sum()
    return post, Ez


def _blocks(v, n):
    """Block-average the last axis down to n samples (n divides the length or it is padded)."""
    m = v.shape[-1]
    if m <= n:
        return v
    k = m // n
    return v[..., : k * n].reshape(*v.shape[:-1], n, k).mean(axis=-1)


def propagate_pulse(fiber: Fiber, seed: SeedField, st_pre: State, sim, progress=None, rng=None):
    rng = rng or np.random.default_rng(seed.meta.get("cfg", {}).get("rng_seed", 1) + 17)
    t = seed.t
    dt, nt = seed.dt, t.size
    lam0 = seed.lam0
    nu0 = C0 / lam0
    w0 = 2 * np.pi * nu0
    hnu0 = H * nu0
    nz = st_pre.n.shape[0]
    dz = fiber.L / nz
    f = np.fft.fftfreq(nt, dt)
    Om = 2 * np.pi * f
    nu_f = np.maximum(nu0 + f, 1.0)
    lam_f = C0 / nu_f
    nl_model = seed.meta.get("nl_model", resolve_nl_model(sim, 1e-9))

    sm = fiber.signal_modes(lam0)
    G, aeffs, a_bend, wl = sm["G"], sm["aeff"], sm["alpha_bend"], sm["w"]
    nm = G.shape[0]
    A_r = fiber.A_r
    Ns, Np = fiber.N_s, fiber.N_p
    A = np.sqrt(wl)[:, None] * seed.A.astype(complex)[None, :]
    if nm > 1 and sim.quantum_noise:
        A[1:] += np.sqrt(hnu0 / dt / 2) * (rng.normal(size=(nm - 1, nt)) + 1j * rng.normal(size=(nm - 1, nt)))

    saE0, seE0 = (float(v) for v in fiber.xs_er_at(lam0))
    esa0 = float(fiber.esa_at(lam0))
    saY0, seY0 = (float(v) for v in fiber.xs_yb_at(lam0))
    ntr, cs_n = saE0 / (saE0 + seE0), saE0 + seE0
    gnl = 2 * np.pi * fiber.ds.n2_m2_W / (lam0 * aeffs) if sim.spm else np.zeros(nm)
    use_spec = sim.spectral_gain
    saEf, seEf = fiber.xs_er_at(lam_f)
    esaf = fiber.esa_at(lam_f)
    dsa, dse, desa = saEf - saE0, seEf - seE0, esaf - esa0
    hnu_f = H * nu_f
    D = -1j * (fiber.beta2 / 2 * Om ** 2 + fiber.beta3 / 6 * Om ** 3) if sim.dispersion else np.zeros(nt)
    Dop = np.exp(D * dz)
    noise = sim.quantum_noise and sim.spont_noise

    # Raman: band cascade (ns) or delayed response in the field (gnlse)
    gnlse = nl_model == "gnlse"
    K = 0 if (gnlse or not sim.raman) else max(1, int(sim.raman_orders))
    OmR = sim.raman_shift_THz * 1e12
    nuR = np.array([nu0 - (k + 1) * OmR for k in range(K)])
    lamR = C0 / nuR if K else np.zeros(0)
    gR = fiber.ds.gR_m_W / aeffs                                      # (nm,)
    PR = np.zeros((K, nm, nt))
    if K:
        saR, seR = fiber.xs_er_at(lamR)
        esaR = fiber.esa_at(lamR)
        Psp = H * nuR * sim.raman_noise_THz * 1e12
    fR = 0.18
    hRf = raman_response(nt, dt) if (gnlse and sim.raman) else None

    # SBS field store (low-passed dominant modes)
    sbs_modes = [0] if nm == 1 else list(np.argsort(-wl)[:3])
    sbs_store = None
    if sim.sbs:
        Tw = nt * dt
        _, rms0 = spectral_widths(t, seed.A, lam0)
        dnu = max(C0 * rms0 * 2.355 / lam0 ** 2, 1e6)
        dts = min(sim.sbs_dt_ns * 1e-9, 0.2 / dnu)
        M = int(min(2 ** math.ceil(math.log2(max(Tw / dts, 64))), 16384, nt))
        sbs_store = dict(M=M, dt=Tw / M, modes=sbs_modes, A=np.zeros((0,), np.complex64),
                         aeff=aeffs[sbs_modes], res_limited=Tw / M > 0.2 / dnu * 1.01)

    def lowpass(Af_modes):
        M = sbs_store["M"]
        h = M // 2
        sub = np.concatenate([Af_modes[:, :h], Af_modes[:, -h:]], axis=1)
        return (_ifft(sub) * (M / nt)).astype(np.complex64)

    st = st_pre.copy()
    Ez = np.zeros(nz + 1); Ppk = np.zeros(nz + 1)
    Em_z = np.zeros((nz + 1, nm)); ER_z = np.zeros((K, nz + 1))
    stride = max(1, int(math.ceil(nz / N_SNAP_Z)))
    snap_idx = sorted(set(list(range(0, nz + 1, stride)) + [nz]))
    snap_P = np.zeros((len(snap_idx), min(N_SNAP_T, nt)), np.float32)
    snap_S = np.zeros((len(snap_idx), min(N_SNAP_F, nt)), np.float32)
    P0 = np.abs(A) ** 2
    Em_z[0] = P0.sum(axis=1) * dt
    Ez[0], Ppk[0] = Em_z[0].sum(), float(P0.sum(axis=0).max())
    snap_P[0] = _blocks(P0.sum(axis=0), snap_P.shape[1])
    if sbs_store is not None:
        sbs_store["A"] = np.zeros((len(snap_idx), len(sbs_modes), sbs_store["M"]), np.complex64)
        sbs_store["z_idx"] = np.array(snap_idx)
    Af0 = _fft(A)
    snap_S[0] = _blocks(np.fft.fftshift((np.abs(Af0) ** 2).sum(axis=0)), snap_S.shape[1]) * snap_S.shape[1] / nt * dt / nt
    if sbs_store is not None:
        sbs_store["A"][0] = lowpass(Af0[sbs_modes])
    B_int = G_R = 0.0
    quench_z = np.zeros(nz)                                           # doubly excited pairs quenched after the pulse
    photon_gain = 0.0
    ions_R = 0.0
    nsub_max = 1
    step = max(1, nz // 20)

    for j in range(nz):
        n0, x_pre, y0 = st.n[j].copy(), st.x[j].copy(), st.y[j]
        p0 = 0.5 * x_pre                                              # excited fraction of paired ions
        # ------------------------------------------------ 1. saturated gain
        P = np.abs(A) ** 2
        Eb = float(P.sum() * dt)
        Uc = (np.cumsum(P, axis=1) - 0.5 * P) * dt
        for it in range(2):
            Fr = (G.T @ Uc) / (hnu0 * A_r[:, None])                  # photons per m^2, (nr, nt)
            e_ = np.exp(-cs_n * Fr)
            n_t = ntr + (n0 - ntr)[:, None] * e_
            if Np > 0:
                p_t = ntr + (p0 - ntr)[:, None] * e_
                gs, gp, gl = _species_gain(fiber, n_t, p_t, saE0, seE0, esa0)
            else:
                gs = Ns * ((saE0 + seE0) * n_t - saE0)
                gp = 0.0
                gl = -esa0 * Ns * n_t if esa0 else 0.0
            lnG = (G @ (gs + gp + gl)) * dz                          # (nm, nt)
            ff = _expm1_over(lnG)
            if it == 0:
                Pf = P * ff
                Uc = (np.cumsum(Pf, axis=1) - 0.5 * Pf) * dt
        A *= np.exp(0.5 * lnG)
        Wr = G.T @ (P * ff)                                          # power in each ring, (nr, nt)
        ex_s = (Wr * gs).sum(axis=1) * dz * dt                       # J given by isolated ions per ring
        ex_p = (Wr * gp).sum(axis=1) * dz * dt if Np > 0 else 0.0
        dE_gain = float((P * np.expm1(lnG)).sum() * dt)
        n_new = n0 - (ex_s / (hnu0 * Ns * A_r * dz) if Ns > 0 else 0.0)
        p_new = p0 - (ex_p / (hnu0 * 2 * Np * A_r * dz) if Np > 0 else 0.0)
        photon_gain += dE_gain
        # ------------------------- 2+3. spectral gain, dispersion, spontaneous noise
        Af = _fft(A)
        nb_, pb_ = 0.5 * (n0 + n_new), 0.5 * (p0 + p_new)
        if use_spec:
            ce = Ns * nb_ + 2 * Np * pb_
            ca_s, ca_p = -Ns * (1 - nb_), -2 * Np * (1 - pb_)
            ca = ca_s + ca_p
            dg = (G @ ce)[:, None] * dse + (G @ ca)[:, None] * dsa - (G @ ce)[:, None] * desa
            S = np.abs(Af) ** 2 * (dt / nt)                           # J per bin
            fac = _expm1_over(dg * dz)
            Sph = S * fac / hnu_f                                     # mean photons per bin
            Xe, Xa = (Sph @ dse) * dz, (Sph @ dsa) * dz              # (nm,)
            ph_s = G.T @ Xe * Ns * nb_ + G.T @ Xa * ca_s              # photons given by isolated ions per ring
            ph_p = G.T @ Xe * 2 * Np * pb_ + G.T @ Xa * ca_p
            if Ns > 0:
                n_new = n_new - ph_s / (Ns * A_r * dz)
            if Np > 0:
                p_new = p_new - ph_p / (2 * Np * A_r * dz)
            Af *= np.exp(0.5 * dg * dz)
            photon_gain += float((S * np.expm1(dg * dz)).sum())
        if noise:
            N2 = Ns * nb_ + 2 * Np * pb_                              # excited Er per ring
            psd = hnu_f[None, :] * seEf[None, :] * (G @ N2)[:, None] * dz     # W/Hz per mode
            sd = np.sqrt(np.maximum(psd, 0) / dt * nt / 2)
            Af += sd * (rng.normal(size=Af.shape) + 1j * rng.normal(size=Af.shape))
        if not gnlse:
            Af *= Dop[None, :]
        A = _ifft(Af)
        # Yb at the signal (normally ~0), background and bend loss
        gY = (G @ (fiber.N_yb * (seY0 * y0 - saY0 * (1 - y0)))) if fiber.has_yb else np.zeros(nm)
        A *= np.exp(0.5 * (gY - fiber.alpha_bg - a_bend) * dz)[:, None]
        # ------------------------------------------------ 4. nonlinearity
        Pmax = (np.abs(A) ** 2).max(axis=1)
        nsub = int(max(1, math.ceil(float((gnl * Pmax).max()) * dz / sim.max_phase_step)))
        if K:
            nsub = max(nsub, int(math.ceil(float((gR * Pmax).max()) * dz / 0.5)))
        nsub = min(nsub, sim.max_nl_substeps)
        nsub_max = max(nsub_max, nsub)
        h = dz / nsub
        if gnlse:
            Dh = np.exp(D * h / 2)[None, :]
            steep = (1 + Om / w0)[None, :] if sim.self_steepening else 1.0
            g_ = gnl[:, None]

            def Nop(Bf):
                a = _ifft(Bf)
                I = np.abs(a) ** 2
                if hRf is not None:
                    conv = _ifft(hRf[None, :] * _fft(I)).real
                    I = (1 - fR) * I + fR * conv
                return -1j * g_ * steep * _fft(a * I)
            Af = _fft(A)
            for _ in range(nsub):
                AI = Dh * Af
                k1 = Dh * Nop(Af) * h
                k2 = Nop(AI + k1 / 2) * h
                k3 = Nop(AI + k2 / 2) * h
                k4 = Nop(Dh * (AI + k3)) * h
                Af = Dh * (AI + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6
            A = _ifft(Af)
        else:
            for _ in range(nsub):
                Ps = np.abs(A) ** 2
                if K:
                    pump = Ps
                    Ps_new = Ps
                    for k in range(K):
                        nup = nu0 if k == 0 else nuR[k - 1]
                        gk = gR[:, None] * (nup / nu0)
                        PRn = (PR[k] + Psp[k]) * np.exp(np.minimum(gk * pump * h, 50.0)) - Psp[k]
                        dR = PRn - PR[k]
                        pump_new = pump - (nup / nuR[k]) * dR
                        bad = pump_new < 0
                        if bad.any():
                            dR[bad] = pump[bad] * nuR[k] / nup
                            PRn[bad] = PR[k][bad] + dR[bad]
                            pump_new[bad] = 0.0
                        if k == 0:
                            Ps_new = pump_new
                        else:
                            PR[k - 1] = pump_new
                        PR[k] = PRn
                        pump = PRn
                    scale = np.sqrt(np.divide(Ps_new, Ps, out=np.ones_like(Ps), where=Ps > 0))
                    A *= scale
                if gnl.any():
                    xpm = 2 * PR.sum(axis=0) if K else 0.0
                    A *= np.exp(-1j * gnl[:, None] * (np.abs(A) ** 2 + xpm) * h)
        # Er gain / absorption of the Stokes orders, booked against the rings
        for k in range(K):
            gs_k, gp_k, gl_k = _species_gain(fiber, n_new, p_new, saR[k], seR[k], esaR[k])
            lnGk = (G @ (gs_k + gp_k + gl_k)) * dz
            Eb_k = PR[k].sum(axis=1) * dt
            ffk = np.where(np.abs(lnGk) > 1e-12, np.expm1(lnGk) / np.where(lnGk == 0, 1, lnGk), 1.0)
            Wk = G.T @ (Eb_k * ffk)
            hk = H * nuR[k]
            if Ns > 0:
                n_new = n_new - Wk * gs_k * dz / (hk * Ns * A_r * dz)
            if Np > 0:
                p_new = p_new - Wk * gp_k * dz / (hk * 2 * Np * A_r * dz)
            ions_R += float((Wk * (gs_k + gp_k)).sum() * dz / hk)
            PR[k] *= np.exp(lnGk - (fiber.alpha_bg + a_bend) * dz)[:, None]
        st.n[j] = np.clip(n_new, 0.0, 1.0)
        if Np > 0:
            st.x[j], P11 = _pair_quench(x_pre, p_new, ntr)
            quench_z[j] = float(P11 @ A_r) * Np * dz
        # ------------------------------------------------ records
        P = np.abs(A) ** 2
        Ptot = P.sum(axis=0)
        Em_z[j + 1] = P.sum(axis=1) * dt
        Ez[j + 1] = Em_z[j + 1].sum()
        Ppk[j + 1] = float(Ptot.max())
        for k in range(K):
            ER_z[k, j + 1] = float(PR[k].sum() * dt)
        B_int += float((gnl * P.max(axis=1)).max()) * dz
        G_R += float((gR * P.max(axis=1)).max()) * dz
        if (j + 1) in snap_idx:
            q = snap_idx.index(j + 1)
            snap_P[q] = _blocks(Ptot, snap_P.shape[1])
            Af = _fft(A)
            snap_S[q] = _blocks(np.fft.fftshift((np.abs(Af) ** 2).sum(axis=0)), snap_S.shape[1]) \
                * snap_S.shape[1] / nt * dt / nt
            if sbs_store is not None:
                sbs_store["A"][q] = lowpass(Af[sbs_modes])
        if progress and (j % step == 0):
            progress(j / nz, f"pulse propagation: cell {j+1}/{nz}")

    ions_removed = float(((st_pre.n - st.n) @ A_r * Ns + (st_pre.x - st.x) @ A_r * Np).sum() * dz)
    Ptot = (np.abs(A) ** 2).sum(axis=0)
    if nm == 1:
        A_tot = A[0].copy()
    else:
        dom = int(np.argmax(Em_z[-1]))
        A_tot = np.sqrt(Ptot) * np.exp(1j * np.angle(A[dom]))
    nuS = np.fft.fftshift(nu_f)
    return dict(t=t, A_in=seed.A, A_out=A_tot, A_modes=A, mode_names=sm["names"], PR=PR.sum(axis=1),
                st_post=st, Ez=Ez, Em_z=Em_z, ERz=ER_z, Ppk_z=Ppk, B=B_int, G_R=G_R, nsub_max=nsub_max,
                photon_gain=photon_gain, ions_removed=ions_removed, ions_raman=ions_R, lamR=lamR,
                ions_quenched_z=quench_z,
                Aeff=float(aeffs[int(np.argmax(wl))]), gamma_nl=float(gnl[int(np.argmax(wl))]),
                nl_model=nl_model, snap_P=snap_P, snap_S=snap_S,
                snap_t=_blocks(t, snap_P.shape[1]), snap_lam=C0 / _blocks(nuS, snap_S.shape[1]),
                snap_z=np.array(snap_idx) * dz,
                sbs_store=sbs_store)


def pulse_diagnostics(res, seed: SeedField, fiber: Fiber, sim, sbs=None):
    """Window, aliasing and margin checks on a finished pulse run."""
    t, A = res["t"], res["A_out"]
    Am = res["A_modes"]
    P = np.abs(A) ** 2
    warn = []
    edge = max(P[: max(1, t.size // 50)].max(), P[-max(1, t.size // 50):].max())
    if edge > 1e-4 * P.max():
        warn.append(f"output pulse reaches the edge of the time window ({edge/P.max():.1e} of peak): "
                    "raise the window multiplier")
    Af = np.fft.fftshift((np.abs(np.fft.fft(Am, axis=1)) ** 2).sum(axis=0))
    k = max(1, Af.size // 20)
    edge_s = (Af[:k].sum() + Af[-k:].sum()) / max(Af.sum(), 1e-300)
    if edge_s > 1e-3:
        warn.append(f"{edge_s*100:.2f} % of the output spectrum lies in the outer 10 % of the spectral "
                    "window: widen the spectral window (aliasing)")
    st_in, st_out = pulse_stats(t, res["A_in"]), pulse_stats(t, A)
    fw_in, rms_in = spectral_widths(t, res["A_in"], seed.lam0)
    fw_out, rms_out = spectral_widths(t, Am, seed.lam0)
    tau = max(st_out["fwhm"], 1e-12)
    I_pk = st_out["P_peak"] / res["Aeff"]
    I_bulk = sim.damage_bulk_GW_cm2_10ns * 1e13 * math.sqrt(tau / 10e-9)
    I_facet = sim.damage_facet_GW_cm2_10ns * 1e13 * math.sqrt(tau / 10e-9)
    # analytic SBS estimate (always reported; the simulated Stokes wave supersedes it when enabled)
    Pz = res["Ppk_z"]
    z = np.linspace(0, fiber.L, Pz.size)
    Leff = float(np.trapezoid(Pz, z) / max(Pz[-1], 1e-30))
    dnu_s = C0 * rms_out * 2.355 / seed.lam0 ** 2
    dnuB = sim.sbs_dnuB_MHz * 1e6
    gB = sim.sbs_gB_m_W * dnuB / (dnuB + dnu_s) * min(1.0, tau / (sim.sbs_tau_ns * 1e-9))
    P_sbs = 21 * res["Aeff"] / (gB * max(Leff, 1e-9))
    dtt = t[1] - t[0]
    if res["nl_model"] == "gnlse":
        # Raman energy = field energy red-shifted by more than half the Stokes shift
        nu = np.fft.fftfreq(t.size, dtt)
        S = (np.abs(np.fft.fft(Am, axis=1)) ** 2).sum(axis=0)
        red = nu < -0.5 * sim.raman_shift_THz * 1e12
        E_R = float(S[red].sum() / max(S.sum(), 1e-300) * st_out["E"])
    else:
        E_R = float(np.sum(res["PR"]) * dtt)
    raman_frac = E_R / max(E_R + st_out["E"], 1e-300) if res["nl_model"] != "gnlse" else E_R / max(st_out["E"], 1e-300)
    K = res["PR"].shape[0] if res["nl_model"] != "gnlse" else 0
    if raman_frac > 0.01:
        last = float(np.sum(res["PR"][-1]) * dtt) / max(E_R, 1e-300) if K else 0.0
        msg = f"Raman conversion {raman_frac*100:.1f} % (> 1 %)"
        if K and last > 0.1:
            msg += f"; {last*100:.0f} % of it is in the highest modelled Stokes order ({K}): add orders"
        warn.append(msg)
    if I_pk > I_bulk:
        warn.append(f"peak intensity {I_pk/1e13:.1f} GW/cm2 exceeds the indicative bulk damage level "
                    f"{I_bulk/1e13:.1f} GW/cm2 for a {tau*1e9:.2f} ns pulse")
    if sbs is None and st_out["P_peak"] > P_sbs:
        warn.append(f"peak power exceeds the estimated SBS threshold ({st_out['P_peak']/P_sbs:.1f}x); "
                    "broaden the seed linewidth")
    if res["nsub_max"] >= sim.max_nl_substeps:
        warn.append(f"nonlinear sub-steps capped at {sim.max_nl_substeps} per cell: increase nz")
    elif res["nsub_max"] > 200:
        warn.append(f"up to {res['nsub_max']} nonlinear sub-steps per cell were needed; results are "
                    "valid but increase nz to speed up")
    return dict(in_stats=st_in, out_stats=st_out, lw_in_fwhm=fw_in, lw_in_rms=rms_in,
                lw_out_fwhm=fw_out, lw_out_rms=rms_out, I_peak=I_pk, I_bulk=I_bulk, I_facet=I_facet,
                P_sbs=P_sbs, L_eff=Leff, E_raman=E_R, raman_frac=raman_frac, warnings=warn)
