"""Seed pulse construction: analytic shapes, CSV import, previous-stage import."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import SeedConfig, SimConfig
from .xsections import H, C0


@dataclass
class SeedField:
    t: np.ndarray                    # s, uniform grid
    A: np.ndarray                    # complex envelope, |A|^2 in W
    lam0: float                      # carrier wavelength, m
    f_rep: float                     # Hz
    ase_lam: Optional[np.ndarray] = None   # m, input CW spectrum (ASE of a previous stage, leakage)
    ase_psd: Optional[np.ndarray] = None   # W per m of wavelength
    notes: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def dt(self):
        return float(self.t[1] - self.t[0])

    @property
    def P(self):
        return np.abs(self.A) ** 2

    @property
    def energy(self):
        return float(np.sum(self.P) * self.dt)


# ------------------------------------------------------------------ analysis
def fwhm(t, P):
    P = np.asarray(P, float)
    if P.max() <= 0:
        return 0.0
    k = np.where(P >= 0.5 * P.max())[0]
    i0, i1 = k[0], k[-1]
    def cross(i, j):                  # linear interpolation of the half-max crossing between i and j
        if j < 0 or j >= P.size or P[i] == P[j]:
            return t[i]
        return t[j] + (0.5 * P.max() - P[j]) * (t[i] - t[j]) / (P[i] - P[j])
    return float(cross(i1, i1 + 1) - cross(i0, i0 - 1))


def pulse_stats(t, A):
    P = np.abs(A) ** 2
    dt = t[1] - t[0]
    E = float(P.sum() * dt)
    if E <= 0:
        return dict(E=0.0, P_peak=0.0, fwhm=0.0, t_peak=0.0, centroid=0.0, rms=0.0)
    c = float((t * P).sum() * dt / E)
    return dict(E=E, P_peak=float(P.max()), fwhm=fwhm(t, P), t_peak=float(t[np.argmax(P)]),
                centroid=c, rms=float(np.sqrt(((t - c) ** 2 * P).sum() * dt / E)))


def spectrum(t, A, lam0):
    """Energy spectral density on a wavelength grid: returns lam (m, ascending), S (J per m).

    A may be 2-D (one envelope per mode); mode spectra add incoherently."""
    dt = t[1] - t[0]
    n = t.size
    Af = np.fft.fftshift(np.fft.fft(A, axis=-1), axes=-1) * dt
    f = np.fft.fftshift(np.fft.fftfreq(n, dt))
    nu = C0 / lam0 + f
    lam = C0 / nu
    S_nu = np.abs(Af) ** 2                       # J/Hz
    if S_nu.ndim > 1:
        S_nu = S_nu.sum(axis=0)
    S_lam = S_nu * nu ** 2 / C0                  # J/m
    order = np.argsort(lam)
    return lam[order], S_lam[order]


def spectral_widths(t, A, lam0):
    lam, S = spectrum(t, A, lam0)
    if S.sum() <= 0:
        return 0.0, 0.0
    p = S / S.sum()
    m = (lam * p).sum()
    rms = float(np.sqrt((p * (lam - m) ** 2).sum()))
    k = max(1, S.size // 400)
    Ss = np.convolve(S, np.ones(k) / k, mode="same")
    sel = lam[Ss >= 0.5 * Ss.max()]
    return float(np.ptp(sel)) if sel.size > 1 else 0.0, rms


# ------------------------------------------------------------------- shapes
def _shape(cfg: SeedConfig, t):
    T = cfg.fwhm_ns * 1e-9
    if cfg.shape == "gaussian":
        return np.exp(-4 * np.log(2) * (t / T) ** 2)
    if cfg.shape == "supergaussian":
        m = cfg.sg_order
        return np.exp(-np.log(2) * (2 * np.abs(t) / T) ** (2 * m))
    if cfg.shape == "sech2":
        return 1 / np.cosh(1.7627 * t / T) ** 2
    if cfg.shape == "square":
        e = max(cfg.edge_ns * 1e-9, 1e-12) / 2.197     # 10-90 % of a logistic edge = 2.197 k
        return 1 / (1 + np.exp((np.abs(t) - T / 2) / e))
    if cfg.shape == "asymmetric":
        tr, tf = cfg.rise_ns * 1e-9, cfg.fall_ns * 1e-9
        x = np.clip(np.stack([-t / tr, t / tf]), -700, 700)
        return 1 / (np.exp(x[0]) + np.exp(x[1])) ** 2
    raise ValueError(f"unknown seed shape '{cfg.shape}'")


def _characteristic_fwhm(cfg: SeedConfig):
    if cfg.shape == "asymmetric":
        tr, tf = cfg.rise_ns * 1e-9, cfg.fall_ns * 1e-9
        tt = np.linspace(-30 * tr, 30 * tf, 200001)
        return fwhm(tt, _shape(cfg, tt)), 30 * (tr + tf)
    T = cfg.fwhm_ns * 1e-9
    if cfg.shape == "square":
        return T, T + 8 * cfg.edge_ns * 1e-9
    return T, 0.0


def make_grid(fwhm_s, extent_s, lam0, sim: SimConfig, linewidth_nm=0.0, min_span_nm=None):
    from .pulse import resolve_nl_model
    Tw = max(sim.window_mult * fwhm_s, 1.5 * extent_s)
    tl_nm = 0.441 / fwhm_s * lam0 ** 2 / C0 * 1e9
    needed = max(2.0, 10 * linewidth_nm, 20 * tl_nm, min_span_nm or 0.0)
    span = sim.spectral_window_nm or max(sim.min_spectral_nm, needed)
    if resolve_nl_model(sim, fwhm_s) == "gnlse" and not sim.spectral_window_nm:
        # the field must hold at least two Stokes orders on both sides of the carrier
        span = max(span, 3.4 * sim.raman_shift_THz * 1e12 * lam0 ** 2 / C0 * 1e9)
    dt_req = (lam0 ** 2 / C0) / (span * 1e-9)
    n = int(2 ** np.ceil(np.log2(max(Tw / dt_req, 256))))
    notes = []
    if n > sim.nt_max:
        n = sim.nt_max
        got = (lam0 ** 2 / C0) / (Tw / n) * 1e9
        if got < needed or sim.spectral_window_nm:
            notes.append(f"time grid capped at nt = {n}: spectral window reduced to {got:.2f} nm (requested "
                         f"{span:.2f} nm). Raise 'max time points' or shorten the window if the output spectrum "
                         "reaches the edge.")
    t = (np.arange(n) - n // 2) * (Tw / n)
    return t, notes


def phase_noise(t, dlam_nm, lam0, rng):
    """Random phase giving a Gaussian line of the requested FWHM (envelope untouched).

    The instantaneous frequency is a smooth Gaussian process with rms dnu / 2.355 and a correlation
    time of ~5 / dnu (slow-modulation limit, as for a phase modulator driven by filtered noise), so the
    phase has no singularities and the spectrum has no artificial wings."""
    if dlam_nm <= 0:
        return np.ones(t.size, complex)
    dt = t[1] - t[0]
    dnu = C0 * dlam_nm * 1e-9 / lam0 ** 2
    fq = np.fft.fftfreq(t.size, dt)
    bw = dnu / 5.0
    w = np.fft.ifft(np.exp(-0.5 * (fq / bw) ** 2) * np.fft.fft(rng.normal(size=t.size))).real
    w *= (dnu / 2.355) / max(np.std(w), 1e-300)
    phi = 2 * np.pi * np.cumsum(w) * dt
    return np.exp(1j * phi)


def add_quantum_noise(t, A, lam0, rng):
    """One photon per temporal-spectral mode of the grid: seeds MI and spontaneous processes."""
    dt = t[1] - t[0]
    s = np.sqrt(H * C0 / lam0 / dt / 2)
    return A + s * (rng.normal(size=t.size) + 1j * rng.normal(size=t.size))


def resample(t_old, A_old, t_new):
    dt_o, dt_n = t_old[1] - t_old[0], t_new[1] - t_new[0]
    A = A_old
    if dt_n > dt_o * 1.0001:                     # anti-alias before decimating
        f = np.fft.fftfreq(t_old.size, dt_o)
        A = np.fft.ifft(np.fft.fft(A) * (np.abs(f) <= 0.5 / dt_n))
    re = np.interp(t_new, t_old, A.real, left=0.0, right=0.0)
    im = np.interp(t_new, t_old, A.imag, left=0.0, right=0.0)
    return re + 1j * im


def bandpass(lam, center_nm, fwhm_nm, order=1.0):
    if not fwhm_nm or fwhm_nm <= 0:
        return np.ones_like(lam)
    x = (np.asarray(lam) - center_nm * 1e-9) / (fwhm_nm * 1e-9 / 2)
    return np.exp(-np.log(2) * np.abs(x) ** (2 * order))


# -------------------------------------------------------------------- build
def build_seed(cfg: SeedConfig, sim: SimConfig, csv_data=None, stage=None) -> SeedField:
    rng = np.random.default_rng(cfg.rng_seed)
    lam0 = cfg.wavelength_nm * 1e-9
    f_rep = cfg.rep_rate_kHz * 1e3
    notes = []
    ase_lam = ase_psd = None

    if cfg.source == "analytic":
        fw, ext = _characteristic_fwhm(cfg)
        t, gn = make_grid(fw, ext, lam0, sim, cfg.linewidth_nm)
        notes += gn
        env = _shape(cfg, t)
        if cfg.shape == "asymmetric":
            c = (t * env).sum() / env.sum()
            env = _shape(cfg, t + c)
        P = env / env.max()
        if cfg.specify == "peak":
            P = P * cfg.peak_W
        else:
            P = P * cfg.energy_uJ * 1e-6 / (P.sum() * (t[1] - t[0]))
        A = np.sqrt(P) * phase_noise(t, cfg.linewidth_nm, lam0, rng)

    elif cfg.source == "csv":
        if csv_data is None:
            raise ValueError("csv seed selected but no data supplied")
        tc = np.asarray(csv_data["t_ns"], float) * 1e-9
        Pc = np.maximum(np.asarray(csv_data["P"], float), 0.0)
        ph = np.asarray(csv_data.get("phase", np.zeros_like(Pc)), float)
        fw = fwhm(tc, Pc)
        t, gn = make_grid(fw, np.ptp(tc), lam0, sim, cfg.linewidth_nm)
        notes += gn
        c = np.sum(tc * Pc) / np.sum(Pc)
        A = resample(tc - c, np.sqrt(Pc) * np.exp(1j * ph), t)
        A = A * phase_noise(t, cfg.linewidth_nm, lam0, rng)
        E = np.sum(np.abs(A) ** 2) * (t[1] - t[0])
        if cfg.rescale_energy or np.max(Pc) <= 1.0 + 1e-9:
            A = A * np.sqrt(cfg.energy_uJ * 1e-6 / E)
            if not cfg.rescale_energy:
                notes.append("CSV power looks normalised (max <= 1); scaled to the seed energy setting")

    elif cfg.source == "stage":
        if stage is None:
            raise ValueError("stage seed selected but no stage file loaded")
        ts, As, lam0 = stage["t"], stage["A"], float(stage["lam0"])
        f_rep = float(stage.get("f_rep", f_rep)) if cfg.rep_rate_kHz <= 0 else f_rep
        # inter-stage band-pass filter and loss, applied to the field and the ASE
        n = ts.size
        dt_s = ts[1] - ts[0]
        nu = C0 / lam0 + np.fft.fftfreq(n, dt_s)
        Tf = bandpass(C0 / nu, cfg.filter_center_nm, cfg.filter_fwhm_nm, cfg.filter_order)
        loss = 10 ** (-cfg.interstage_loss_dB / 10)
        As = np.fft.ifft(np.fft.fft(As) * np.sqrt(Tf)) * np.sqrt(loss)
        st = pulse_stats(ts, As)
        wl, _ = spectral_widths(ts, As, lam0)
        min_span = (lam0 ** 2 / C0) / dt_s * 1e9 if sim.spectral_window_nm is None else None
        t, gn = make_grid(st["fwhm"], np.ptp(ts), lam0, sim, 0.0, min_span_nm=min_span)
        notes += gn
        A = resample(ts - st["centroid"], As, t)
        if cfg.rescale_energy:
            A = A * np.sqrt(cfg.energy_uJ * 1e-6 / (np.sum(np.abs(A) ** 2) * (t[1] - t[0])))
        if stage.get("ase_lam") is not None:
            ase_lam = np.asarray(stage["ase_lam"], float)
            ase_psd = np.asarray(stage["ase_psd"], float) * bandpass(
                ase_lam, cfg.filter_center_nm, cfg.filter_fwhm_nm, cfg.filter_order) * loss
        notes.append(f"imported stage '{stage.get('name', 'previous stage')}': "
                     f"{pulse_stats(t, A)['E']*1e6:.3f} uJ after {cfg.interstage_loss_dB:g} dB loss"
                     + (f" and a {cfg.filter_fwhm_nm:g} nm filter" if cfg.filter_fwhm_nm else ""))
    else:
        raise ValueError(f"unknown seed source '{cfg.source}'")

    cw_lines = list(stage.get("cw_lines", [])) if (cfg.source == "stage" and stage) else []
    if cfg.cw_leak_dB is not None:
        E = np.sum(np.abs(A) ** 2) * (t[1] - t[0])
        P_leak = E * f_rep * 10 ** (cfg.cw_leak_dB / 10)
        cw_lines.append((lam0, P_leak))
        notes.append(f"CW leakage of {P_leak*1e3:.3g} mW at the seed wavelength between pulses")

    if sim.quantum_noise:
        A = add_quantum_noise(t, A, lam0, rng)
    from .pulse import resolve_nl_model
    fw_now = pulse_stats(t, A)["fwhm"] or (t[-1] - t[0]) / 8
    return SeedField(t=t, A=A, lam0=lam0, f_rep=f_rep, ase_lam=ase_lam, ase_psd=ase_psd, notes=notes,
                     meta=dict(cfg=cfg.to_dict(), cw_lines=cw_lines, nl_model=resolve_nl_model(sim, fw_now)))
