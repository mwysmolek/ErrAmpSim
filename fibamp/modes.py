"""Step-index LP modes: propagation constants, radial profiles, ring overlaps, A_eff, bend loss.

Weakly guiding step-index fibre. For each azimuthal order l the characteristic
equation (Gloge)

    u J_{l-1}(u) K_l(w) + w K_{l-1}(w) J_l(u) = 0,     u^2 + w^2 = V^2

is solved for every guided radial order m. Intensity profiles are azimuthally
averaged (cos^2(l phi) -> 1/2), which is what a radially resolved inversion sees.
Bend loss follows Marcuse (JOSA 66, 216, 1976) with the elasto-optic correction
R_eff = 1.28 R.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, kve

N_SILICA = 1.444


@dataclass
class LPMode:
    l: int
    m: int
    u: float
    w: float
    V: float

    @property
    def name(self):
        return f"LP{self.l}{self.m}"

    @property
    def degeneracy(self):
        return 2 if self.l == 0 else 4

    @property
    def b(self):
        return 1.0 - (self.u / self.V) ** 2


def _char(l, u, V):
    w = np.sqrt(np.maximum(V * V - u * u, 1e-300))
    if l == 0:
        return -u * jv(1, u) * kve(0, w) + w * kve(1, w) * jv(0, u)
    return u * jv(l - 1, u) * kve(l, w) + w * kve(l - 1, w) * jv(l, u)


def solve_modes(V: float, max_modes: int | None = None) -> list[LPMode]:
    """All guided LP modes at normalised frequency V, lowest order (largest b) first."""
    out = []
    if V <= 0:
        return out
    n = int(max(400, 60 * V))
    uu = np.linspace(1e-6, V * (1 - 1e-9), n)
    for l in range(0, int(V) + 2):
        f = _char(l, uu, V)
        s = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        m = 0
        for i in s:
            try:
                u = brentq(lambda x: float(_char(l, x, V)), uu[i], uu[i + 1], xtol=1e-12)
            except ValueError:
                continue
            m += 1
            out.append(LPMode(l, m, u, math.sqrt(max(V * V - u * u, 0.0)), V))
        if l > 0 and not s.size:
            break
    out.sort(key=lambda md: md.u)
    return out[:max_modes] if max_modes else out


def radial_intensity(md: LPMode, r, a):
    """Azimuthally averaged intensity (unnormalised) at radius r for core radius a."""
    r = np.asarray(r, float)
    x = r / a
    inside = jv(md.l, md.u * x) ** 2
    w = max(md.w, 1e-9)
    ratio = kve(md.l, w * np.maximum(x, 1e-12)) / kve(md.l, w) * np.exp(-w * (x - 1.0))
    outside = (jv(md.l, md.u) * ratio) ** 2
    return np.where(x <= 1.0, inside, outside)


def _grid(md: LPMode, a, r_max):
    rm = min(r_max, a * (1 + 14.0 / max(md.w, 0.3)))
    rm = max(rm, 1.2 * a)
    return np.linspace(0.0, rm, 6000)


def ring_overlaps(md: LPMode, a, ring_edges, r_max):
    """Fraction of the mode power inside each annulus [edges[k], edges[k+1]]."""
    r = _grid(md, a, r_max)
    I = radial_intensity(md, r, a) * r
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(r))])
    tot = cum[-1]
    c = np.interp(ring_edges, r, cum) / tot
    return np.diff(c)


def aeff(md: LPMode, a, r_max):
    r = _grid(md, a, r_max)
    I = radial_intensity(md, r, a)
    s1 = np.trapezoid(I * r, r)
    s2 = np.trapezoid(I * I * r, r)
    if md.l == 0:
        return (2 * np.pi * s1) ** 2 / (2 * np.pi * s2)
    # I(r, phi) = R(r)^2 * 2 cos^2(l phi): int cos^2 = pi, int cos^4 = 3 pi / 4
    return (2 * np.pi * s1) ** 2 / (2 * np.pi * 1.5 * s2)


def bend_loss(md: LPMode, a, lam, R_bend, n_core=N_SILICA):
    """Power attenuation coefficient (1/m) of a coiled step-index fibre (Marcuse 1976)."""
    if not R_bend or R_bend <= 0:
        return 0.0
    R = 1.28 * R_bend
    kappa, gam = md.u / a, max(md.w, 1e-6) / a
    beta = 2 * np.pi * n_core / lam
    e = 2.0 if md.l == 0 else 1.0
    ga = gam * a
    # K_{l-1} K_{l+1} with exponential scaling: kve(x) = K(x) e^x
    lk = (math.log(max(kve(abs(md.l - 1), ga), 1e-300)) + math.log(max(kve(md.l + 1, ga), 1e-300)) - 2 * ga)
    ln_alpha = (0.5 * math.log(math.pi) + 2 * math.log(kappa) - (2 * gam ** 3 / (3 * beta ** 2)) * R
                - math.log(e) - 1.5 * math.log(gam) - 2 * math.log(md.V) - 0.5 * math.log(R) - lk)
    return float(math.exp(min(ln_alpha, 50.0)))
