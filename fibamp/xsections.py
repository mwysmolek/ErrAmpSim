"""Spectroscopic cross-section library.

Tables are CSV files with columns ``lambda_nm, sigma_a_m2, sigma_e_m2`` and
optional ``#`` comment lines. The shipped tables are representative
literature-class spectra (see the header of each file); replace them with your
fibre vendor's measured spectra for quantitative work -- either by dropping a
CSV into ``fibamp/data`` or by uploading one in the app.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import numpy as np

H = 6.62607015e-34
C0 = 2.99792458e8
KB = 1.380649e-23

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

LIBRARY = {
    "Er (aluminosilicate)": "er_aluminosilicate.csv",
    "Er (phosphosilicate)": "er_phosphosilicate.csv",
    "Yb (phosphosilicate)": "yb_phosphosilicate.csv",
    "ESA: Er 4I13/2": "er_esa.csv",
}

UNIT_SCALE = {"m2": 1.0, "cm2": 1e-4, "1e-25 m2": 1e-25, "1e-24 m2": 1e-24, "pm2": 1e-24}


@dataclass
class CrossSection:
    """Tabulated absorption / emission cross sections on a wavelength grid (SI)."""

    name: str
    lam: np.ndarray          # m, ascending
    sa: np.ndarray           # m^2
    se: np.ndarray           # m^2
    notes: str = ""
    source: str = field(default="")

    def __call__(self, lam_m):
        """Interpolated (sigma_a, sigma_e) at wavelengths in metres; zero outside the table."""
        lam_m = np.asarray(lam_m, float)
        sa = np.interp(lam_m, self.lam, self.sa, left=0.0, right=0.0)
        se = np.interp(lam_m, self.lam, self.se, left=0.0, right=0.0)
        return sa, se

    @property
    def range_nm(self):
        return float(self.lam[0] * 1e9), float(self.lam[-1] * 1e9)

    def to_csv(self) -> str:
        lines = [f"# {ln}" for ln in self.notes.splitlines() if ln.strip()]
        lines.append("lambda_nm,sigma_a_m2,sigma_e_m2")
        lines += [f"{l*1e9:.3f},{a:.5e},{e:.5e}" for l, a, e in zip(self.lam, self.sa, self.se)]
        return "\n".join(lines) + "\n"


def _parse(text: str, name: str, unit: str = "m2", source: str = "") -> CrossSection:
    notes, rows, header = [], [], None
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if ln.startswith("#"):
            notes.append(ln.lstrip("#").strip())
            continue
        parts = [p.strip() for p in ln.replace(";", ",").replace("\t", ",").split(",") if p.strip()]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            header = parts
            continue
        if len(vals) < 2:
            continue
        rows.append(vals[:3] if len(vals) >= 3 else [vals[0], vals[1], 0.0])
    if len(rows) < 3:
        raise ValueError(f"cross-section table '{name}' has fewer than 3 data rows")
    arr = np.array(rows, float)
    arr = arr[np.argsort(arr[:, 0])]
    lam = arr[:, 0]
    # accept nm or um in the first column
    if lam.max() < 10.0:
        lam = lam * 1e3
    s = UNIT_SCALE[unit]
    return CrossSection(name=name, lam=lam * 1e-9, sa=np.maximum(arr[:, 1] * s, 0.0),
                        se=np.maximum(arr[:, 2] * s, 0.0), notes="\n".join(notes), source=source)


def load_table(path: str, name: str | None = None, unit: str = "m2") -> CrossSection:
    with open(path, "r", encoding="utf-8") as fh:
        return _parse(fh.read(), name or os.path.basename(path), unit, source=path)


def from_bytes(data: bytes, name: str, unit: str = "m2") -> CrossSection:
    return _parse(io.BytesIO(data).read().decode("utf-8", errors="replace"), name, unit,
                  source="uploaded")


_CACHE: dict[str, CrossSection] = {}


def library(name: str) -> CrossSection:
    if name not in _CACHE:
        if name not in LIBRARY:
            raise KeyError(f"unknown cross-section table '{name}'; available: {list(LIBRARY)}")
        _CACHE[name] = load_table(os.path.join(DATA_DIR, LIBRARY[name]), name)
    return _CACHE[name]


def register(xs: CrossSection) -> None:
    """Make a custom (e.g. uploaded vendor) table available under its name."""
    _CACHE[xs.name] = xs
    LIBRARY.setdefault(xs.name, "")


def mccumber(lam_m, sigma_a, lam_zero_m, T=300.0):
    """Emission cross section from absorption via McCumber theory."""
    dE = H * C0 * (1.0 / lam_zero_m - 1.0 / np.asarray(lam_m, float))
    return np.asarray(sigma_a, float) * np.exp(dE / (KB * T))
