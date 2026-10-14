"""Starting points: complete, physically sensible amplifier setups.

The bare dataclass defaults are field-by-field defaults, not a working design (an Er-only
fibre with an Er:Yb-style 5 dB/m cladding absorption solves to ~6e26 Er/m^3). Every new
project starts from one of these instead. Fibre data are datasheet-class nominal values;
replace them with your own fibre's datasheet.

``NUMERICS`` holds the accuracy levels offered in the app: *draft* for quick looks and
batches, *standard* (the engine defaults) and *fine* for final numbers.
"""
from __future__ import annotations

from .config import PumpConfig, PumpSource, SeedConfig, SimConfig
from .fiber import AbsorptionSpec, FiberDatasheet


def _p(**kw) -> dict:
    return kw


PRESETS: dict[str, dict] = {
    "L-band core-pumped stage (Er80-8/125 class)": dict(
        notes="1590 nm, 25 ns square pulses at 1 kHz from a 100 nJ seed, 1 W of 976 nm core pump. "
              "LIEKKI Er80-8/125-class fibre (8 um core, NA 0.13, MFD 9.5 um, 80 dB/m peak core absorption).",
        fiber=FiberDatasheet(
            name="Er80-8/125 (nominal)", dopant="Er", er_table="Er (aluminosilicate)",
            d_core_um=8.0, d_clad_um=125.0, na_core=0.13, na_clad=0.46, mfd_um=9.5,
            absorption=[AbsorptionSpec(80.0, 1530.0, "core")], length_m=4.5,
            bg_loss_dB_per_km=10.0, R_in_dB=-50.0, R_out_dB=-50.0),
        pump=PumpConfig(geometry="core", co=PumpSource(1.0, 976.0, 0.5), counter=PumpSource(0.0, 976.0, 0.5)),
        seed=SeedConfig(shape="square", fwhm_ns=25.0, edge_ns=1.0, energy_uJ=0.1, wavelength_nm=1590.0,
                        linewidth_nm=0.1, rep_rate_kHz=1.0),
    ),
    "C-band pre-amplifier (Er80-4/125 class)": dict(
        notes="1550 nm, 10 ns Gaussian pulses at 50 kHz from a 5 nJ seed, 300 mW of 976 nm core pump. "
              "LIEKKI Er80-4/125-class fibre (4 um core, NA 0.20, MFD 6.5 um, 80 dB/m peak core absorption).",
        fiber=FiberDatasheet(
            name="Er80-4/125 (nominal)", dopant="Er", er_table="Er (aluminosilicate)",
            d_core_um=4.0, d_clad_um=125.0, na_core=0.20, na_clad=0.46, mfd_um=6.5,
            absorption=[AbsorptionSpec(80.0, 1530.0, "core")], length_m=2.0,
            bg_loss_dB_per_km=10.0),
        pump=PumpConfig(geometry="core", co=PumpSource(0.3, 976.0, 0.5), counter=PumpSource(0.0, 976.0, 0.5)),
        seed=SeedConfig(shape="gaussian", fwhm_ns=10.0, energy_uJ=0.005, wavelength_nm=1550.0,
                        linewidth_nm=0.02, rep_rate_kHz=50.0),
    ),
    "Er:Yb 25/300 cladding-pumped booster": dict(
        notes="1550 nm, 25 ns Gaussian pulses at 10 kHz from a 10 uJ seed, 10 W of wavelength-stabilised "
              "976 nm cladding pump. Er:Yb 25/300 LMA-class fibre, 5 dB/m cladding absorption at 976 nm.",
        fiber=FiberDatasheet(
            name="Er:Yb 25/300 (nominal)", dopant="Er:Yb", er_table="Er (phosphosilicate)",
            yb_table="Yb (phosphosilicate)", d_core_um=25.0, d_clad_um=300.0, na_core=0.09, na_clad=0.46,
            clad_shape="octagonal", absorption=[AbsorptionSpec(5.0, 976.0, "cladding")], yb_er_ratio=15.0,
            length_m=3.0, bg_loss_dB_per_km=30.0, signal_mode="LP01"),
        pump=PumpConfig(geometry="cladding", co=PumpSource(10.0, 976.0, 0.5), counter=PumpSource(0.0, 976.0, 0.5)),
        seed=SeedConfig(shape="gaussian", fwhm_ns=25.0, energy_uJ=10.0, wavelength_nm=1550.0,
                        linewidth_nm=0.1, rep_rate_kHz=10.0),
    ),
}

DEFAULT_PRESET = next(iter(PRESETS))

NUMERICS: dict[str, dict] = {
    "draft": _p(nz=60, ase_step_nm=2.0, max_periods=200, max_steps_per_period=200, nt_max=2 ** 16,
                consistency_iters=0, tol=5e-4),
    "standard": {},
    "fine": _p(nz=300, ase_step_nm=0.25, tol=5e-5, consistency_iters=2, cfl=0.15, max_phase_step=0.02),
}


def numerics(level: str, base: SimConfig | None = None) -> SimConfig:
    """SimConfig at an accuracy level; physics switches of ``base`` are kept."""
    d = (base or SimConfig()).to_dict()
    std = SimConfig().to_dict()
    for k in set().union(*[set(v) for v in NUMERICS.values()]):
        d[k] = std[k]
    d.update(NUMERICS[level])
    return SimConfig.from_dict(d)


def preset(name: str) -> dict:
    """Plain dicts (fiber, pump, seed, sim, notes) for a project."""
    p = PRESETS[name]
    return dict(fiber=p["fiber"].to_dict(), pump=p["pump"].to_dict(), seed=p["seed"].to_dict(),
                sim=SimConfig().to_dict(), notes=p["notes"])
