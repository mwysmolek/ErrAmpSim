"""Run configuration: pump, seed and numerical settings (all JSON-serialisable)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class PumpSource:
    power_W: float = 0.0
    wavelength_nm: float = 976.0
    fwhm_nm: float = 0.5             # diode spectral width; wavelength-stabilised ~0.5, free-running ~3-5
    n_sub: int = 7                   # spectral sub-bands used to represent the diode line


@dataclass
class PumpConfig:
    geometry: str = "cladding"       # "cladding" or "core"
    co: PumpSource = field(default_factory=lambda: PumpSource(10.0))
    counter: PumpSource = field(default_factory=lambda: PumpSource(0.0))
    pulsed: bool = False
    on_us: float = 200.0             # pump pulse duration
    lead_us: float = 0.0             # pump pulse ends this long before the seed pulse arrives
    off_fraction: float = 0.0        # pump level between pump pulses (simmer), fraction of on-power

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["co"] = PumpSource(**d.get("co", {}))
        d["counter"] = PumpSource(**d.get("counter", {}))
        return PumpConfig(**{k: v for k, v in d.items() if k in PumpConfig.__dataclass_fields__})


@dataclass
class SeedConfig:
    source: str = "analytic"         # "analytic", "csv", "stage"
    shape: str = "gaussian"          # gaussian, supergaussian, sech2, square, asymmetric
    fwhm_ns: float = 25.0
    sg_order: float = 3.0            # super-Gaussian order m in exp(-(t/T)^(2m))
    edge_ns: float = 1.0             # 10-90 % edge of the "square" shape
    rise_ns: float = 5.0             # asymmetric (Q-switch-like) shape: 1/e rise
    fall_ns: float = 20.0            # asymmetric shape: 1/e fall
    energy_uJ: float = 10.0
    specify: str = "energy"          # "energy" or "peak"
    peak_W: float = 400.0
    wavelength_nm: float = 1550.0
    linewidth_nm: float = 0.0        # 0 -> transform-limited; >0 -> phase-noise broadened
    rep_rate_kHz: float = 10.0
    rescale_energy: bool = False     # csv / stage import: rescale to energy_uJ
    interstage_loss_dB: float = 0.0  # stage import: isolator + splices + coupler
    filter_center_nm: float = 1550.0
    filter_fwhm_nm: float = 0.0      # stage import: 0 -> no band-pass filter
    filter_order: float = 1.0        # 1 Gaussian, >1 flatter super-Gaussian
    cw_leak_dB: Optional[float] = None   # modulator extinction: CW leakage relative to average seed power
    rng_seed: int = 1

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return SeedConfig(**{k: v for k, v in dict(d).items() if k in SeedConfig.__dataclass_fields__})


@dataclass
class SimConfig:
    nz: int = 150
    ase: bool = True
    ase_step_nm: float = 0.5
    er_ase_nm: tuple = (1480.0, 1640.0)
    yb_ase_nm: tuple = (1000.0, 1120.0)
    window_mult: float = 8.0         # time window = window_mult x pulse FWHM (>= 8 recommended)
    spectral_window_nm: Optional[float] = None   # None -> automatic
    min_spectral_nm: float = 10.0    # automatic spectral window is at least this wide (in-band ASE, SPM wings)
    nt_max: int = 2 ** 18
    tol: float = 2e-4                # periodic steady state: max change of pre-pulse inversion
    max_periods: int = 400
    cfl: float = 0.3
    max_steps_per_period: int = 400  # time-step floor = period / this (stiff parasitic-lasing cases)
    quantum_noise: bool = True
    raman: bool = True
    spm: bool = True
    dispersion: bool = True
    spectral_gain: bool = True
    max_phase_step: float = 0.05     # rad of SPM per nonlinear sub-step
    consistency_iters: int = 1
    raman_noise_THz: float = 5.0
    raman_shift_THz: float = 13.2
    raman_width_THz: float = 6.0
    raman_orders: int = 3            # band model: cascaded Stokes orders
    nl_model: str = "auto"           # "auto", "band" (ns: SPM + Raman bands) or "gnlse" (ps: full field)
    gnlse_below_ns: float = 0.3      # auto: pulses shorter than this use the GNLSE
    self_steepening: bool = True
    spont_noise: bool = True         # spontaneous emission into the signal modes (in-band ASE in the window)
    max_nl_substeps: int = 2000
    sbs: bool = True                 # simulate the backward Brillouin Stokes wave
    sbs_dt_ns: float = 0.1           # SBS time step (refined automatically for broad linewidths)
    sbs_shift_GHz: float = 10.8
    fibre_temperature_K: float = 300.0   # phonon occupation for spontaneous Brillouin scattering
    sbs_gB_m_W: float = 5e-11
    sbs_dnuB_MHz: float = 50.0
    sbs_tau_ns: float = 10.0
    damage_bulk_GW_cm2_10ns: float = 30.0
    damage_facet_GW_cm2_10ns: float = 1.0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = {k: v for k, v in dict(d).items() if k in SimConfig.__dataclass_fields__}
        for k in ("er_ase_nm", "yb_ase_nm"):
            if k in d:
                d[k] = tuple(d[k])
        return SimConfig(**d)
