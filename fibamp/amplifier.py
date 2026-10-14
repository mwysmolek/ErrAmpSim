"""Amplifier stage: couples the inter-pulse and intra-pulse solvers and assembles results."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .config import PumpConfig, SeedConfig, SimConfig
from .fiber import Fiber, FiberDatasheet, DB
from .inversion import SlowModel, Measurement, State, er_excited, yb_excited
from .pulse import fn_deplete, propagate_pulse, pulse_diagnostics
from .sbs import simulate_sbs
from .seed import build_seed, SeedField, spectrum
from .xsections import H, C0

VERSION = "2.0"


@dataclass
class AmpResult:
    fiber: Fiber
    seed: SeedField
    t: np.ndarray
    A_in: np.ndarray
    A_out: np.ndarray
    P_raman: np.ndarray          # (orders, nt) Stokes power per order (band model), summed over modes
    lam_raman: np.ndarray        # (orders,)
    z_nodes: np.ndarray
    z_cells: np.ndarray
    st_pre: State
    st_post: State
    E_z: np.ndarray
    Ppk_z: np.ndarray
    ER_z: np.ndarray             # (orders, nz+1)
    bands: object
    avg: dict
    trace: dict
    diag: dict
    budget: dict
    summary: dict
    warnings: list
    notes: list
    config: dict
    pulse: dict = field(default_factory=dict)     # mode fields, snapshots, per-mode energies
    sbs: dict | None = None
    thermal: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)

    # convenience views (area-weighted over rings)
    @property
    def nE_pre(self):
        return er_excited(self.fiber, self.st_pre)

    @property
    def nE_post(self):
        return er_excited(self.fiber, self.st_post)

    @property
    def nY_pre(self):
        return yb_excited(self.fiber, self.st_pre)

    # --------------------------------------------------------------- spectra
    def spectra(self, n_raman=400):
        """Average power spectral densities (W per nm) of every component."""
        f_rep = self.seed.f_rep
        lam_s, S_s = spectrum(self.t, self.pulse["A_modes"], self.seed.lam0)
        lam_i, S_i = spectrum(self.t, self.A_in, self.seed.lam0)
        b = self.bands
        ase = b.kind == "ase"
        fw, bw = ase & (b.direction > 0), ase & (b.direction < 0)
        P_out = self.avg["P_out"]
        out = dict(
            signal_out=(lam_s * 1e9, S_s * f_rep * 1e-9),
            signal_in=(lam_i * 1e9, S_i * f_rep * 1e-9),
            ase_fwd=(b.lam[fw] * 1e9, P_out[fw] / (b.dlam[fw] * 1e9)),
            ase_bwd=(b.lam[bw] * 1e9, P_out[bw] / (b.dlam[bw] * 1e9)),
        )
        dtt = self.t[1] - self.t[0]
        G = max(self.diag.get("G_R", 1.0), 1.0)
        width = self.config["sim"]["raman_width_THz"] * 1e12 / np.sqrt(G)
        lams, pr = [], []
        for k, lr in enumerate(np.atleast_1d(self.lam_raman)):
            E_R = float(np.sum(self.P_raman[k]) * dtt)
            nuR = C0 / lr
            nu = nuR + np.linspace(-3, 3, n_raman) * width
            lamR = C0 / nu
            prof = np.exp(-4 * np.log(2) * ((nu - nuR) / width) ** 2)
            prof_lam = prof * nu ** 2 / C0
            prof_lam /= abs(np.trapezoid(prof_lam, lamR)) + 1e-300
            o = np.argsort(lamR)
            lams.append(lamR[o] * 1e9)
            pr.append(prof_lam[o] * E_R * f_rep * 1e-9)
        if lams:
            out["raman"] = (np.concatenate(lams), np.concatenate(pr))
        else:
            out["raman"] = (np.zeros(0), np.zeros(0))
        if self.sbs is not None and np.isfinite(self.sbs["E"]) and self.sbs["E"] > 0:
            lb = self.sbs["lam"] * 1e9
            wdt = max(self.diag.get("lw_out_fwhm", 0) * 1e9, 1e-4)
            l = lb + np.linspace(-3, 3, 61) * wdt
            p = np.exp(-4 * np.log(2) * ((l - lb) / wdt) ** 2)
            p /= np.trapezoid(p, l)
            out["sbs_bwd"] = (l, p * self.sbs["E"] * f_rep)
        else:
            out["sbs_bwd"] = (np.zeros(0), np.zeros(0))
        return out

    def osa(self, rbw_nm=0.1, span=None, end="output"):
        """What an optical spectrum analyser with resolution rbw_nm shows (W per rbw).

        Every component is binned onto one grid and the total is convolved with a Gaussian
        filter of FWHM rbw_nm. end = 'output' (signal + forward ASE + Raman) or 'input'
        (backward ASE + SBS Stokes leaving the input end)."""
        sp = self.spectra()
        comps = ("signal_out", "ase_fwd", "raman") if end == "output" else ("ase_bwd", "sbs_bwd")
        pk = max([float(np.max(sp[k][1])) for k in comps if len(sp[k][1])] + [1e-300])
        live = [k for k in comps if len(sp[k][1]) and float(np.max(sp[k][1])) > 1e-9 * pk]
        lo = min([float(sp[k][0][sp[k][1] > 1e-9 * pk].min()) for k in live] + [1480.0])
        hi = max([float(sp[k][0][sp[k][1] > 1e-9 * pk].max()) for k in live] + [1640.0])
        if span:
            lo, hi = span
        step = max(rbw_nm / 6, (hi - lo) / 60000)
        grid = np.arange(lo, hi + step, step)
        edges = np.concatenate([[grid[0] - step / 2], grid + step / 2])
        parts = {}
        for k in comps:
            l, psd = sp[k]
            if len(l) < 2:
                continue
            o = np.argsort(l)
            l, psd = l[o], psd[o]
            # integrate the density exactly onto the grid bins
            cum = np.concatenate([[0.0], np.cumsum(0.5 * (psd[1:] + psd[:-1]) * np.diff(l))])
            if k.startswith("ase"):
                # piecewise-constant bands: integrate the band density
                db = np.gradient(l)
                le = np.concatenate([[l[0] - db[0] / 2], l + db / 2])
                cum = np.concatenate([[0.0], np.cumsum(psd * db)])
                l = le
            Pbin = np.diff(np.interp(edges, l, cum, left=0.0, right=cum[-1]))
            parts[k] = Pbin
        tot = np.sum(list(parts.values()), axis=0) if parts else np.zeros(grid.size)
        sig = rbw_nm / 2.3548 / step
        half = int(max(3, 4 * sig))
        kern = np.exp(-0.5 * (np.arange(-half, half + 1) / max(sig, 1e-9)) ** 2)
        kern /= kern.sum()
        conv = {k: np.convolve(v, kern, mode="same") * (rbw_nm / step) for k, v in parts.items()}
        return grid, np.convolve(tot, kern, mode="same") * (rbw_nm / step), conv

    # ----------------------------------------------------------------- export
    def to_stage(self, name="stage"):
        """Everything the next stage needs, as a plain dict (see io.save_stage)."""
        b = self.bands
        fw = (b.kind == "ase") & (b.direction > 0)
        return dict(name=name, t=self.t, A=self.A_out, lam0=self.seed.lam0, f_rep=self.seed.f_rep,
                    ase_lam=b.lam[fw], ase_psd=self.avg["P_out"][fw] / b.dlam[fw],
                    E_raman=self.diag["E_raman"], lam_raman=np.atleast_1d(self.lam_raman).tolist(),
                    summary=self.summary, config=self.config, version=VERSION)


def _progress_wrap(progress, lo, hi):
    if progress is None:
        return None
    return lambda frac, msg: progress(lo + (hi - lo) * min(max(frac, 0.0), 1.0), msg)


def _ledger(fiber, model, meas, avg, st_pre, st_post, dz, f_rep, ions_raman=0.0):
    """Er and Yb excitation ledgers over the measured period: (removed / created) - 1."""
    b = model.b
    hnuE = H * C0 / 1535e-9
    hnu_b = H * b.nu
    phE = avg["ion_pop_E"] / hnu_b
    _, lase_ph = meas.lasing_power()
    tr = avg["transfer_rate"] if fiber.has_yb else 0.0
    pp, ase = b.is_pump, b.kind == "ase"
    A = fiber.A_r
    sig = float(((st_pre.n - st_post.n) @ A * fiber.N_s + (st_pre.x - st_post.x) @ A * fiber.N_p).sum()) * dz * f_rep
    upE = -phE[pp].sum() + tr
    downE = sig + phE[ase].sum() + avg["decay_er"] / hnuE + avg["upc"] / hnuE + lase_ph["Er"]
    bal = float(downE / max(upE, 1e-300) - 1)
    bal_y = None
    if fiber.has_yb:
        phY = avg["ion_Y"] / hnu_b
        upY = -phY[pp].sum()
        downY = phY[ase].sum() + avg["decay_yb"] / (H * C0 / 975e-9) + tr + avg["transfer_wasted_rate"] + lase_ph["Yb"]
        bal_y = float(downY / max(upY, 1e-300) - 1)
    return bal, bal_y, sig


def run_amplifier(ds: FiberDatasheet, pump: PumpConfig, seed_cfg: SeedConfig, sim: SimConfig,
                  csv_data=None, stage=None, progress=None) -> AmpResult:
    t0 = time.time()
    fiber = Fiber(ds)
    seed = build_seed(seed_cfg, sim, csv_data=csv_data, stage=stage)
    notes = list(fiber.notes) + list(seed.notes)
    warnings = list(fiber.warnings)
    lam_s = seed.lam0
    E_seed = seed.energy
    input_spec = None
    if seed.ase_lam is not None:
        input_spec = (seed.ase_lam, seed.ase_psd)
    model = SlowModel(fiber, pump, sim, lam_s, seed.f_rep, input_spec=input_spec,
                      cw_lines=seed.meta.get("cw_lines", []))
    notes += model.notes
    dz = model.dz
    # The periodic iteration uses a cheap energy-only depletion (fn_deplete) corrected towards the full
    # pulse solver by an additive offset on the post-pulse state. (A multiplicative correction of the
    # depletion is not a contraction when a strong pulse bleaches the fibre and oscillates period-2.)
    off_n = np.zeros((sim.nz, fiber.nr))
    off_x = np.zeros((sim.nz, fiber.nr))

    def deplete(st):
        post, _ = fn_deplete(fiber, st, E_seed, lam_s, dz)
        return State(np.clip(post.n + off_n, 0, 1), np.clip(post.x + off_x, 0, 1), st.y)

    n_outer = 1 + max(0, int(sim.consistency_iters))
    st = None
    conv = {}
    res = None
    span = 0.85 / n_outer
    refinements = 0
    for refine in range(3):
        if refine:
            # near parasitic-lasing threshold the inversion ODE is stiff: tighten the step and re-solve
            model.sim = replace(sim, cfl=sim.cfl / 3 ** refine,
                                max_steps_per_period=sim.max_steps_per_period * 5 ** refine)
            refinements = refine
        # pulse / inversion consistency: repeat until the full solver's correction stops changing
        k, max_pass = 0, (n_outer + 3 if refine == 0 else 1)
        while True:
            lo = min(k, n_outer - 1) * span
            st, conv = model.solve_periodic(deplete, st, _progress_wrap(progress, lo, lo + 0.45 * span),
                                            label=f"inversion (pass {k+1}"
                                                  + (f", refinement {refine}" if refine else "") + ")")
            res = propagate_pulse(fiber, seed, st, sim, _progress_wrap(progress, lo + 0.45 * span, lo + span))
            post_fn, _ = fn_deplete(fiber, st, E_seed, lam_s, dz)
            new_n, new_x = res["st_post"].n - post_fn.n, res["st_post"].x - post_fn.x
            d_off = float(max(np.max(np.abs(new_n - off_n)), np.max(np.abs(new_x - off_x))))
            off_n, off_x = new_n, new_x
            k += 1
            if k >= max_pass or (k >= n_outer and d_off < 5 * sim.tol) or not conv.get("converged", False):
                break
        if d_off >= 5 * sim.tol and refine == 0:
            notes.append(f"pulse/inversion consistency reached {d_off:.1e} (inversion units) after {k} passes; "
                         "the pulse result varies from pass to pass (noise-driven nonlinearity)")
        if k > 1 and d_off >= 5 * sim.tol and conv.get("converged", False):
            # final periodic solve with the last correction so the measured period is periodic
            st, conv = model.solve_periodic(deplete, st, None, label="inversion (final)")
        # resolved measurement period at the converged state
        if progress:
            progress(0.88, "measuring one period")
        meas = Measurement(model)
        post = deplete(st)
        st_end, _, steps = model.evolve(post, 0.0, model.T, measure=meas)
        avg = meas.averages()
        bal_q, _, _ = _ledger(fiber, model, meas, avg, st, res["st_post"], dz, seed.f_rep)
        if abs(bal_q) <= 0.005 or not conv.get("converged", False) or refine >= 1:
            break
    if refinements:
        notes.append(f"inversion time step refined {refinements}x automatically to close the photon balance")
    model.sim = sim
    if not conv.get("converged", False):
        if conv.get("stagnated"):
            why = ("the fibre is at the parasitic-lasing threshold, where it self-pulses" if model.lasing_seen
                   else "possible period doubling or relaxation oscillation")
            warnings.insert(0, f"NO PERIODIC STEADY STATE after {conv['iterations']} periods: the inversion does not "
                               f"settle from pulse to pulse ({why}). The numbers are for one representative period and "
                               "change from pulse to pulse; a real amplifier here would be unstable.")
        else:
            warnings.append(f"periodic steady state not reached in {sim.max_periods} periods "
                            "(possible period doubling or relaxation oscillation); results are for the last period")

    # ------------------------------------------------------------ SBS
    sbs = None
    if sim.sbs:
        if progress:
            progress(0.92, "Brillouin Stokes wave")
        mid = State(0.5 * (st.n + res["st_post"].n), 0.5 * (st.x + res["st_post"].x), st.y)
        gz = np.array([float(fiber.signal_modes(lam_s)["G"][0] @ g) for g in _ring_gain(fiber, mid, lam_s)])
        sbs = simulate_sbs(fiber, res, sim, gz, lam_s)

    diag = pulse_diagnostics(res, seed, fiber, sim, sbs=sbs)
    for k_ in ("G_R", "B", "nsub_max"):
        diag[k_] = res[k_]
    diag["conv"] = conv
    diag["max_round_trip_gain"] = meas.max_RT
    diag["steps_per_period"] = steps
    diag["nl_model"] = res["nl_model"]
    warnings += diag["warnings"]
    if meas.max_RT > 0.5:
        warnings.append(f"end-face reflections close a parasitic cavity: max round-trip ASE gain "
                        f"{meas.max_RT:.2f} (lasing at 1). Improve the angle cleave / end caps.")

    # ------------------------------------------------------------ budget
    b = model.b
    f_rep = seed.f_rep
    E_out = diag["out_stats"]["E"]
    pp = b.is_pump
    ase = b.kind == "ase"
    fw, bw = ase & (b.direction > 0), ase & (b.direction < 0)
    P_launch = float(avg["P_inj"][pp].sum())
    P_res = float(avg["P_out"][pp].sum())
    P_abs = float(-avg["ion_x"][pp].sum())
    P_bg_pump = float(avg["bg"][pp].sum())
    P_ase_f = float(avg["P_out"][fw].sum())
    P_ase_b = float(avg["P_out"][bw].sum())
    P_ase_in = float(avg["P_inj"][fw].sum())
    P_sig = (E_out - E_seed) * f_rep
    P_R = diag["E_raman"] * f_rep
    P_sbs = sbs["E"] * f_rep if (sbs and np.isfinite(sbs["E"])) else 0.0
    P_guided_sp = float(avg["spont_in"][ase].sum())
    P_spont = avg["decay_er"] + avg["decay_yb"] - P_guided_sp
    P_quench_esa = float(-avg["ion_loss_E"].sum())
    lase_P, lase_ph = meas.lasing_power()
    P_lase = float(sum(lase_P.values()))
    P_heat = P_abs - (P_sig + P_R + P_sbs + (P_ase_f + P_ase_b - P_ase_in) + float(avg["bg"][ase].sum())
                      + P_spont + avg["upc"] + P_lase + P_quench_esa)
    if P_lase > 0:
        top = max(lase_P.items(), key=lambda kv: kv[1])
        warnings.append(f"PARASITIC LASING / AT THRESHOLD between the end faces: {P_lase:.3g} W average removed at "
                        f"the lasing threshold, strongest at {top[0][1]:.0f} nm ({top[0][0]}); the gain is clamped, "
                        "which caps the stored energy. Close to threshold, how this power splits between parasitic "
                        f"lasing and ASE (together {P_lase + P_ase_f + P_ase_b:.3g} W) depends on the time step: "
                        "lower the inversion time-step factor to resolve it; the pulse results change by ~1 % only. "
                        "Reduce end reflections (angle cleave, end caps), shorten the fibre, lower the hold-off "
                        "gain or pulse the pump closer to the seed.")
    bal, bal_y, sig_ions = _ledger(fiber, model, meas, avg, st, res["st_post"], dz, f_rep)
    diag["photon_balance_error"] = bal
    if bal_y is not None:
        diag["photon_balance_error_yb"] = bal_y
    if abs(bal) > 0.01 and conv.get("converged", False):
        warnings.append(f"Er photon-number balance closes only to {bal*100:+.1f} %: reduce the "
                        "time-step factor (cfl) or increase nz")

    # ------------------------------------------------------------ thermal
    ext_z = (((st.n - res["st_post"].n) * fiber.N_s + (st.x - res["st_post"].x) * fiber.N_p) @ fiber.A_r
             ) * H * C0 / lam_s * f_rep                           # W/m extracted by the pulse
    ext_z = ext_z - res["ions_quenched_z"] / dz * H * C0 / 1535e-9 * f_rep   # post-pulse pair quenching is heat
    heat_z = avg["heat_z"] - ext_z - avg["lasing_z"] / dz
    dT = heat_z * fiber.thermal_resistance()
    thermal = dict(z=model.z_cells, heat_W_m=heat_z, dT_core=dT)
    if dT.max() > 80:
        warnings.append(f"core temperature rise up to {dT.max():.0f} K (heat {heat_z.max():.2f} W/m): coating damage "
                        "risk and cross sections shift; heat-sink the fibre or spread the pump")
    elif dT.max() > 30:
        notes.append(f"core runs up to {dT.max():.0f} K above ambient; set the fibre temperature for the "
                     "cross sections accordingly")

    ntr = float(fiber.transparency(lam_s))
    xtr = min(1.0, 2 * ntr)
    stored = float((np.maximum(st.n - ntr, 0) * fiber.N_s + np.maximum(st.x - xtr, 0) * fiber.N_p
                    ).sum(axis=0) @ fiber.A_r) * dz * H * C0 / lam_s
    g0L = float(sum(fiber.signal_modes(lam_s)["G"][0] @ g for g in _ring_gain(fiber, st, lam_s)) * dz * DB)
    budget = {
        "pump launched (W)": P_launch,
        "pump residual (W)": P_res,
        "pump absorbed by ions (W)": P_abs,
        "pump background loss (W)": P_bg_pump,
        "signal power added (W)": P_sig,
        "Raman Stokes (W)": P_R,
        "SBS Stokes, backward (W)": P_sbs,
        "forward ASE out (W)": P_ase_f,
        "  of which input spectrum passed through (W, input)": P_ase_in,
        "backward ASE out at input end (W)": P_ase_b,
        "parasitic lasing (W)": P_lase,
        "spontaneous decay, sideways (W)": P_spont,
        "Er upconversion loss (W)": avg["upc"],
        "Er pair quenching + ESA loss (W)": P_quench_esa,
        "heat: quantum defect + transfer + remainder (W)": P_heat,
    }
    tot_out_energy = E_out + (P_ase_f / f_rep) + diag["E_raman"]
    Em = res["Em_z"][-1]
    summary = {
        "seed energy (uJ)": E_seed * 1e6,
        "output pulse energy (uJ)": E_out * 1e6,
        "energy added by the stage (uJ)": (E_out - E_seed) * 1e6,
        "energy gain (dB)": 10 * np.log10(max(E_out, 1e-300) / max(E_seed, 1e-300)),
        "output peak power (kW)": diag["out_stats"]["P_peak"] / 1e3,
        "output FWHM (ns)": diag["out_stats"]["fwhm"] * 1e9,
        "input FWHM (ns)": diag["in_stats"]["fwhm"] * 1e9,
        "output peak time vs input centre (ns)": diag["out_stats"]["t_peak"] * 1e9,
        "average signal power out (W)": E_out * f_rep,
        "pulse fraction of output energy (%)": E_out / max(tot_out_energy, 1e-300) * 100,
        "forward ASE, average (W)": P_ase_f,
        "backward ASE at input, average (W)": P_ase_b,
        "stored energy before pulse, above transparency (uJ)": stored * 1e6,
        "extraction of stored energy (%)": (E_out - E_seed) / max(stored, 1e-300) * 100,
        "small-signal gain before pulse, g0L (dB)": g0L,
        "pump absorbed (%)": P_abs / max(P_launch, 1e-300) * 100,
        "optical efficiency vs launched pump (%)": P_sig / max(P_launch, 1e-300) * 100,
        "optical efficiency vs absorbed pump (%)": P_sig / max(P_abs, 1e-300) * 100,
        "B-integral (rad)": diag["B"],
        "nonlinear model": res["nl_model"],
        "Raman gain exponent G (nepers)": diag["G_R"],
        "Raman conversion (%)": diag["raman_frac"] * 100,
        "peak intensity (GW/cm2)": diag["I_peak"] / 1e13,
        "indicative bulk damage (GW/cm2)": diag["I_bulk"] / 1e13,
        "indicative bare-facet damage (GW/cm2)": diag["I_facet"] / 1e13,
        "SBS threshold estimate, analytic (kW)": diag["P_sbs"] / 1e3,
        "input linewidth rms (nm)": diag["lw_in_rms"] * 1e9,
        "output linewidth rms (nm)": diag["lw_out_rms"] * 1e9,
        "max parasitic round-trip gain": meas.max_RT,
        "parasitic lasing, average (W)": P_lase,
        "max core temperature rise (K)": float(dT.max()),
        "periods to steady state": conv.get("iterations", 0),
    }
    if sbs is not None:
        summary["SBS steady-state gain exponent, max (nepers)"] = sbs["G_B"]
        summary["SBS gain exponent with linewidth + transient relief (nepers)"] = sbs["G_eff"]
        if sbs["runaway"]:
            summary["SBS reflected energy (uJ)"] = float("nan")
            warnings.insert(0, f"SBS RUNAWAY: the Brillouin gain exponent reaches {sbs['G_eff']:.0f} nepers after "
                               "linewidth and transient relief (threshold ~21). A pulse like this would be reflected / "
                               "broken up by SBS; the forward result "
                               "below is not physical. Broaden the seed linewidth (phase modulation, >~1 GHz), "
                               "shorten the fibre or lower the peak power.")
        else:
            frac = sbs["E"] / max(E_out, 1e-300)
            summary["SBS reflected energy (uJ)"] = sbs["E"] * 1e6
            summary["SBS reflected fraction (%)"] = frac * 100
            summary["pump depletion by SBS (%)"] = sbs["pump_depletion"] * 100
            if frac > 0.5:
                warnings.insert(0, f"SBS-LIMITED: the backward Brillouin Stokes carries {frac*100:.0f} % of the output "
                                   "energy; it is amplified by the Er gain as well, and beyond this point it would "
                                   "deplete the inversion and break up the pulse. The forward result is not physical. "
                                   "Broaden the seed linewidth, shorten the fibre or lower the pulse energy.")
            elif frac > 0.01:
                summary["output energy after SBS depletion, estimate (uJ)"] = E_out * (1 - sbs["pump_depletion"]) * 1e6
                warnings.append(f"SBS: {frac*100:.1f} % of the output energy is reflected backwards (gain exponent "
                                f"{sbs['G_B']:.0f}); the pulse loses ~{sbs['pump_depletion']*100:.0f} %. The forward "
                                "solver does not feed this back: treat its output as an upper bound. Broaden the seed "
                                "linewidth or shorten the fibre.")
            elif frac > 1e-4:
                notes.append(f"SBS reflects {frac*100:.2g} % of the output energy backwards (below threshold)")
        if sbs.get("res_limited"):
            notes.append("SBS time step could not resolve the full seed linewidth: SBS is overestimated")
    if len(Em) > 1:
        names = res["mode_names"]
        for nm_, e in zip(names, Em):
            summary[f"output energy in {nm_} (%)"] = e / max(Em.sum(), 1e-300) * 100
    if fiber.pair_k == 0 and fiber.N_er > 1.5e25:
        notes.append(f"N_Er = {fiber.N_er:.2g} m^-3 is heavily doped; real fibres at this level usually have "
                     "5-20 % of ions in clusters. Set 'Er ions in pairs' (sidebar) if results look optimistic.")
    if diag["I_peak"] > diag["I_facet"]:
        notes.append("peak intensity exceeds the bare-facet damage level: an end cap is required")
    if g0L > 40:
        notes.append(f"small-signal gain before the pulse is {g0L:.0f} dB; the ASE and reflection "
                     "treatment in this model is what limits it, check the backward-ASE and round-trip values")
    cfg = dict(fiber=ds.to_dict(), pump=pump.to_dict(), seed=seed_cfg.to_dict(), sim=sim.to_dict(),
               version=VERSION)
    if progress:
        progress(1.0, "done")
    pulse = {k: res[k] for k in ("A_modes", "mode_names", "Em_z", "snap_P", "snap_S", "snap_t", "snap_lam", "snap_z")}
    return AmpResult(fiber=fiber, seed=seed, t=seed.t, A_in=res["A_in"], A_out=res["A_out"],
                     P_raman=res["PR"], lam_raman=res["lamR"], z_nodes=model.z_nodes,
                     z_cells=model.z_cells, st_pre=st, st_post=res["st_post"],
                     E_z=res["Ez"], Ppk_z=res["Ppk_z"], ER_z=res["ERz"], bands=b, avg=avg,
                     trace={k: np.asarray(v) for k, v in meas.trace.items()}, diag=diag,
                     budget=budget, summary=summary, warnings=warnings, notes=notes, config=cfg,
                     pulse=pulse, sbs=sbs, thermal=thermal, timing=dict(total_s=time.time() - t0))


def _ring_gain(fiber, st: State, lam):
    """Per-cell ring gain density (nz, nr) at wavelength lam, Er + Yb (1/m per unit overlap)."""
    saE, seE = (float(v) for v in fiber.xs_er_at(lam))
    esa = float(fiber.esa_at(lam))
    g = (fiber.N_s * (seE * st.n - saE * (1 - st.n)) + fiber.N_p * (seE * st.x - 2 * saE * (1 - st.x))
         - fiber.N_p * saE * st.x - esa * (fiber.N_s * st.n + fiber.N_p * st.x))
    if fiber.has_yb:
        saY, seY = (float(v) for v in fiber.xs_yb_at(lam))
        g = g + fiber.N_yb * (seY * st.y - saY * (1 - st.y))
    return g
