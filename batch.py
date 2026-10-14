"""Batch runs: parameter sweeps over the amplifier model.

A batch is a base configuration (fibre datasheet, pump, seed, numerics) plus a set
of sweep axes. Each axis is written as ``start:stop:step`` (stop included), a
comma list ``a, b, c``, or a single value. The cases are the Cartesian product of
all axes. Every case runs in its own worker process (``python -m fibamp.batch
--worker``), so a crash or a hang in one case cannot take the batch down, and the
batch keeps going on all CPU cores.

Results are stored under ``~/fibamp/batches/<name>/``: ``batch.json`` (setup),
``results.jsonl`` (one line per case, written as cases finish, so an interrupted
batch keeps what it has) and the reports written by :mod:`fibamp.report`.

Command line (no browser needed)::

    python -m fibamp.batch my_batch.json          # a file written by save_spec / the app
"""
from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

BATCH_DIR = os.path.join(os.path.expanduser("~"), "fibamp", "batches")

# axis key -> (label, unit, where it goes)
AXES = {
    "fibre":            ("fibre (saved library entry)", "", "fiber"),
    "length_m":         ("fibre length", "m", "fiber.length_m"),
    "pair_fraction":    ("Er ions in pairs", "fraction", "fiber.pair_fraction"),
    "bend_radius_cm":   ("coil radius", "cm", "fiber.bend_radius_cm"),
    "seed_fwhm_ns":     ("pulse width (FWHM)", "ns", "seed.fwhm_ns"),
    "seed_energy_uJ":   ("seed energy", "µJ", "seed.energy_uJ"),
    "seed_nm":          ("signal wavelength", "nm", "seed.wavelength_nm"),
    "seed_linewidth_nm": ("seed linewidth", "nm", "seed.linewidth_nm"),
    "rep_kHz":          ("repetition rate", "kHz", "seed.rep_rate_kHz"),
    "pump_co_W":        ("co-pump power", "W", "pump.co.power_W"),
    "pump_counter_W":   ("counter-pump power", "W", "pump.counter.power_W"),
    "pump_nm":          ("pump wavelength", "nm", "pump.*.wavelength_nm"),
}

# metrics written for every case: key -> (summary/budget key, label, unit)
METRICS = {
    "E_out_uJ":       ("output pulse energy (uJ)", "output pulse energy", "µJ"),
    "gain_dB":        ("energy gain (dB)", "energy gain", "dB"),
    "peak_kW":        ("output peak power (kW)", "peak power", "kW"),
    "fwhm_out_ns":    ("output FWHM (ns)", "output FWHM", "ns"),
    "P_avg_W":        ("average signal power out (W)", "average signal power", "W"),
    "pulse_frac_pct": ("pulse fraction of output energy (%)", "pulse share of output", "%"),
    "ase_fwd_W":      ("forward ASE, average (W)", "forward ASE", "W"),
    "ase_bwd_W":      ("backward ASE at input, average (W)", "backward ASE", "W"),
    "extraction_pct": ("extraction of stored energy (%)", "extraction of stored energy", "%"),
    "g0L_dB":         ("small-signal gain before pulse, g0L (dB)", "small-signal gain g0L", "dB"),
    "pump_abs_pct":   ("pump absorbed (%)", "pump absorbed", "%"),
    "eff_launched_pct": ("optical efficiency vs launched pump (%)", "efficiency vs launched pump", "%"),
    "B_rad":          ("B-integral (rad)", "B-integral", "rad"),
    "raman_pct":      ("Raman conversion (%)", "Raman conversion", "%"),
    "sbs_pct":        ("SBS reflected fraction (%)", "SBS reflected", "%"),
    "sbs_G_eff":      ("SBS gain exponent with linewidth + transient relief (nepers)", "SBS gain exponent", "nepers"),
    "I_peak_GWcm2":   ("peak intensity (GW/cm2)", "peak intensity", "GW/cm²"),
    "lw_out_nm":      ("output linewidth rms (nm)", "output linewidth (rms)", "nm"),
    "lasing_W":       ("parasitic lasing, average (W)", "parasitic lasing", "W"),
    "dT_K":           ("max core temperature rise (K)", "core temperature rise", "K"),
}


# ------------------------------------------------------------------ parsing
def parse_axis(text):
    """'a:b:step' (inclusive), 'a, b, c' or a single value -> list of floats (or strings for names)."""
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        return list(text)
    s = str(text).strip()
    if not s:
        return []
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        if len(parts) == 2:
            a, b = parts
            step = (b - a) / 4 if b != a else 1.0
        elif len(parts) == 3:
            a, b, step = parts
        else:
            raise ValueError(f"'{s}': use start:stop:step")
        if step == 0 or (b - a) / step < 0:
            raise ValueError(f"'{s}': the step must move from start towards stop")
        n = int(math.floor((b - a) / step + 1e-9)) + 1
        if n > 1000:
            raise ValueError(f"'{s}' gives {n} values; use a coarser step")
        return [round(a + i * step, 10) for i in range(n)]
    out = []
    for p in s.replace(";", ",").split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            out.append(p)
    return out


@dataclass
class BatchSpec:
    name: str
    fiber: dict                       # FiberDatasheet.to_dict()
    pump: dict
    seed: dict
    sim: dict
    axes: dict = field(default_factory=dict)          # axis key -> list of values
    objective: str = "E_out_uJ"
    maximize: bool = True
    constraints: dict = field(default_factory=dict)   # metric key -> (op, value), e.g. {"sbs_pct": ["<", 1.0]}
    keep_traces: bool = True
    custom_xs: dict = field(default_factory=dict)     # cross-section tables not shipped with fibamp
    fibres: dict = field(default_factory=dict)        # name -> datasheet dict for the 'fibre' axis
    created: str = ""

    def to_dict(self):
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d):
        return BatchSpec(**{k: v for k, v in d.items() if k in BatchSpec.__dataclass_fields__})

    def n_cases(self):
        return int(np.prod([max(1, len(v)) for v in self.axes.values()])) if self.axes else 1

    def cases(self):
        keys = [k for k, v in self.axes.items() if len(v)]
        for i, combo in enumerate(itertools.product(*[self.axes[k] for k in keys])):
            yield i, dict(zip(keys, combo))


# ---------------------------------------------------------------- one case
def _apply(spec: BatchSpec, values: dict):
    """Configs for one case: base config with the swept values substituted."""
    import copy
    fib = copy.deepcopy(spec.fiber)
    pump = copy.deepcopy(spec.pump)
    seed = copy.deepcopy(spec.seed)
    if "fibre" in values:
        name = values["fibre"]
        if name not in spec.fibres:
            raise ValueError(f"fibre '{name}' is not in the batch's fibre list")
        keep_len = fib.get("length_m")
        fib = copy.deepcopy(spec.fibres[name])
        fib["length_m"] = keep_len
    for k, v in values.items():
        if k == "fibre":
            continue
        where = AXES[k][2]
        if where.startswith("fiber."):
            fib[where[6:]] = v
        elif where.startswith("seed."):
            seed[where[5:]] = v
        elif where == "pump.*.wavelength_nm":
            pump["co"]["wavelength_nm"] = v
            pump["counter"]["wavelength_nm"] = v
        elif where.startswith("pump."):
            side, key = where[5:].split(".")
            pump[side][key] = v
    return fib, pump, seed


def _trace_pulse(res, n=400):
    t = res.t
    P = np.abs(res.A_out) ** 2
    Pi = np.abs(res.A_in) ** 2
    k = np.where(np.maximum(P, Pi) > 1e-3 * max(P.max(), Pi.max(), 1e-300))[0]
    if k.size < 2:
        return None
    a, b = k[0], k[-1]
    w = b - a
    a, b = max(0, a - w // 4), min(t.size - 1, b + w // 4)
    idx = np.linspace(a, b, n).astype(int)
    return dict(t_ns=(t[idx] * 1e9).round(4).tolist(), P_out_W=P[idx].round(6).tolist(),
                P_in_W=Pi[idx].round(6).tolist())


def _trace_spectrum(res, n=600, rbw=0.5):
    g, tot, _ = res.osa(rbw)
    if g.size < 2:
        return None
    idx = np.linspace(0, g.size - 1, min(n, g.size)).astype(int)
    dbm = 10 * np.log10(np.maximum(tot[idx], 1e-30)) + 30
    return dict(lam_nm=g[idx].round(3).tolist(), dBm_per_rbw=dbm.round(3).tolist(), rbw_nm=rbw)


def run_case(job: dict) -> dict:
    """Run one case (in the worker process). job = {spec fields..., values, index}."""
    from . import xsections as xs
    from .amplifier import run_amplifier
    from .config import PumpConfig, SeedConfig, SimConfig
    from .fiber import FiberDatasheet
    for name, tab in (job.get("custom_xs") or {}).items():
        xs.register(xs.CrossSection(name, np.asarray(tab["lam"]), np.asarray(tab["sa"]), np.asarray(tab["se"]),
                                    notes="batch"))
    spec = BatchSpec.from_dict(job["spec"])
    values = job["values"]
    t0 = time.time()
    fib, pump, seed = _apply(spec, values)
    res = run_amplifier(FiberDatasheet.from_dict(fib), PumpConfig.from_dict(pump), SeedConfig.from_dict(seed),
                        SimConfig.from_dict(spec.sim))
    s, b = res.summary, res.budget
    out = dict(index=job["index"], values=values, ok=True)
    m = {}
    for key, (src, _, _) in METRICS.items():
        v = s.get(src, b.get(src))
        m[key] = float(v) if isinstance(v, (int, float, np.floating)) and v == v else None
    m["converged"] = bool(res.diag["conv"]["converged"])
    m["photon_balance_pct"] = float(res.diag["photon_balance_error"]) * 100
    m["sbs_runaway"] = bool(res.sbs and res.sbs.get("runaway"))
    m["N_er_m3"] = float(res.fiber.N_er)
    m["peak_abs_1530_dB_per_m"] = float(-res.fiber.small_signal_coeffs(1530e-9, 0.0) * 4.342944819)
    out["metrics"] = m
    out["warnings"] = [w[:200] for w in res.warnings]
    if spec.keep_traces:
        out["pulse"] = _trace_pulse(res)
        out["spectrum"] = _trace_spectrum(res)
    out["seconds"] = round(time.time() - t0, 2)
    return out


def _worker_main():
    job = json.loads(sys.stdin.read())
    try:
        out = run_case(job)
    except Exception as e:                       # noqa: BLE001 -- report, never crash the batch
        import traceback
        out = dict(index=job["index"], values=job["values"], ok=False, error=f"{type(e).__name__}: {e}",
                   trace=traceback.format_exc()[-1500:])
    sys.stdout.write("RESULT" + json.dumps(out) + "\n")


# --------------------------------------------------------------- the batch
def custom_tables(spec: BatchSpec):
    """Cross-section tables that only live in this process (uploaded), shipped to the workers."""
    from . import xsections as xs
    names = {spec.fiber.get("er_table"), spec.fiber.get("yb_table")}
    for f in spec.fibres.values():
        names |= {f.get("er_table"), f.get("yb_table")}
    out = {}
    for n in names:
        if n and not xs.LIBRARY.get(n):
            try:
                t = xs.library(n)
            except KeyError:
                continue
            out[n] = dict(lam=t.lam.tolist(), sa=t.sa.tolist(), se=t.se.tolist())
    return out


def batch_dir(name):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name).strip("_") or "batch"
    return os.path.join(BATCH_DIR, safe)


def save_spec(spec: BatchSpec, folder=None) -> str:
    folder = folder or batch_dir(spec.name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "batch.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec.to_dict(), fh, indent=2)
    return path


def load_batch(folder):
    with open(os.path.join(folder, "batch.json"), encoding="utf-8") as fh:
        spec = BatchSpec.from_dict(json.load(fh))
    rows = []
    p = os.path.join(folder, "results.jsonl")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    rows.sort(key=lambda r: r["index"])
    return spec, rows


def list_batches():
    if not os.path.isdir(BATCH_DIR):
        return []
    out = []
    for d in sorted(os.listdir(BATCH_DIR)):
        if os.path.exists(os.path.join(BATCH_DIR, d, "batch.json")):
            out.append(d)
    return out


def run_batch(spec: BatchSpec, workers=None, progress=None, folder=None, timeout_s=3600, resume=True):
    """Run every case; returns the list of result rows (also written to results.jsonl as they finish)."""
    folder = folder or batch_dir(spec.name)
    os.makedirs(folder, exist_ok=True)
    spec.custom_xs = custom_tables(spec)
    if not spec.created:
        spec.created = time.strftime("%Y-%m-%d %H:%M")
    save_spec(spec, folder)
    path = os.path.join(folder, "results.jsonl")
    done = {}
    if resume and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for l in fh:
                if l.strip():
                    r = json.loads(l)
                    if r.get("ok"):
                        done[r["index"]] = r
    todo = [(i, v) for i, v in spec.cases() if i not in done]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=pkg_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sd = spec.to_dict()

    def one(i, values):
        job = dict(spec=sd, values=values, index=i, custom_xs=spec.custom_xs)
        try:
            p = subprocess.run([sys.executable, "-m", "fibamp.batch", "--worker"], input=json.dumps(job),
                               capture_output=True, text=True, timeout=timeout_s, env=env, cwd=pkg_root)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
            if line:
                return json.loads(line[0][6:])
            return dict(index=i, values=values, ok=False, error=(p.stderr.strip().splitlines() or ["worker died"])[-1])
        except subprocess.TimeoutExpired:
            return dict(index=i, values=values, ok=False, error=f"timed out after {timeout_s:.0f} s")

    rows = list(done.values())
    n_all = spec.n_cases()
    t0 = time.time()
    mode = "a" if (resume and done) else "w"
    with open(path, mode, encoding="utf-8") as fh, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, i, v) for i, v in todo]
        for k, f in enumerate(as_completed(futs)):
            r = f.result()
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if progress:
                n_done = len(rows)
                el = time.time() - t0
                eta = el / (k + 1) * (len(todo) - k - 1)
                progress(n_done / max(n_all, 1), f"{n_done}/{n_all} cases, {el/60:.1f} min elapsed, "
                                                  f"~{eta/60:.1f} min left")
    rows.sort(key=lambda r: r["index"])
    return rows


# ------------------------------------------------------------ evaluation
def passes(row, constraints):
    """True when a finished case meets every constraint (and converged)."""
    if not row.get("ok"):
        return False
    m = row["metrics"]
    if not m.get("converged") or m.get("sbs_runaway"):
        return False
    for key, (op, val) in (constraints or {}).items():
        v = m.get(key)
        if v is None:
            return False
        if op == "<" and not v < val:
            return False
        if op == ">" and not v > val:
            return False
    return True


def ranked(spec: BatchSpec, rows):
    ok = [r for r in rows if passes(r, spec.constraints) and r["metrics"].get(spec.objective) is not None]
    return sorted(ok, key=lambda r: r["metrics"][spec.objective], reverse=spec.maximize)


def table(rows, spec: BatchSpec):
    """Flat records (axis values + metrics + status) for DataFrames / CSV."""
    out = []
    for r in rows:
        rec = {k: r["values"].get(k) for k in spec.axes if spec.axes[k]}
        if r.get("ok"):
            rec.update({k: v for k, v in r["metrics"].items()})
            rec["meets constraints"] = passes(r, spec.constraints)
            rec["warnings"] = len(r.get("warnings", []))
        else:
            rec["error"] = r.get("error")
            rec["meets constraints"] = False
        rec["seconds"] = r.get("seconds")
        out.append(rec)
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--worker":
        _worker_main()
        return
    if not argv:
        print(__doc__)
        return
    with open(argv[0], encoding="utf-8") as fh:
        spec = BatchSpec.from_dict(json.load(fh))
    folder = batch_dir(spec.name)
    print(f"batch '{spec.name}': {spec.n_cases()} cases -> {folder}")
    rows = run_batch(spec, progress=lambda f, m: print(m, flush=True), folder=folder)
    from .report import write_reports
    paths = write_reports(spec, rows, folder)
    print("reports:", *paths.values(), sep="\n  ")


if __name__ == "__main__":
    main()
