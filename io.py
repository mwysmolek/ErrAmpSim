"""File formats.

Stage file (``.fibamp.npz``): the complete output of one stage -- complex field
on its time grid, carrier wavelength, repetition rate, forward ASE spectrum and
the run configuration -- so it can seed the next stage, or be loaded back for
plotting. JSON configuration files hold a fibre datasheet or a full run setup.
"""
from __future__ import annotations

import io
import json
import os

import numpy as np

from .config import PumpConfig, SeedConfig, SimConfig
from .fiber import FiberDatasheet

FIBRE_DIR = os.path.join(os.path.expanduser("~"), "fibamp", "fibres")
RUN_DIR = os.path.join(os.path.expanduser("~"), "fibamp", "runs")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def stage_bytes(stage: dict) -> bytes:
    buf = io.BytesIO()
    meta = {k: v for k, v in stage.items() if k not in ("t", "A", "ase_lam", "ase_psd")}
    np.savez_compressed(buf, t=stage["t"], A_re=np.real(stage["A"]), A_im=np.imag(stage["A"]),
                        ase_lam=np.asarray(stage.get("ase_lam", np.zeros(0))),
                        ase_psd=np.asarray(stage.get("ase_psd", np.zeros(0))),
                        meta=np.array(json.dumps(meta, default=_json_default)))
    return buf.getvalue()


def load_stage(src) -> dict:
    """Load a stage file from a path or bytes."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    d = np.load(src, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    st = dict(meta)
    st["t"] = d["t"]
    st["A"] = d["A_re"] + 1j * d["A_im"]
    st["ase_lam"] = d["ase_lam"] if d["ase_lam"].size else None
    st["ase_psd"] = d["ase_psd"] if d["ase_psd"].size else None
    return st


def save_stage(stage: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(stage_bytes(stage))
    return path


def run_config_json(ds: FiberDatasheet, pump: PumpConfig, seed: SeedConfig, sim: SimConfig) -> str:
    return json.dumps(dict(fiber=ds.to_dict(), pump=pump.to_dict(), seed=seed.to_dict(),
                           sim=sim.to_dict()), indent=2, default=_json_default)


def parse_run_config(text: str):
    d = json.loads(text)
    return (FiberDatasheet.from_dict(d["fiber"]), PumpConfig.from_dict(d["pump"]),
            SeedConfig.from_dict(d["seed"]), SimConfig.from_dict(d["sim"]))


# ---------------------------------------------------------------- fibre library
def list_fibres(folder=FIBRE_DIR):
    if not os.path.isdir(folder):
        return []
    return sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".json"))


def save_fibre(ds: FiberDatasheet, folder=FIBRE_DIR) -> str:
    os.makedirs(folder, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_. " else "_" for c in ds.name).strip() or "fibre"
    path = os.path.join(folder, safe + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ds.to_dict(), fh, indent=2, default=_json_default)
    return path


def load_fibre(name, folder=FIBRE_DIR) -> FiberDatasheet:
    with open(os.path.join(folder, name + ".json"), encoding="utf-8") as fh:
        return FiberDatasheet.from_dict(json.load(fh))


# ---------------------------------------------------------------- tables
def pulse_csv(res) -> str:
    t = res.t * 1e9
    lines = ["t_ns,P_in_W,P_out_W,phase_out_rad,P_raman_W"]
    Pin, Pout = np.abs(res.A_in) ** 2, np.abs(res.A_out) ** 2
    ph = np.unwrap(np.angle(res.A_out))
    PR = np.atleast_2d(res.P_raman).sum(axis=0) if np.size(res.P_raman) else np.zeros(t.size)
    step = max(1, t.size // 20000)
    for i in range(0, t.size, step):
        lines.append(f"{t[i]:.6f},{Pin[i]:.6e},{Pout[i]:.6e},{ph[i]:.6e},{PR[i]:.6e}")
    return "\n".join(lines) + "\n"


def spectrum_csv(res, rbw_nm=0.1) -> str:
    sp = res.spectra()
    lines = ["component,lambda_nm,psd_W_per_nm"]
    for k, (l, s) in sp.items():
        step = max(1, len(l) // 20000)
        lines += [f"{k},{x:.5f},{y:.6e}" for x, y in zip(l[::step], s[::step])]
    for end in ("output", "input"):
        g, tot, _ = res.osa(rbw_nm, end=end)
        step = max(1, g.size // 20000)
        lines += [f"osa_{end}_rbw{rbw_nm:g}nm,{x:.5f},{y / rbw_nm:.6e}" for x, y in zip(g[::step], tot[::step])]
    return "\n".join(lines) + "\n"


def profile_csv(res) -> str:
    lines = ["z_m,inversion_pre,inversion_post,pulse_energy_J,peak_power_W,raman_energy_J,heat_W_per_m,dT_core_K"]
    zc = res.z_cells
    ER = np.atleast_2d(res.ER_z).sum(axis=0) if np.size(res.ER_z) else np.zeros(res.z_nodes.size)
    pre, post = res.nE_pre, res.nE_post
    heat, dT = res.thermal.get("heat_W_m", np.zeros(zc.size)), res.thermal.get("dT_core", np.zeros(zc.size))
    for i, z in enumerate(res.z_nodes):
        j = min(i, zc.size - 1)
        lines.append(f"{z:.5f},{pre[j]:.5f},{post[j]:.5f},{res.E_z[i]:.6e},"
                     f"{res.Ppk_z[i]:.6e},{ER[i]:.6e},{heat[j]:.5e},{dT[j]:.4f}")
    return "\n".join(lines) + "\n"


def report_json(res) -> str:
    return json.dumps(dict(summary=res.summary, budget=res.budget, warnings=res.warnings,
                           notes=res.notes, fibre=res.fiber.summary(res.seed.lam0), config=res.config),
                      indent=2, default=_json_default)
