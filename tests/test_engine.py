"""The engine as the app uses it: presets, projects, file formats, the CLI."""
from __future__ import annotations

import json

import numpy as np
import pytest

from fibamp.amplifier import run_amplifier
from fibamp.config import SimConfig
from fibamp.fiber import Fiber, FiberDatasheet
from fibamp.presets import NUMERICS, PRESETS, numerics
from fibamp.project import Project, parse_seed_csv

FAST = "C-band pre-amplifier (Er80-4/125 class)"


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_are_real_fibres(name):
    """Every starting point solves to a doping a real fibre could have (the bare defaults do not)."""
    f = Fiber(PRESETS[name]["fiber"])
    assert 1e24 < f.N_er < 1.2e26
    assert not f.warnings


def test_bare_defaults_are_not_a_design():
    """Why presets exist: the dataclass defaults put an Er:Yb absorption on an Er fibre."""
    assert Fiber(FiberDatasheet()).N_er > 1e26


@pytest.fixture(scope="module")
def fast_result():
    p = Project.from_preset(FAST)
    ds, pump, seed, _ = p.configs()
    return run_amplifier(ds, pump, seed, numerics("draft"))


def test_fast_preset_amplifies(fast_result):
    s = fast_result.summary
    assert 15 < s["energy gain (dB)"] < 40
    assert fast_result.diag["conv"]["converged"]
    assert abs(fast_result.diag["photon_balance_error"]) < 0.01


def test_numerics_levels_keep_physics_switches():
    base = SimConfig(sbs=False, raman=False)
    for lvl in NUMERICS:
        s = numerics(lvl, base)
        assert s.sbs is False and s.raman is False
    assert numerics("draft").nz < numerics("standard").nz < numerics("fine").nz


def test_project_round_trip(tmp_path):
    p = Project.from_preset(FAST)
    p.notes = "ünïcode ✓"
    path = p.save(str(tmp_path / "x.ampstudio"))
    q = Project.load(path)
    assert q.fiber == p.fiber and q.pump == p.pump and q.seed == p.seed and q.sim == p.sim
    assert q.notes == p.notes


def test_old_run_setup_imports():
    """A run-setup JSON written by the original fibamp app opens as a project."""
    from fibamp.io import run_config_json
    ds, pump, seed, sim = Project.from_preset(FAST).configs()
    p = Project.from_json(run_config_json(ds, pump, seed, sim))
    assert p.fiber["name"] == ds.name and p.seed["wavelength_nm"] == seed.wavelength_nm


def test_unknown_keys_are_dropped():
    d = json.loads(Project.from_preset(FAST).to_json())
    d["fiber"]["from_the_future"] = 1
    d["sim"]["nope"] = 2
    p = Project.from_json(json.dumps(d))
    assert "from_the_future" not in p.fiber and "nope" not in p.sim


def test_custom_cross_sections_travel_with_the_project(tmp_path):
    from fibamp import xsections as xs
    t = xs.library("Er (aluminosilicate)")
    mine = xs.CrossSection("vendor Er test", t.lam.copy(), t.sa * 1.1, t.se * 1.1)
    p = Project.from_preset(FAST)
    p.add_xs_table(mine)
    p.fiber["er_table"] = mine.name
    text = p.to_json()
    xs._CACHE.pop(mine.name, None)
    xs.LIBRARY.pop(mine.name, None)
    q = Project.from_json(text)
    ds, *_ = q.configs()
    assert Fiber(ds).xs_er.name == mine.name


def test_seed_csv_parsing():
    text = "t_ns,P_W,phase\n" + "\n".join(f"{t},{np.exp(-t*t/20):.6f},0" for t in range(-10, 11))
    d = parse_seed_csv(text, "pulse.csv")
    assert len(d["t_ns"]) == 21 and max(d["P"]) == pytest.approx(1.0) and "phase" in d
    d2 = parse_seed_csv("\n".join(f"{t*1e-9},{1+t}" for t in range(5)))      # seconds, no header
    assert d2["t_ns"][-1] == pytest.approx(4.0)


def test_csv_seed_runs():
    p = Project.from_preset(FAST)
    t = np.linspace(-30, 30, 301)
    p.seed_csv = dict(t_ns=t.tolist(), P=np.exp(-t ** 2 / 50).tolist())
    p.seed["source"] = "csv"
    from fibamp.seed import build_seed
    csv_data, stage = p.seed_inputs()
    s = build_seed(*[c for c in p.configs()[2:]], csv_data=csv_data)
    assert s.energy == pytest.approx(p.seed["energy_uJ"] * 1e-6, rel=1e-3)     # normalised CSV -> energy


def test_stage_chain(tmp_path, fast_result):
    from fibamp import io as fio
    path = fio.save_stage(fast_result.to_stage("one"), str(tmp_path / "s.fibamp.npz"))
    p = Project.from_preset(FAST)
    p.seed["source"], p.stage_file = "stage", path
    csv_data, stage = p.seed_inputs()
    assert stage["name"] == "one"
    ds, pump, seed, _ = p.configs()
    r2 = run_amplifier(ds, pump, seed, numerics("draft"), stage=stage)
    assert r2.summary["seed energy (uJ)"] == pytest.approx(fast_result.summary["output pulse energy (uJ)"], rel=0.02)


def test_cli_new_and_run(tmp_path):
    from fibamp.cli import main
    proj = str(tmp_path / "p.ampstudio")
    assert main(["new", "C-band", proj]) == 0
    out = tmp_path / "out.json"
    assert main(["run", proj, "--accuracy", "draft", "--json", str(out), "--csv", str(tmp_path / "csv")]) == 0
    rep = json.loads(out.read_text())
    assert rep["summary"]["output pulse energy (uJ)"] > 0
    assert (tmp_path / "csv" / "pulse.csv").exists()
