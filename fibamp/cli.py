"""Headless commands (no window): ``python -m fibamp.cli`` or ``kayei_ampstudio.exe <command>``.

    run PROJECT [--accuracy draft|standard|fine] [--json OUT] [--stage OUT.fibamp.npz] [--csv DIR]
    qswitch PROJECT [--json OUT] [--stage OUT.fibamp.npz]
    batch PROJECT_OR_SPEC [--workers N] [--accuracy draft|standard|fine] [--axis KEY=VALUES ...]
    presets                          list the starting points
    new PRESET OUT.ampstudio         a project file from a preset

PROJECT is a ``.ampstudio`` project, or a run-setup JSON written by the original fibamp app.
The packaged exe is windowed, so console output lands in the log file in the data folder;
use the file outputs (--json, --stage, --csv) in scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

COMMANDS = ("run", "qswitch", "batch", "presets", "new")


def _project(path):
    from .project import Project
    return Project.load(path)


def _sim(p, accuracy):
    from .config import SimConfig
    from .presets import numerics
    base = SimConfig.from_dict(p.sim)
    return numerics(accuracy, base) if accuracy else base


def cmd_run(a) -> int:
    from . import io as fio
    from .amplifier import run_amplifier
    p = _project(a.project)
    ds, pump, seed, _ = p.configs()
    csv_data, stage = p.seed_inputs()
    res = run_amplifier(ds, pump, seed, _sim(p, a.accuracy), csv_data=csv_data, stage=stage,
                        progress=lambda f, m: print(f"{f*100:5.1f} %  {m}", flush=True) if a.verbose else None)
    for k, v in res.summary.items():
        print(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")
    for w in res.warnings:
        print("WARNING:", w)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            fh.write(fio.report_json(res))
    if a.stage:
        fio.save_stage(res.to_stage(p.name), a.stage)
    if a.csv:
        os.makedirs(a.csv, exist_ok=True)
        for name, text in (("pulse.csv", fio.pulse_csv(res)), ("spectrum.csv", fio.spectrum_csv(res)),
                           ("profile.csv", fio.profile_csv(res))):
            with open(os.path.join(a.csv, name), "w", encoding="utf-8") as fh:
                fh.write(text)
    return 0


def cmd_qswitch(a) -> int:
    from . import io as fio
    from .config import PumpConfig
    from .fiber import FiberDatasheet
    from .qswitch import QSwitchConfig, QSwitchModel
    p = _project(a.project)
    p.register_xs()
    m = QSwitchModel(FiberDatasheet.from_dict(p.fiber), PumpConfig.from_dict(p.pump),
                     QSwitchConfig.from_dict(p.qswitch), _sim(p, a.accuracy))
    res = m.solve()
    for k, v in res["summary"].items():
        print(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(summary=res["summary"], warnings=res["warnings"], notes=res["notes"]), fh, indent=2)
    if a.stage and res["pulse"].get("ok"):
        fio.save_stage(m.export_seed(res), a.stage)
    return 0


def batch_spec(p, accuracy=None):
    """BatchSpec from a project's batch draft (axes as text, as typed in the app)."""
    from .batch import AXES, BatchSpec, parse_axis
    from .io import list_fibres, load_fibre
    b = p.batch
    axes = {}
    for k, text in (b.get("axes") or {}).items():
        if k not in AXES:
            raise ValueError(f"unknown sweep axis '{k}'")
        vals = parse_axis(text)
        if vals:
            axes[k] = vals
    fibres = {}
    if "fibre" in axes:
        have = set(list_fibres())
        for n in axes["fibre"]:
            if str(n) not in have:
                raise ValueError(f"fibre '{n}' is not in the fibre library")
            fibres[str(n)] = load_fibre(str(n)).to_dict()
        axes["fibre"] = [str(n) for n in axes["fibre"]]
    sim = _sim(p, accuracy or b.get("accuracy") or None).to_dict()
    cons = {k: list(v) for k, v in (b.get("constraints") or {}).items()}
    return BatchSpec(name=b.get("name") or p.name, fiber=p.fiber, pump=p.pump, seed=p.seed, sim=sim, axes=axes,
                     objective=b.get("objective", "E_out_uJ"), maximize=bool(b.get("maximize", True)),
                     constraints=cons, keep_traces=bool(b.get("keep_traces", True)), fibres=fibres)


def cmd_batch(a) -> int:
    from .batch import BatchSpec, batch_dir, run_batch
    from .report import write_reports
    with open(a.spec, encoding="utf-8") as fh:
        d = json.load(fh)
    if "axes" in d and "fiber" in d and "batch" not in d:
        spec = BatchSpec.from_dict(d)
    else:
        p = _project(a.spec)
        p.register_xs()
        if a.axis:                                   # --axis length_m=2:6:1 replaces the project's axes
            p.batch = dict(p.batch, axes=dict(x.split("=", 1) for x in a.axis))
        spec = batch_spec(p, a.accuracy)
    folder = batch_dir(spec.name)
    print(f"batch '{spec.name}': {spec.n_cases()} cases -> {folder}", flush=True)
    rows = run_batch(spec, workers=a.workers or None, folder=folder,
                     progress=lambda f, m: print(m, flush=True))
    paths = write_reports(spec, rows, folder)
    print("reports:", *paths.values(), sep="\n  ")
    n_ok = sum(1 for r in rows if r.get("ok"))
    print(f"{n_ok}/{len(rows)} cases finished")
    return 0 if n_ok == len(rows) else 2


def cmd_presets(_a) -> int:
    from .presets import PRESETS
    for k, v in PRESETS.items():
        print(f"{k}\n    {v['notes']}")
    return 0


def cmd_new(a) -> int:
    from .presets import PRESETS
    from .project import Project
    names = [k for k in PRESETS if k.lower().startswith(a.preset.lower())] or \
            [k for k in PRESETS if a.preset.lower() in k.lower()]
    if not names:
        print(f"no preset matches '{a.preset}'; see 'presets'")
        return 2
    Project.from_preset(names[0]).save(a.out)
    print(f"{names[0]} -> {a.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kayei_ampstudio", description="kayEi AmpStudio, headless")
    sp = ap.add_subparsers(dest="cmd", required=True)
    acc = dict(choices=("draft", "standard", "fine"), default=None, help="override the project's numerics")
    r = sp.add_parser("run", help="simulate the amplifier stage of a project")
    r.add_argument("project")
    r.add_argument("--accuracy", **acc)
    r.add_argument("--json")
    r.add_argument("--stage")
    r.add_argument("--csv")
    r.add_argument("-v", "--verbose", action="store_true")
    q = sp.add_parser("qswitch", help="design the project's Q-switched oscillator")
    q.add_argument("project")
    q.add_argument("--accuracy", **acc)
    q.add_argument("--json")
    q.add_argument("--stage")
    b = sp.add_parser("batch", help="run a project's batch sweep (or a batch.json)")
    b.add_argument("spec")
    b.add_argument("--workers", type=int, default=0)
    b.add_argument("--accuracy", **acc)
    b.add_argument("--axis", action="append", default=[], metavar="KEY=VALUES",
                   help="sweep axis, e.g. length_m=2:6:1 or pump_co_W=0.5,1 (repeatable; replaces the project's axes)")
    sp.add_parser("presets", help="list the starting points")
    n = sp.add_parser("new", help="write a project file from a preset")
    n.add_argument("preset")
    n.add_argument("out")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "qswitch": cmd_qswitch, "batch": cmd_batch, "presets": cmd_presets,
            "new": cmd_new}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
