"""The computations the window starts (run off the UI thread by :class:`runner.Runner`).

Each takes a frozen snapshot (project JSON), so editing while a run is going cannot change it.
"""
from __future__ import annotations

from fibamp.project import Project


def simulate(project_json: str, progress=None, stop=None):
    from fibamp.amplifier import run_amplifier
    p = Project.from_json(project_json)
    ds, pump, seed, sim = p.configs()
    csv_data, stage = p.seed_inputs()
    return run_amplifier(ds, pump, seed, sim, csv_data=csv_data, stage=stage, progress=progress)


def qswitch(project_json: str, progress=None, stop=None):
    from fibamp.config import PumpConfig, SimConfig
    from fibamp.fiber import FiberDatasheet
    from fibamp.qswitch import QSwitchConfig, QSwitchModel
    p = Project.from_json(project_json)
    p.register_xs()
    m = QSwitchModel(FiberDatasheet.from_dict(p.fiber), PumpConfig.from_dict(p.pump),
                     QSwitchConfig.from_dict(p.qswitch), SimConfig.from_dict(p.sim))
    res = m.solve(progress=progress)
    g_per_m = res["g0"] / max(m.fiber.L, 1e-9)
    res["map"] = m.design_map(g_per_m, L_range=(0.1, max(3.0, 2 * m.fiber.L)))
    res["stage"] = m.export_seed(res) if res["pulse"].get("ok") else None
    res["L"] = m.fiber.L
    return res


def batch(spec_dict: dict, workers: int = 0, progress=None, stop=None):
    from fibamp.batch import BatchSpec, batch_dir, run_batch
    spec = BatchSpec.from_dict(spec_dict)
    folder = batch_dir(spec.name)

    def report(f, m):                     # Stop must not raise here: finished cases are kept
        if progress is not None:
            try:
                progress(f, m)
            except Exception:  # noqa: BLE001
                pass

    rows = run_batch(spec, workers=workers or None, folder=folder, stop=stop, progress=report)
    return dict(spec=spec, rows=rows, folder=folder, stopped=bool(stop and stop.is_set()))
