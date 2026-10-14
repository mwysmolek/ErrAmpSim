"""A project: everything one amplifier design needs, in one JSON file (``.ampstudio``).

Fibre datasheet, pump, seed and numerics; the Q-switch designer's settings; the batch
draft; any uploaded cross-section tables and seed pulse, so a project file opens the
same on another PC. A stage file used as seed is referenced by path (it can be large).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime

import numpy as np

from . import xsections as xs
from .config import PumpConfig, SeedConfig, SimConfig
from .fiber import FiberDatasheet
from .presets import DEFAULT_PRESET, preset
from .qswitch import QSwitchConfig

FILE_FORMAT = "kayei-ampstudio-project"
FILE_VERSION = 1
EXT = ".ampstudio"
OPENABLE = (EXT, ".json")


def default_batch() -> dict:
    return dict(name="length and pump sweep", axes={"length_m": "2:6:1", "pump_co_W": "0.5, 1.0"},
                objective="E_out_uJ", maximize=True, constraints={"sbs_pct": ["<", 1.0]},
                workers=0, keep_traces=True, accuracy="draft")


@dataclass
class Project:
    name: str = "Untitled amplifier"
    notes: str = ""
    fiber: dict = field(default_factory=dict)
    pump: dict = field(default_factory=dict)
    seed: dict = field(default_factory=dict)
    sim: dict = field(default_factory=dict)
    qswitch: dict = field(default_factory=lambda: QSwitchConfig().to_dict())
    batch: dict = field(default_factory=default_batch)
    custom_xs: dict = field(default_factory=dict)       # name -> {lam_nm, sa, se}
    seed_csv: dict | None = None                        # {t_ns, P, phase?, file}
    stage_file: str = ""                                # .fibamp.npz used when seed.source == "stage"
    saved_at: str = ""

    # ------------------------------------------------------------------ build
    @staticmethod
    def from_preset(name: str = DEFAULT_PRESET) -> "Project":
        p = preset(name)
        return Project(name=name, notes=p["notes"], fiber=p["fiber"], pump=p["pump"], seed=p["seed"], sim=p["sim"])

    def configs(self):
        """(FiberDatasheet, PumpConfig, SeedConfig, SimConfig), with custom tables registered."""
        self.register_xs()
        return (FiberDatasheet.from_dict(self.fiber), PumpConfig.from_dict(self.pump),
                SeedConfig.from_dict(self.seed), SimConfig.from_dict(self.sim))

    def register_xs(self) -> None:
        for name, t in (self.custom_xs or {}).items():
            xs.register(xs.CrossSection(name, np.asarray(t["lam_nm"], float) * 1e-9, np.asarray(t["sa"], float),
                                        np.asarray(t["se"], float), notes=t.get("notes", ""), source="project"))

    def add_xs_table(self, table: xs.CrossSection) -> None:
        self.custom_xs[table.name] = dict(lam_nm=(table.lam * 1e9).tolist(), sa=table.sa.tolist(),
                                          se=table.se.tolist(), notes=table.notes)
        xs.register(table)

    def seed_inputs(self):
        """(csv_data, stage) for run_amplifier, as the seed source needs them."""
        from .io import load_stage
        src = self.seed.get("source", "analytic")
        csv_data = stage = None
        if src == "csv":
            if not self.seed_csv:
                raise ValueError("The seed is set to 'CSV' but no pulse file has been loaded.")
            csv_data = {k: np.asarray(v, float) for k, v in self.seed_csv.items() if k in ("t_ns", "P", "phase")}
        elif src == "stage":
            if not self.stage_file or not os.path.exists(self.stage_file):
                raise ValueError("The seed is set to 'previous stage' but the stage file is missing: "
                                 f"{self.stage_file or '(none chosen)'}")
            stage = load_stage(self.stage_file)
        return csv_data, stage

    # ------------------------------------------------------------------- file
    def to_json(self) -> str:
        d = asdict(self)
        d.update(format=FILE_FORMAT, version=FILE_VERSION, saved_at=datetime.now().isoformat(timespec="seconds"))
        return json.dumps(d, indent=1, default=_json_default)

    @staticmethod
    def from_json(text: str) -> "Project":
        d = json.loads(text)
        if not isinstance(d, dict):
            raise ValueError("Not a project file.")
        if "fiber" in d and "format" not in d and {"pump", "seed", "sim"} <= set(d):
            d = dict(name="Imported run setup", **{k: d[k] for k in ("fiber", "pump", "seed", "sim")})
        if "fiber" not in d:
            raise ValueError("Not a kayEi AmpStudio project (no fibre in it).")
        names = {f.name for f in fields(Project)}
        p = Project(**{k: v for k, v in d.items() if k in names})
        # normalise through the dataclasses: unknown keys dropped, missing ones defaulted
        p.fiber = FiberDatasheet.from_dict(p.fiber).to_dict()
        p.pump = PumpConfig.from_dict(p.pump or {}).to_dict()
        p.seed = SeedConfig.from_dict(p.seed or {}).to_dict()
        p.sim = SimConfig.from_dict(p.sim or {}).to_dict()
        p.qswitch = QSwitchConfig.from_dict(p.qswitch or {}).to_dict()
        p.batch = {**default_batch(), **(p.batch or {})}
        p.custom_xs = dict(p.custom_xs or {})
        return p

    def save(self, path: str) -> str:
        atomic_write(path, self.to_json())
        return path

    @staticmethod
    def load(path: str) -> "Project":
        with open(path, encoding="utf-8") as fh:
            return Project.from_json(fh.read())


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def atomic_write(path: str, text: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip() or "project"


def parse_seed_csv(text: str, name: str = "") -> dict:
    """Pulse shape from CSV text: time in ns, power (any unit; scaled to the seed energy when
    the maximum is <= 1) and an optional phase in rad. A header row is optional; columns are
    picked by name (t / time, p / power, phase) when there is one, else by position."""
    rows, header = [], None
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = [x.strip() for x in ln.replace(";", ",").replace("\t", ",").split(",") if x.strip()]
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            if not rows:
                header = [h.lower() for h in parts]
    rows = [r for r in rows if len(r) >= 2]
    if len(rows) < 4:
        raise ValueError("the pulse file needs at least 4 rows of time, power")
    a = np.array([r[:3] + [0.0] * (3 - len(r[:3])) for r in rows], float)
    it, ip, iph = 0, 1, 2 if a.shape[1] > 2 and any(len(r) > 2 for r in rows) else None
    if header:
        def find(*keys):
            for i, h in enumerate(header):
                if any(h.startswith(k) for k in keys):
                    return i
            return None
        it = find("t", "time") if find("t", "time") is not None else 0
        ip = find("p", "power", "i", "intensity") if find("p", "power", "i", "intensity") is not None else 1
        iph = find("phase", "phi")
    t = a[:, it]
    if np.ptp(t) < 1e-6:                     # seconds, not ns
        t = t * 1e9
    o = np.argsort(t)
    out = dict(t_ns=t[o].tolist(), P=np.maximum(a[o, ip], 0).tolist(), file=name)
    if iph is not None and iph < a.shape[1]:
        out["phase"] = a[o, iph].tolist()
    return out
