"""Batches: worker processes, file protocol, stop / resume, the project's batch draft."""
from __future__ import annotations

import threading

from fibamp.batch import load_batch, run_batch
from fibamp.cli import batch_spec
from fibamp.project import Project

FAST = "C-band pre-amplifier (Er80-4/125 class)"


def _spec(name, axes):
    p = Project.from_preset(FAST)
    p.batch = dict(p.batch, name=name, axes=axes, accuracy="draft", keep_traces=False)
    return batch_spec(p)


def test_batch_runs_cases_in_workers(tmp_path):
    spec = _spec("two cases", {"length_m": "1.5, 2.5"})
    rows = run_batch(spec, workers=2, folder=str(tmp_path))
    assert [r["ok"] for r in rows] == [True, True], rows
    assert rows[0]["metrics"]["E_out_uJ"] > 0
    spec2, rows2 = load_batch(str(tmp_path))
    assert spec2.name == spec.name and len(rows2) == 2
    assert not list(tmp_path.glob("jobs-*"))                 # job files cleaned up


def test_stopped_batch_keeps_nothing_half_done_and_resumes(tmp_path):
    spec = _spec("stop", {"length_m": "1, 1.5, 2, 2.5"})
    stop = threading.Event()
    stop.set()                                               # stopped before anything started
    rows = run_batch(spec, workers=1, folder=str(tmp_path), stop=stop)
    assert rows == []
    rows = run_batch(spec, workers=2, folder=str(tmp_path))
    assert len(rows) == 4 and all(r["ok"] for r in rows)


def test_broken_case_is_reported_not_fatal(tmp_path):
    spec = _spec("broken", {"length_m": "1.5"})
    spec.fiber = dict(spec.fiber, d_doped_um=99.0)           # doped region larger than the core: ValueError
    rows = run_batch(spec, workers=1, folder=str(tmp_path))
    assert rows[0]["ok"] is False and "doped diameter" in rows[0]["error"]


def test_fibre_axis_needs_library_entries():
    import pytest
    with pytest.raises(ValueError, match="fibre library"):
        _spec("lib", {"fibre": "not a fibre"})
