"""The desktop app, headless: pages, forms, a run end to end."""
from __future__ import annotations

import copy
import time

import pytest

pytest.importorskip("PySide6.QtWebEngineWidgets")

from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fibamp.presets import PRESETS, numerics  # noqa: E402
from fibamp.project import Project  # noqa: E402

FAST = "C-band pre-amplifier (Er80-4/125 class)"


@pytest.fixture(scope="module")
def win():
    from amp_qt import theme
    from amp_qt.document import Document
    from amp_qt.mainwindow import MainWindow
    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app)
    w = MainWindow(Document(), smoke=True)
    w.show()
    yield w
    w.runner.wait(5000)
    w.close()


def _wait(cond, timeout=180.0):
    t0 = time.time()
    while not cond():
        QCoreApplication.processEvents()
        time.sleep(0.02)
        if time.time() - t0 > timeout:
            raise TimeoutError
    for _ in range(20):
        QCoreApplication.processEvents()


@pytest.mark.parametrize("name", list(PRESETS))
def test_forms_do_not_change_a_project_on_their_own(win, name):
    """Loading a preset into the pages and reading it back must give the same project (no unit or
    rounding drift through the widgets)."""
    p = Project.from_preset(name)
    win.doc.replace(copy.deepcopy(p), None)
    QCoreApplication.processEvents()
    fib = win.pages["fibre"]
    got = fib.form.merged(win.doc.project.fiber)
    got["absorption"] = fib.absorb.values()
    assert got == p.fiber
    launch = win.pages["launch"]
    assert launch.pump.merged(p.pump) == p.pump
    assert launch.seed.merged(p.seed) == p.seed
    assert win.pages["numerics"].form.merged(p.sim) == p.sim
    assert win.pages["qswitch"].form.merged(p.qswitch) == p.qswitch


def test_simulate_shows_results(win):
    p = Project.from_preset(FAST)
    p.sim = numerics("draft").to_dict()
    win.doc.replace(p, None)
    win.simulate()
    _wait(lambda: not win.runner.is_busy())
    res = win.pages["results"].res
    assert res is not None, win.last_error
    assert res.summary["output pulse energy (uJ)"] > 0
    assert win.pages["results"].kpis.tiles["Output pulse energy"].v.text() != "–"
    # an edit after the run marks the results stale
    win.doc.update("pump", dict(win.doc.project.pump, co=dict(win.doc.project.pump["co"], power_W=0.31)))
    assert win.pages["results"].stale.isVisible() or not win.pages["results"].isVisible()


def test_next_stage_from_output(win, tmp_path):
    from fibamp import io as fio
    res = win.pages["results"].res
    assert res is not None
    path = fio.save_stage(res.to_stage("stage 1"), str(tmp_path / "s1.fibamp.npz"))
    win.new_stage_project(path)
    p = win.doc.project
    assert p.seed["source"] == "stage" and p.stage_file == path and p.name.endswith("next stage")
    win.simulate()
    _wait(lambda: not win.runner.is_busy())
    r2 = win.pages["results"].res
    assert r2 is not res, win.last_error
    assert r2.summary["seed energy (uJ)"] > 0.5 * res.summary["output pulse energy (uJ)"]


def test_stop_cancels_a_run(win):
    p = Project.from_preset(FAST)
    p.sim = numerics("standard").to_dict()
    win.doc.replace(p, None)
    win.simulate()
    _wait(lambda: win.runner.is_busy(), 10)
    win.runner.stop()
    _wait(lambda: not win.runner.is_busy(), 60)
    assert win.last_error == ""


def test_accuracy_switch(win):
    win.doc.replace(Project.from_preset(FAST), None)
    panel = win.panel
    panel.acc.setCurrentIndex(panel.acc.findData("draft"))
    assert win.doc.project.sim["nz"] == numerics("draft").nz
    from amp_qt.pages.numerics_page import level_of
    assert level_of(win.doc.project.sim) == "draft"


def test_save_and_reopen(win, tmp_path):
    win.doc.replace(Project.from_preset(FAST), None)
    win.doc.update("name", "saved one")
    path = win.doc.save(str(tmp_path / "a.ampstudio"))
    win.doc.replace(Project.from_preset(next(iter(PRESETS))), None)
    win.doc.open(path)
    assert win.doc.project.name == "saved one" and not win.doc.dirty


def test_batch_page_builds_a_spec(win):
    win.doc.replace(Project.from_preset(FAST), None)
    spec = win.pages["batch"].spec_for_run()
    assert spec.n_cases() == 10 and "length_m" in spec.axes
