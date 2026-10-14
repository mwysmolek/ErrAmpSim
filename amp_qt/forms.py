"""Forms generated from field lists: one place says label, unit, range and help for every input.

A ``Form`` edits a nested dict (dotted keys such as ``co.power_W``) and emits ``changed`` after
every edit. Rows can depend on other values (``when``) and hide when they do not apply.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QSpinBox, QVBoxLayout, QWidget)

from .widgets import NO_WHEEL


@dataclass
class F:
    key: str
    label: str
    kind: str = "float"           # float, int, bool, choice, sci, opt, text, pair
    unit: str = ""
    tip: str = ""
    lo: float = 0.0
    hi: float = 1e9
    dec: int = 3
    step: Optional[float] = None
    choices: Optional[list] = None            # [(value, label)] for "choice"
    default: Any = None                       # value an "opt" field takes when switched on
    when: Optional[Callable[[dict], bool]] = None


def get_path(d: dict, key: str):
    for part in key.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(part)
    return d


def set_path(d: dict, key: str, value) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


class _Field:
    """One input widget (or a small composite) bound to a key."""

    def __init__(self, f: F, on_change: Callable[[], None]):
        self.f = f
        self.widget = self._make(on_change)
        if f.tip:
            self.widget.setToolTip(f.tip)

    def _spin(self, f: F, on_change) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(f.lo, f.hi)
        w.setDecimals(f.dec)
        w.setSingleStep(f.step or 10 ** -min(f.dec, 2))
        w.setKeyboardTracking(False)
        if f.unit:
            w.setSuffix(f" {f.unit}")
        w.setMinimumWidth(120)
        w.installEventFilter(NO_WHEEL)
        w.setFocusPolicy(Qt.StrongFocus)
        w.valueChanged.connect(lambda _v: on_change())
        return w

    def _make(self, on_change) -> QWidget:
        f = self.f
        if f.kind == "float":
            self.spin = self._spin(f, on_change)
            return self.spin
        if f.kind == "int":
            w = QSpinBox()
            w.setRange(int(f.lo), int(min(f.hi, 2 ** 31 - 1)))
            w.setSingleStep(int(f.step or 1))
            w.setKeyboardTracking(False)
            if f.unit:
                w.setSuffix(f" {f.unit}")
            w.installEventFilter(NO_WHEEL)
            w.setFocusPolicy(Qt.StrongFocus)
            w.valueChanged.connect(lambda _v: on_change())
            self.spin = w
            return w
        if f.kind == "bool":
            self.check = QCheckBox()
            self.check.toggled.connect(lambda _b: on_change())
            return self.check
        if f.kind == "choice":
            w = QComboBox()
            for v, lab in f.choices or []:
                w.addItem(lab, v)
            w.installEventFilter(NO_WHEEL)
            w.setFocusPolicy(Qt.StrongFocus)
            w.currentIndexChanged.connect(lambda _i: on_change())
            self.combo = w
            return w
        if f.kind in ("sci", "text"):
            w = QLineEdit()
            if f.kind == "sci":
                v = QDoubleValidator()
                v.setNotation(QDoubleValidator.ScientificNotation)
                w.setValidator(v)
            w.editingFinished.connect(on_change)
            self.edit = w
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(w, 1)
            if f.unit:
                u = QLabel(f.unit)
                u.setProperty("muted", True)
                lay.addWidget(u)
            return row
        if f.kind == "opt":
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            self.check = QCheckBox()
            self.check.setToolTip("Off: the model's own estimate is used")
            self.spin = self._spin(f, on_change)
            self.check.toggled.connect(self.spin.setEnabled)
            self.check.toggled.connect(lambda _b: on_change())
            lay.addWidget(self.check)
            lay.addWidget(self.spin, 1)
            return row
        if f.kind == "pair":
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            self.a, self.b = self._spin(f, on_change), self._spin(f, on_change)
            lay.addWidget(self.a, 1)
            lay.addWidget(QLabel("to"))
            lay.addWidget(self.b, 1)
            return row
        raise ValueError(f.kind)

    def get(self):
        f = self.f
        if f.kind in ("float",):
            return float(self.spin.value())
        if f.kind == "int":
            return int(self.spin.value())
        if f.kind == "bool":
            return bool(self.check.isChecked())
        if f.kind == "choice":
            return self.combo.currentData()
        if f.kind == "sci":
            try:
                return float(self.edit.text().replace(",", "."))
            except ValueError:
                return f.default
        if f.kind == "text":
            return self.edit.text()
        if f.kind == "opt":
            return float(self.spin.value()) if self.check.isChecked() else None
        if f.kind == "pair":
            return (float(self.a.value()), float(self.b.value()))     # tuple, as in SimConfig

    def set(self, v) -> None:
        f = self.f
        if f.kind in ("float", "int"):
            if v is not None:
                self.spin.setValue(v)
        elif f.kind == "bool":
            self.check.setChecked(bool(v))
        elif f.kind == "choice":
            i = self.combo.findData(v)
            if i < 0 and v is not None:                  # a value not in the list (e.g. a custom table)
                self.combo.addItem(str(v), v)
                i = self.combo.count() - 1
            self.combo.setCurrentIndex(max(i, 0))
        elif f.kind == "sci":
            self.edit.setText("" if v is None else f"{float(v):.4g}")
        elif f.kind == "text":
            self.edit.setText("" if v is None else str(v))
        elif f.kind == "opt":
            on = v is not None
            self.check.setChecked(on)
            self.spin.setEnabled(on)
            self.spin.setValue(v if on else (f.default if f.default is not None else self.spin.value()))
        elif f.kind == "pair":
            if v:
                self.a.setValue(float(v[0]))
                self.b.setValue(float(v[1]))


class Form(QWidget):
    """Groups of fields in boxes; ``cols`` boxes side by side."""
    changed = Signal()

    def __init__(self, groups: list[tuple[str, list[F]]], cols: int = 1, parent=None):
        super().__init__(parent)
        self._loading = False
        self.fields: dict[str, _Field] = {}
        self._rows: list[tuple[QFormLayout, int, F]] = []
        self.boxes: dict[str, QGroupBox] = {}
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        for i, (title, fs) in enumerate(groups):
            box = QGroupBox(title)
            form = QFormLayout(box)
            form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            for f in fs:
                fld = _Field(f, self._edited)
                self.fields[f.key] = fld
                lab = QLabel(f.label)
                if f.tip:
                    lab.setToolTip(f.tip)
                form.addRow(lab, fld.widget)
                self._rows.append((form, form.rowCount() - 1, f))
            self.boxes[title] = box
            grid.addWidget(box, i // cols, i % cols, Qt.AlignTop)
        for c in range(cols):
            grid.setColumnStretch(c, 1)

    def _edited(self) -> None:
        if self._loading:
            return
        self.refresh_visibility()
        self.changed.emit()

    def values(self) -> dict:
        out: dict = {}
        for k, fld in self.fields.items():
            set_path(out, k, fld.get())
        return out

    def merged(self, base: dict) -> dict:
        """``base`` with this form's values written over it (keys the form does not show survive)."""
        import copy
        out = copy.deepcopy(base)
        for k, fld in self.fields.items():
            set_path(out, k, fld.get())
        return out

    def set_values(self, d: dict) -> None:
        self._loading = True
        try:
            for k, fld in self.fields.items():
                v = get_path(d, k)
                if v is not None or fld.f.kind == "opt":
                    fld.set(v)
        finally:
            self._loading = False
        self.refresh_visibility()

    def refresh_visibility(self) -> None:
        vals = self.values()
        shown: dict[int, bool] = {}
        for form, row, f in self._rows:
            vis = True
            if f.when is not None:
                try:
                    vis = bool(f.when(vals))
                except Exception:  # noqa: BLE001 - a broken condition just shows the row
                    vis = True
                form.setRowVisible(row, vis)
            shown[id(form)] = shown.get(id(form), False) or vis
        for box in self.boxes.values():                # a box with nothing to show disappears
            box.setVisible(shown.get(id(box.layout()), True))

    def set_choices(self, key: str, choices: list) -> None:
        fld = self.fields[key]
        cur = fld.get()
        self._loading = True
        try:
            fld.combo.clear()
            for v, lab in choices:
                fld.combo.addItem(lab, v)
            fld.set(cur)
        finally:
            self._loading = False


def stack(*widgets: QWidget) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        lay.addWidget(x)
    return w
