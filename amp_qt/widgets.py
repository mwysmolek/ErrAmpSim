"""Shared widgets: KPI tiles, banners, a list-of-dicts table, small helpers."""
from __future__ import annotations

import math
from typing import Callable, Optional

from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QObject, Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox, QComboBox, QFrame, QHBoxLayout, QHeaderView,
                               QLabel, QScrollArea, QTableView, QVBoxLayout, QWidget)

from . import theme as T


class Kpi(QFrame):
    def __init__(self, title: str, tip: str = "", parent=None):
        super().__init__(parent)
        self.setProperty("card", True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        self.t = QLabel(title)
        self.t.setProperty("muted", True)
        self.v = QLabel("–")
        self.v.setProperty("kpi", True)
        self.d = QLabel("")
        self.d.setProperty("muted", True)
        self.d.setWordWrap(True)
        lay.addWidget(self.t)
        lay.addWidget(self.v)
        lay.addWidget(self.d)
        if tip:
            self.setToolTip(tip)

    def set(self, value: str, delta: str = "", color: Optional[str] = None) -> None:
        self.v.setText(value)
        self.d.setText(delta)
        self.d.setStyleSheet(f"color:{color};" if color else "")


class KpiRow(QWidget):
    def __init__(self, items: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.tiles: dict[str, Kpi] = {}
        for title, tip in items:
            k = Kpi(title, tip)
            self.tiles[title] = k
            lay.addWidget(k, 1)

    def set(self, title: str, value: str, delta: str = "", color: Optional[str] = None) -> None:
        self.tiles[title].set(value, delta, color)

    def clear(self) -> None:
        for k in self.tiles.values():
            k.set("–")


class Banner(QFrame):
    """A coloured message strip: kind = notice (amber), danger (red) or info (cyan)."""

    def __init__(self, text: str = "", kind: str = "notice", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)
        self.lab = QLabel(text)
        self.lab.setWordWrap(True)
        self.lab.setTextFormat(Qt.RichText)
        self.lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.lab, 1)
        self.set(text, kind)

    def set(self, html: str, kind: str = "notice") -> None:
        for k in ("notice", "danger", "info"):
            self.setProperty(k, k == kind)
        self.style().unpolish(self)
        self.style().polish(self)
        self.lab.setText(html)
        self.setVisible(bool(html))


class RowsModel(QAbstractTableModel):
    """Read-only view of a list of dicts (columns in the given order)."""

    def __init__(self, rows: list[dict], columns: list[str], fmt: Optional[dict] = None,
                 headers: Optional[dict] = None, colors: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self.rows, self.columns = rows, columns
        self.fmt = fmt or {}
        self.headers = headers or {}
        self.colors = colors

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            c = self.columns[section]
            return self.headers.get(c, c)
        return None

    def data(self, index, role=Qt.DisplayRole):
        row = self.rows[index.row()]
        col = self.columns[index.column()]
        v = row.get(col)
        if role == Qt.DisplayRole:
            if col in self.fmt:
                try:
                    return self.fmt[col](v)
                except (TypeError, ValueError):
                    return str(v)
            return fmt_value(v)
        if role == Qt.TextAlignmentRole and isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.ForegroundRole and self.colors:
            c = self.colors(row, col)
            if c:
                from PySide6.QtGui import QColor
                return QColor(c)
        if role == Qt.ToolTipRole and isinstance(v, str) and len(v) > 40:
            return v
        return None


def fmt_value(v, nd: int = 4) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "–"
        if math.isinf(v):
            return "∞"
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1e5 or a < 1e-3:
            return f"{v:.3e}"
        return f"{v:.{nd}g}"
    return str(v)


class Table(QTableView):
    rowActivated = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doubleClicked.connect(self._activated)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def _activated(self, index) -> None:
        m = self.model()
        if isinstance(m, RowsModel) and 0 <= index.row() < len(m.rows):
            self.rowActivated.emit(m.rows[index.row()])

    def set_rows(self, rows: list[dict], columns: Optional[list[str]] = None, fmt: Optional[dict] = None,
                 headers: Optional[dict] = None, colors: Optional[Callable] = None) -> None:
        if columns is None:
            columns = []
            for r in rows:
                for k in r:
                    if k not in columns:
                        columns.append(k)
        self.setModel(RowsModel(rows, columns, fmt, headers, colors, self))
        self.resizeColumnsToContents()
        for i in range(len(columns)):
            self.setColumnWidth(i, min(max(self.columnWidth(i) + 12, 60), 420))

    def set_kv(self, d: dict) -> None:
        """A {quantity: value} dict as a two-column table, the value column always visible."""
        self.set_rows(kv_rows(d), ["quantity", "value"])
        h = self.horizontalHeader()
        h.setStretchLastSection(False)
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)

    def selected_row(self) -> Optional[dict]:
        m = self.model()
        idx = self.selectionModel().selectedRows() if self.selectionModel() else []
        if isinstance(m, RowsModel) and idx:
            return m.rows[idx[0].row()]
        return None


def kv_rows(d: dict) -> list[dict]:
    """A {name: value} dict as two-column table rows."""
    return [dict(quantity=k, value=v) for k, v in d.items()]


def legend_below(fig, height: int):
    """Move a horizontal legend under the plot (it collides with subplot titles on top)."""
    fig.update_layout(legend=dict(orientation="h", x=0, y=-0.16, yanchor="top"), height=height,
                      margin=dict(b=130))
    return fig


def section(title: str) -> QLabel:
    lab = QLabel(title)
    lab.setProperty("h2", True)
    return lab


def muted(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("muted", True)
    lab.setWordWrap(True)
    return lab


def scroll(widget: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setWidget(widget)
    return sa


class NoWheel(QObject):
    """Spin boxes and combo boxes ignore the mouse wheel unless they have focus, so scrolling a
    long form never changes a value by accident."""

    def eventFilter(self, obj, ev):  # noqa: N802
        if ev.type() == QEvent.Wheel and isinstance(obj, (QAbstractSpinBox, QComboBox)) and not obj.hasFocus():
            ev.ignore()
            return True
        return False


NO_WHEEL = NoWheel()


def fmt_energy_uJ(v: float) -> str:
    if v is None or not math.isfinite(v):
        return "–"
    if v >= 1000:
        return f"{v/1000:.3g} mJ"
    if v >= 1:
        return f"{v:.3g} µJ"
    return f"{v*1000:.3g} nJ"


def fmt_power_kW(v: float) -> str:
    if v is None or not math.isfinite(v):
        return "–"
    if v >= 1000:
        return f"{v/1000:.3g} MW"
    if v >= 1:
        return f"{v:.3g} kW"
    return f"{v*1000:.3g} W"


GOOD, WARN, BAD = T.GOOD, T.WARN, T.BAD
