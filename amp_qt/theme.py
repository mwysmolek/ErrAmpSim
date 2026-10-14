"""Look of the desktop app: the kayEi navy palette as a Qt style sheet (shared with RiskStudio and TwinStudio)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPalette
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

BG = "#001a45"
BG2 = "#002259"
PANEL = "#0b2a66"
LINE = "#1e3a6e"
INK = "#e6edf7"
MUTED = "#9fb3c8"
ACCENT = "#22d3ee"
ACCENT_INK = "#002238"
GOOD, WARN, BAD = "#22c55e", "#f59e0b", "#ef4444"
CHECK = (ASSETS / "check.png").as_posix()

QSS = f"""
* {{ font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif; font-size: 13px; }}
QMainWindow, QDialog, QWidget {{ background: {BG}; color: {INK}; }}
QToolTip {{ background: {PANEL}; color: {INK}; border: 1px solid {LINE}; padding: 4px 6px; }}
QMenuBar {{ background: {BG2}; border-bottom: 1px solid {LINE}; }}
QMenuBar::item:selected {{ background: rgba(34,211,238,.18); border-radius: 4px; }}
QMenu {{ background: {PANEL}; border: 1px solid {LINE}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: rgba(34,211,238,.18); }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 4px 6px; }}
QToolBar {{ background: {BG2}; border: 0; border-bottom: 1px solid {LINE}; spacing: 4px; padding: 3px; }}
QToolBar QToolButton {{ padding: 4px 8px; border-radius: 6px; }}
QToolButton:hover, QToolBar QToolButton:hover {{ background: rgba(34,211,238,.14); }}
QToolButton:checked {{ background: rgba(34,211,238,.24); border: 1px solid {ACCENT}; }}
QStatusBar {{ background: {BG2}; border-top: 1px solid {LINE}; color: {MUTED}; }}
QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none; }}
QDockWidget::title {{ background: {BG2}; padding: 6px 10px; border-bottom: 1px solid {LINE};
                      font-weight: 600; letter-spacing: .06em; }}
QTabWidget::pane {{ border: 1px solid {LINE}; border-radius: 6px; top: -1px; }}
QTabBar::tab {{ background: transparent; padding: 7px 14px; border-bottom: 2px solid transparent; color: {MUTED}; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {INK}; }}
QGroupBox {{ border: 1px solid {LINE}; border-radius: 8px; margin-top: 12px; padding: 10px 8px 6px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
    background: {BG2}; border: 1px solid {LINE}; border-radius: 6px; padding: 4px 6px;
    selection-background-color: {ACCENT}; selection-color: {ACCENT_INK}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {MUTED}; }}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {LINE}; selection-background-color: rgba(34,211,238,.25); }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 14px; border: 0; }}
QPushButton {{ background: {BG2}; border: 1px solid {LINE}; border-radius: 6px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; background: rgba(34,211,238,.10); }}
QPushButton:pressed {{ background: rgba(34,211,238,.25); }}
QPushButton:disabled {{ color: {MUTED}; border-color: {LINE}; }}
QPushButton[primary="true"] {{ background: {ACCENT}; color: {ACCENT_INK}; font-weight: 600; border-color: {ACCENT}; }}
QPushButton[primary="true"]:hover {{ background: #67e8f9; }}
QPushButton[danger="true"]:hover {{ border-color: {BAD}; background: rgba(239,68,68,.15); }}
QPushButton[chip="true"] {{ border-radius: 12px; padding: 3px 10px; font-size: 12px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; }}
QCheckBox::indicator {{ border: 1px solid {MUTED}; border-radius: 3px; background: {BG2}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url("{CHECK}"); }}
QSlider::groove:horizontal {{ height: 4px; background: {LINE}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; background: {ACCENT}; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QHeaderView::section {{ background: {BG2}; color: {MUTED}; padding: 5px 6px; border: 0; border-bottom: 1px solid {LINE}; border-right: 1px solid {LINE}; }}
QTableView, QTreeView, QListView {{ background: {BG}; alternate-background-color: {BG2}; gridline-color: {LINE};
    border: 1px solid {LINE}; border-radius: 6px; selection-background-color: rgba(34,211,238,.25); selection-color: {INK}; }}
QTableView::item, QListView::item {{ padding: 3px 6px; }}
QListView::item:selected, QTreeView::item:selected {{ background: rgba(34,211,238,.25); }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {LINE}; min-height: 24px; border-radius: 5px; }}
QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {LINE}; min-width: 24px; border-radius: 5px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QSplitter::handle {{ background: {LINE}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QProgressBar {{ background: {BG2}; border: 1px solid {LINE}; border-radius: 4px; text-align: center; height: 8px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QFrame[card="true"] {{ background: rgba(148,163,184,.07); border: 1px solid {LINE}; border-radius: 10px; }}
QFrame[verdict="true"] {{ background: rgba(57,135,229,.12); border: 1px solid rgba(57,135,229,.4); border-radius: 10px; }}
QFrame[notice="true"] {{ background: rgba(245,158,11,.14); border: 1px solid rgba(245,158,11,.45); border-radius: 8px; }}
QFrame[danger="true"] {{ background: rgba(239,68,68,.16); border: 1px solid rgba(239,68,68,.55); border-radius: 8px; }}
QFrame[info="true"] {{ background: rgba(34,211,238,.08); border: 1px solid rgba(34,211,238,.35); border-radius: 8px; }}
QScrollArea {{ border: 0; }}
QTextBrowser {{ background: {BG}; border: 0; }}
QLabel[muted="true"] {{ color: {MUTED}; }}
QLabel[h1="true"] {{ font-size: 20px; font-weight: 700; }}
QLabel[h2="true"] {{ font-size: 15px; font-weight: 700; }}
QLabel[kpi="true"] {{ font-size: 26px; font-weight: 600; }}
QLabel[pill="true"] {{ padding: 1px 8px; border-radius: 9px; font-weight: 600; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(INK))
    pal.setColor(QPalette.Base, QColor(BG2))
    pal.setColor(QPalette.AlternateBase, QColor(PANEL))
    pal.setColor(QPalette.Text, QColor(INK))
    pal.setColor(QPalette.Button, QColor(BG2))
    pal.setColor(QPalette.ButtonText, QColor(INK))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor(ACCENT_INK))
    pal.setColor(QPalette.ToolTipBase, QColor(PANEL))
    pal.setColor(QPalette.ToolTipText, QColor(INK))
    pal.setColor(QPalette.PlaceholderText, QColor(MUTED))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(MUTED))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(MUTED))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    f = QFont()
    for name in ("Segoe UI", "Inter", "Helvetica Neue", "DejaVu Sans"):
        if name in QFontDatabase.families():
            f.setFamily(name)
            break
    f.setPointSize(10)
    app.setFont(f)
    if (ASSETS / "favicon.png").exists():
        app.setWindowIcon(QIcon(str(ASSETS / "favicon.png")))


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"
