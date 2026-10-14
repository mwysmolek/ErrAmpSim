"""Plotly figures inside the desktop app: one web view per tab, charts in a grid.

plotly.js is written once to the data folder and loaded from there, so the app works offline.
The web views use an off-the-record profile: Chromium keeps its cache in memory and writes
nothing to the user profile (portable app).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from PySide6.QtCore import QUrl
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView

from fibamp.paths import sub

from . import theme as T

_PLOTLY_FILE: Optional[Path] = None
_PROFILE: Optional[QWebEngineProfile] = None


def profile() -> QWebEngineProfile:
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = QWebEngineProfile()            # no storage name -> off the record
        _PROFILE.setHttpCacheType(QWebEngineProfile.MemoryHttpCache)
        _PROFILE.setPersistentCookiesPolicy(QWebEngineProfile.NoPersistentCookies)
    return _PROFILE


def plotly_js_path() -> Path:
    """plotly.min.js on disk (written on first use)."""
    global _PLOTLY_FILE
    if _PLOTLY_FILE is not None and _PLOTLY_FILE.exists():
        return _PLOTLY_FILE
    import plotly
    f = sub("cache") / f"plotly-{plotly.__version__}.min.js"
    if not f.exists():
        from plotly.offline import get_plotlyjs
        tmp = f.with_suffix(".tmp")
        tmp.write_text(get_plotlyjs(), encoding="utf-8")
        os.replace(tmp, f)
    _PLOTLY_FILE = f
    return f


TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<script src="{js}"></script>
<style>
html,body{{margin:0;background:{bg};color:{ink};font-family:"Segoe UI",system-ui,sans-serif;}}
.grid{{display:grid;grid-template-columns:repeat({cols},minmax(0,1fr));gap:12px;padding:8px;}}
.card{{background:rgba(148,163,184,.06);border:1px solid {line};border-radius:10px;padding:4px;min-width:0;}}
.card.wide{{grid-column:1 / -1;}}
.empty{{padding:40px;color:{muted};text-align:center;font-size:14px;line-height:1.6}}
</style></head><body>
<div class="grid" id="grid">{cards}</div>
<script>
const figs = {figs};
const cfg = {{displayModeBar: "hover", displaylogo: false, responsive: true,
             modeBarButtonsToRemove: ["lasso2d", "select2d"],
             toImageButtonOptions: {{format: "png", scale: 2}}}};
const ink = "{ink}", line = "{line}", muted = "{muted}";
for (const [id, fig] of Object.entries(figs)) {{
  const L = fig.layout = fig.layout || {{}};
  L.paper_bgcolor = "rgba(0,0,0,0)"; L.plot_bgcolor = "rgba(0,0,0,0)";
  L.font = Object.assign({{}}, L.font || {{}}, {{color: ink}});
  L.autosize = true; delete L.width;
  L.legend = Object.assign({{bgcolor: "rgba(0,0,0,0)"}}, L.legend || {{}});
  L.margin = Object.assign({{}}, L.margin || {{}}, {{l: Math.max(56, (L.margin||{{}}).l||0)}});
  if (L.title && typeof L.title === "object") L.title.font = Object.assign({{size: 13}}, L.title.font || {{}});
  for (const ax of Object.keys(L).filter(k => /^[xy]axis/.test(k)))
    L[ax] = Object.assign({{automargin: true}}, L[ax], {{gridcolor: line, linecolor: line, zerolinecolor: line}});
  if (!L.yaxis) L.yaxis = {{automargin: true, gridcolor: line}};
  if (!L.xaxis) L.xaxis = {{gridcolor: line}};
  for (const tr of fig.data || []) if (tr.colorbar) tr.colorbar = Object.assign({{}}, tr.colorbar, {{outlinecolor: line}});
  Plotly.newPlot(id, fig.data, L, cfg);
}}
window.addEventListener("resize", () => {{ for (const id of Object.keys(figs)) Plotly.Plots.resize(id); }});
</script></body></html>"""


class ChartGrid(QWebEngineView):
    """Shows a list of figures in a responsive grid."""

    def __init__(self, parent=None, cols: int = 2, empty: str = "No results yet."):
        super().__init__(parent)
        self.setPage(QWebEnginePage(profile(), self))
        self.cols = cols
        self.page().setBackgroundColor(T.BG)
        self.setMinimumHeight(260)
        self._last = None
        self._n = 0
        self.show_message(empty)

    def show_message(self, text: str) -> None:
        html = TEMPLATE.format(js="", bg=T.BG, ink=T.INK, line=T.LINE, muted=T.MUTED, cols=1,
                               cards=f'<div class="empty">{text}</div>', figs="{}")
        self.setHtml(html)
        self._last = None

    def set_figures(self, figs: list[tuple[Optional[go.Figure], bool]]) -> None:
        """figs: (figure, spans_full_width)."""
        cards, data = [], {}
        sig = [(f.to_json() if f is not None else None, w) for f, w in figs]
        if sig == self._last:
            return
        self._last = sig
        for i, (fig, wide) in enumerate(figs):
            if fig is None:
                continue
            fid = f"f{i}"
            h = int(fig.layout.height or 380)
            cards.append(f'<div class="card{" wide" if wide else ""}"><div id="{fid}" style="height:{h}px"></div></div>')
            data[fid] = json.loads(sig[i][0])
        if not cards:
            self.show_message("Nothing to show.")
            return
        js_file = plotly_js_path()
        html = TEMPLATE.format(js=js_file.name, bg=T.BG, ink=T.INK, line=T.LINE, muted=T.MUTED,
                               cols=self.cols, cards="".join(cards), figs=json.dumps(data))
        # a page loaded from a file may load the chart library from the same folder; setHtml pages
        # may not (and are limited to 2 MB anyway)
        self._n += 1
        page = js_file.parent / f"page-{os.getpid()}-{id(self)}-{self._n}.html"
        tmp = page.with_suffix(".tmp")
        tmp.write_text(html, encoding="utf-8")
        os.replace(tmp, page)
        old = js_file.parent / f"page-{os.getpid()}-{id(self)}-{self._n - 2}.html"
        try:
            old.unlink()
        except OSError:
            pass
        self.load(QUrl.fromLocalFile(str(page)))


def clean_cache() -> None:
    """Remove chart pages left behind by earlier sessions."""
    for f in sub("cache").glob("page-*.html"):
        try:
            f.unlink()
        except OSError:
            pass
