"""Batch reports: interactive HTML, Markdown and LLM-oriented JSON.

All three are built from the same analysis:

* the best case (objective, subject to the constraints and convergence);
* one-at-a-time sensitivity through the best case: for every swept axis, the
  metrics along that axis with every other axis held at its best value;
* a 2-D map over the two most finely swept axes (others at their best values);
* plain-language observations, e.g. "the optimum is at the edge of the swept
  range: extend it", or which constraint stops the objective from growing.

The HTML report is self-contained (plotly.js is embedded once), so it opens
offline and can be mailed. The JSON report carries the setup, the units and
definitions of every metric, every case and the analysis, for an LLM to write
up or query.
"""
from __future__ import annotations

import html
import json
import math
import os
import time

import numpy as np

from .batch import AXES, METRICS, BatchSpec, passes, ranked, table

PALETTE = ["#0b6e74", "#c2185b", "#3b6fb6", "#b5652c", "#5f8f27", "#7c5cbf", "#8a6d00", "#3d5a6c"]


def _label(key):
    if key in METRICS:
        _, lab, unit = METRICS[key]
    elif key in AXES:
        lab, unit, _ = AXES[key]
    else:
        return key
    return f"{lab} ({unit})" if unit else lab


def _fmt(v, nd=4):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1e5 or a < 1e-3:
            return f"{v:.3e}"
        return f"{v:.{nd}g}"
    return str(v)


# ----------------------------------------------------------------- analysis
def analyse(spec: BatchSpec, rows):
    axes = [k for k, v in spec.axes.items() if len(v) > 1]
    fixed = {k: v[0] for k, v in spec.axes.items() if len(v) == 1}
    good = ranked(spec, rows)
    n_ok = sum(1 for r in rows if r.get("ok"))
    n_err = sum(1 for r in rows if not r.get("ok"))
    n_pass = len(good)
    best = good[0] if good else None
    obj = spec.objective
    # which constraint rules out cases that would beat the best one
    blockers = {}
    if best:
        for r in rows:
            if not r.get("ok") or passes(r, spec.constraints):
                continue
            v = r["metrics"].get(obj)
            if v is None:
                continue
            better = v > best["metrics"][obj] if spec.maximize else v < best["metrics"][obj]
            if not better:
                continue
            m = r["metrics"]
            if not m.get("converged"):
                blockers["did not converge"] = blockers.get("did not converge", 0) + 1
            if m.get("sbs_runaway"):
                blockers["SBS runaway"] = blockers.get("SBS runaway", 0) + 1
            for key, (op, val) in spec.constraints.items():
                x = m.get(key)
                if x is None or (op == "<" and not x < val) or (op == ">" and not x > val):
                    nm = f"{METRICS.get(key, (0, key))[1]} {op} {val}"
                    blockers[nm] = blockers.get(nm, 0) + 1
    # sensitivity through the best case
    sens = {}
    if best:
        for ax in axes:
            pts = []
            for r in rows:
                if not r.get("ok"):
                    continue
                if all(r["values"].get(k) == best["values"].get(k) for k in axes if k != ax):
                    pts.append(r)
            pts.sort(key=lambda r: _sortkey(r["values"][ax]))
            sens[ax] = pts
    # 2-D map over the two axes with most values
    grid = None
    num_axes = [a for a in axes if all(isinstance(v, (int, float)) for v in spec.axes[a])]
    if best and len(num_axes) >= 2:
        ax2 = sorted(num_axes, key=lambda a: -len(spec.axes[a]))[:2]
        xs, ys = spec.axes[ax2[0]], spec.axes[ax2[1]]
        Z = [[None] * len(xs) for _ in ys]
        P = [[False] * len(xs) for _ in ys]
        for r in rows:
            if not r.get("ok"):
                continue
            if not all(r["values"].get(k) == best["values"].get(k) for k in axes if k not in ax2):
                continue
            i, j = ys.index(r["values"][ax2[1]]), xs.index(r["values"][ax2[0]])
            Z[i][j] = r["metrics"].get(obj)
            P[i][j] = passes(r, spec.constraints)
        grid = dict(x_axis=ax2[0], y_axis=ax2[1], x=xs, y=ys, z=Z, passes=P)
    obs = _observations(spec, best, sens, blockers, n_pass, n_ok, n_err)
    warn_count = {}
    for r in rows:
        for w in r.get("warnings", []):
            key = w.split(":")[0][:80]
            warn_count[key] = warn_count.get(key, 0) + 1
    return dict(axes=axes, fixed=fixed, best=best, top=good[:10], n_cases=len(rows), n_ok=n_ok, n_err=n_err,
                n_pass=n_pass, blockers=blockers, sensitivity=sens, grid=grid, observations=obs,
                warnings=sorted(warn_count.items(), key=lambda kv: -kv[1]))


def _sortkey(v):
    return (0, v) if isinstance(v, (int, float)) else (1, str(v))


def _observations(spec, best, sens, blockers, n_pass, n_ok, n_err):
    obs = []
    obj_lab = METRICS.get(spec.objective, (0, spec.objective, ""))[1]
    if n_err:
        obs.append(f"{n_err} case(s) failed to run; see the error column.")
    if not best:
        obs.append("No case met the constraints. Relax them or move the sweep ranges.")
        return obs
    obs.append(f"{n_pass} of {n_ok} finished cases meet the constraints.")
    for ax, pts in sens.items():
        vals = [p["values"][ax] for p in pts]
        ys = [p["metrics"].get(spec.objective) for p in pts]
        if len(pts) < 2 or any(y is None for y in ys):
            continue
        bv = best["values"][ax]
        lab = AXES[ax][0]
        unit = AXES[ax][1]
        allv = spec.axes[ax]
        if isinstance(bv, (int, float)) and bv in (allv[0], allv[-1]) and len(allv) > 2:
            side = "upper" if bv == allv[-1] else "lower"
            obs.append(f"Best {lab} ({_fmt(bv)} {unit}) is at the {side} end of the swept range: the optimum may lie "
                       f"beyond it; extend the {lab} axis.")
        diffs = np.diff(ys)
        if len(ys) == 2:
            trend = (f"is {'higher' if ys[1] > ys[0] else 'lower'} at {_fmt(vals[1])} {unit} than at "
                     f"{_fmt(vals[0])} {unit}")
        elif np.all(diffs >= 0):
            trend = "rises steadily"
        elif np.all(diffs <= 0):
            trend = "falls steadily"
        else:
            k = int(np.argmax(ys) if spec.maximize else np.argmin(ys))
            trend = f"peaks near {_fmt(vals[k])} {unit}"
        spread = (max(ys) - min(ys)) / max(abs(best["metrics"][spec.objective]), 1e-300) * 100
        obs.append(f"Along {lab} (others at their best values) the {obj_lab} {trend}; it varies by {spread:.0f} % "
                   "of the best value over the range.")
        fails = [p for p in pts if not passes(p, spec.constraints)]
        if fails:
            fv = [_fmt(p["values"][ax]) for p in fails]
            obs.append(f"  {lab} values excluded by the constraints: {', '.join(fv)} {unit}.")
    if blockers:
        top = max(blockers.items(), key=lambda kv: kv[1])
        obs.append(f"The limit that stops the {obj_lab} from going higher is mostly '{top[0]}' "
                   f"({top[1]} better-scoring case(s) excluded by it).")
    return obs


# ------------------------------------------------------------------- figures
def _figures(spec, rows, an):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    figs = []
    obj = spec.objective
    lay = dict(template="plotly_white", font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", size=13),
               margin=dict(l=64, r=24, t=56, b=56), height=420, legend=dict(orientation="h", y=-0.22))
    best = an["best"]
    # 1. sensitivity: objective and three companions along each axis through the best case (small multiples)
    companions = [k for k in ("pulse_frac_pct", "sbs_pct", "B_rad", "ase_fwd_W") if k != obj][:3]
    panels = [obj] + companions
    for ax, pts in an["sensitivity"].items():
        if len(pts) < 2:
            continue
        x = [p["values"][ax] for p in pts]
        ok = [passes(p, spec.constraints) for p in pts]
        fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.16, horizontal_spacing=0.12,
                            subplot_titles=[_label(k) for k in panels])
        for n, key in enumerate(panels):
            r_, c_ = n // 2 + 1, n % 2 + 1
            col = PALETTE[n]
            fig.add_trace(go.Scatter(x=x, y=[p["metrics"].get(key) for p in pts], mode="lines+markers", showlegend=False,
                                     line=dict(color=col, width=2.5 if n == 0 else 1.8),
                                     marker=dict(size=8, color=[col if o else "white" for o in ok], line=dict(color=col, width=2)),
                                     hovertemplate=f"%{{x}}: %{{y:.4g}}<extra>{METRICS[key][1]}</extra>"), r_, c_)
            if key in spec.constraints:
                op, val = spec.constraints[key]
                fig.add_hline(y=val, line=dict(color="#b3261e", dash="dot", width=1), row=r_, col=c_)
            if isinstance(best["values"][ax], (int, float)):
                fig.add_vline(x=best["values"][ax], line=dict(color="#999", dash="dash", width=1), row=r_, col=c_)
        others = ", ".join(f"{AXES[k][0]} = {_fmt(best['values'][k])} {AXES[k][1]}" for k in an["axes"] if k != ax)
        fig.update_layout(**{**lay, "height": 520, "margin": dict(l=60, r=24, t=96, b=56)},
                          title=dict(text=f"{AXES[ax][0]}: sensitivity through the best case<br><sup>"
                                          + (others + "; " if others else "")
                                          + "hollow markers fail the constraints; red dotted line = constraint limit</sup>",
                                     font=dict(size=16)))
        fig.update_annotations(font=dict(size=12))
        fig.update_xaxes(title_text=_label(ax), row=2)
        figs.append(("sens_" + ax, fig))
    # 2. 2-D map
    g = an["grid"]
    if g:
        z = [[(v if v is not None else float("nan")) for v in row] for row in g["z"]]
        txt = [[("" if pz else "✕") for pz in row] for row in g["passes"]]
        fig = go.Figure(go.Heatmap(x=g["x"], y=g["y"], z=z, colorscale="Tealgrn", colorbar=dict(title=_label(obj)),
                                   text=txt, texttemplate="%{text}", hovertemplate="%{x}, %{y}: %{z:.4g}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[best["values"][g["x_axis"]]], y=[best["values"][g["y_axis"]]], mode="markers",
                                 marker=dict(symbol="star", size=18, color="#c2185b", line=dict(color="white", width=1.5)),
                                 name="best"))
        fig.update_layout(**{**lay, "height": 480}, title=dict(text=f"{METRICS[obj][1]} over {AXES[g['x_axis']][0]} and "
                                                              f"{AXES[g['y_axis']][0]}<br><sup>✕ fails the constraints; "
                                                              "other axes at their best values</sup>", font=dict(size=16)))
        fig.update_xaxes(title_text=_label(g["x_axis"]))
        fig.update_yaxes(title_text=_label(g["y_axis"]))
        figs.append(("map", fig))
    # 3. pulses and spectra of the top cases
    top = [r for r in an["top"][:5] if r.get("pulse")]
    if top:
        fig = go.Figure()
        for i, r in enumerate(top):
            lab = ", ".join(f"{_fmt(v)}" for v in r["values"].values()) or "case"
            fig.add_trace(go.Scatter(x=r["pulse"]["t_ns"], y=np.asarray(r["pulse"]["P_out_W"]) / 1e3, name=f"#{i+1}: {lab}",
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2 if i == 0 else 1.3)))
        fig.update_layout(**lay, title=dict(text="Output pulses of the best cases", font=dict(size=16)))
        fig.update_xaxes(title_text="time (ns)")
        fig.update_yaxes(title_text="power (kW)")
        figs.append(("pulses", fig))
    top = [r for r in an["top"][:5] if r.get("spectrum")]
    if top:
        fig = go.Figure()
        for i, r in enumerate(top):
            lab = ", ".join(f"{_fmt(v)}" for v in r["values"].values()) or "case"
            fig.add_trace(go.Scatter(x=r["spectrum"]["lam_nm"], y=r["spectrum"]["dBm_per_rbw"], name=f"#{i+1}: {lab}",
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2 if i == 0 else 1.2)))
        rbw = top[0]["spectrum"]["rbw_nm"]
        fig.update_layout(**lay, title=dict(text=f"Output spectra of the best cases (OSA view, RBW {rbw} nm)",
                                            font=dict(size=16)))
        fig.update_xaxes(title_text="wavelength (nm)")
        fig.update_yaxes(title_text=f"dBm / {rbw} nm")
        figs.append(("spectra", fig))
    # 4. all cases: objective against the most finely swept axis, coloured by the next one
    if an["axes"]:
        ax1 = sorted(an["axes"], key=lambda a: -len(spec.axes[a]))[0]
        rest = [a for a in an["axes"] if a != ax1]
        ax2 = sorted(rest, key=lambda a: -len(spec.axes[a]))[0] if rest else None
        fig = go.Figure()
        groups = {}
        for r in rows:
            if not r.get("ok"):
                continue
            key = r["values"].get(ax2) if ax2 else "all"
            groups.setdefault(key, []).append(r)
        for i, (k, rs) in enumerate(sorted(groups.items(), key=lambda kv: _sortkey(kv[0]))):
            rs = sorted(rs, key=lambda r: _sortkey(r["values"][ax1]))
            ok = [passes(r, spec.constraints) for r in rs]
            c = PALETTE[i % len(PALETTE)]
            fig.add_trace(go.Scatter(x=[r["values"][ax1] for r in rs], y=[r["metrics"].get(obj) for r in rs],
                                     mode="markers", name=(f"{AXES[ax2][0]} = {_fmt(k)} {AXES[ax2][1]}" if ax2 else "cases"),
                                     marker=dict(size=8, color=[c if o else "white" for o in ok], line=dict(color=c, width=1.5)),
                                     text=[json.dumps(r["values"]) for r in rs], hovertemplate="%{text}<br>%{y:.4g}<extra></extra>"))
        fig.update_layout(**lay, title=dict(text=f"All cases: {METRICS[obj][1]} against {AXES[ax1][0]}"
                                            "<br><sup>every combination of the other axes; hollow markers fail the constraints</sup>",
                                            font=dict(size=16)))
        fig.update_xaxes(title_text=_label(ax1))
        fig.update_yaxes(title_text=_label(obj))
        figs.append(("all", fig))
    return figs


# --------------------------------------------------------------------- HTML
_CSS = """
:root{--bg:#f4f6f6;--surface:#fff;--ink:#16212a;--muted:#59656e;--rule:#d5dcdf;--accent:#0b6e74;--win:#c2185b;--bad:#b3261e}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans","Segoe UI",system-ui,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:32px 20px 64px;display:flex;flex-direction:column;gap:34px}
h1{font:700 40px/1.05 "Barlow Condensed","Arial Narrow",sans-serif;margin:0;text-wrap:balance}
h2{font:600 26px/1.2 "Barlow Condensed","Arial Narrow",sans-serif;margin:0 0 10px}
.eyebrow{font:500 12px/1 "IBM Plex Mono",monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.lede{color:var(--muted);max-width:72ch;margin:8px 0 0}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:18px 20px}
.best{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px 28px}
.best .v{font:600 30px/1.1 "Barlow Condensed",sans-serif;font-variant-numeric:tabular-nums}
.best dl{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;margin:0;font-size:13.5px}
.best dt{color:var(--muted)} .best dd{margin:0;text-align:right;font-family:"IBM Plex Mono",monospace}
ul.obs{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:6px;max-width:80ch}
.tw{overflow-x:auto;background:var(--surface);border:1px solid var(--rule);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
th{background:#e9edee;font:500 11.5px "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);position:sticky;top:0}
td{font-family:"IBM Plex Mono",monospace} td:first-child,th:first-child{text-align:left}
tr.fail td{color:#9aa4ab} tr.top td{background:#fbeef3}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:18px}
.figs .card{padding:8px}
.cfg{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:4px 26px;font:13px/1.5 "IBM Plex Mono",monospace;color:var(--muted)}
.cfg b{color:var(--ink);font-weight:500}
footer{font-size:12.5px;color:var(--muted)}
@media (max-width:560px){.figs{grid-template-columns:1fr}}
"""


def _base_lines(spec):
    f, p, s, m = spec.fiber, spec.pump, spec.seed, spec.sim
    ab = ", ".join(f"{a['value_dB_per_m']:g} dB/m @{a['wavelength_nm']:g} nm ({a['kind']})" for a in f.get("absorption", []))
    lines = [
        ("fibre", f"{f.get('name')} — {f.get('dopant')}, {f.get('d_core_um'):g}/{f.get('d_clad_um'):g} µm, NA {f.get('na_core'):g}"),
        ("absorption", ab),
        ("length", f"{f.get('length_m'):g} m"),
        ("signal mode", f"{f.get('signal_mode')}, rings {f.get('n_rings') or 'auto'}"),
        ("Er pairs / ESA / T", f"{f.get('pair_fraction', 0)*100:g} % / {'on' if f.get('esa', True) else 'off'} / {f.get('temperature_K', 300):g} K"),
        ("pump", f"{p['geometry']}, co {p['co']['power_W']:g} W @{p['co']['wavelength_nm']:g} nm, "
                 f"counter {p['counter']['power_W']:g} W @{p['counter']['wavelength_nm']:g} nm" + (", pulsed" if p.get("pulsed") else "")),
        ("seed", f"{s['shape']}, {s['fwhm_ns']:g} ns, {s['energy_uJ']:g} µJ, {s['wavelength_nm']:g} nm, "
                 f"linewidth {s['linewidth_nm']:g} nm, {s['rep_rate_kHz']:g} kHz"),
        ("numerics", f"nz {m['nz']}, ASE bands {m['ase_step_nm']:g} nm, min. window {m.get('min_spectral_nm', 10):g} nm, "
                     f"SBS {'on' if m.get('sbs', True) else 'off'}, nonlinear model {m.get('nl_model', 'auto')}"),
    ]
    return lines


def _constraint_text(spec):
    parts = [f"{METRICS[k][1]} {op} {_fmt(v)} {METRICS[k][2]}" for k, (op, v) in spec.constraints.items()]
    return ", ".join(parts + ["converged", "no SBS runaway"])


def _row_label(r):
    return ", ".join(f"{AXES[k][0]} {_fmt(v)}{(' ' + AXES[k][1]) if AXES[k][1] else ''}" for k, v in r["values"].items())


def html_report(spec, rows, an=None) -> str:
    import plotly.io as pio
    an = an or analyse(spec, rows)
    figs = _figures(spec, rows, an)
    obj = spec.objective
    best = an["best"]
    head = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(spec.name)} — fibamp batch</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{_CSS}</style></head><body><div class="wrap">"""
    axes_txt = "; ".join(f"{AXES[k][0]}: {', '.join(_fmt(v) for v in vals)} {AXES[k][1]}" for k, vals in spec.axes.items() if vals)
    out = [head,
           f"<header><span class='eyebrow'>fibamp batch · {html.escape(spec.created or time.strftime('%Y-%m-%d %H:%M'))} · "
           f"{an['n_cases']} cases</span><h1>{html.escape(spec.name)}</h1>"
           f"<p class='lede'>Objective: {'maximise' if spec.maximize else 'minimise'} <b>{METRICS[obj][1]}</b>, subject to "
           f"{html.escape(_constraint_text(spec))}. Swept: {html.escape(axes_txt) or 'nothing (single case)'}.</p></header>"]
    if best:
        m = best["metrics"]
        keys = [k for k in ("E_out_uJ", "peak_kW", "gain_dB", "pulse_frac_pct", "sbs_pct", "B_rad", "ase_fwd_W",
                            "ase_bwd_W", "fwhm_out_ns", "dT_K") if k in m]
        dl = "".join(f"<dt>{METRICS[k][1]}</dt><dd>{_fmt(m[k])} {METRICS[k][2]}</dd>" for k in keys)
        vals = "".join(f"<dt>{AXES[k][0]}</dt><dd>{_fmt(v)} {AXES[k][1]}</dd>" for k, v in best["values"].items())
        out.append(f"<section class='card best'><div><span class='eyebrow'>Best case</span>"
                   f"<div class='v'>{_fmt(m[obj])} {METRICS[obj][2]}</div><div class='lede' style='margin:0'>{METRICS[obj][1]}</div>"
                   f"<dl style='margin-top:12px'>{vals}</dl></div><div><dl>{dl}</dl></div></section>")
    else:
        out.append("<section class='card'><b>No case meets the constraints.</b></section>")
    out.append("<section><h2>Observations</h2><ul class='obs'>" + "".join(f"<li>{html.escape(o)}</li>" for o in an["observations"])
               + "</ul></section>")
    if figs:
        out.append("<section><h2>Figures</h2><div class='figs'>")
        first = True
        for name, fig in figs:
            out.append("<div class='card'>" + pio.to_html(fig, include_plotlyjs=True if first else False, full_html=False,
                                                        config=dict(displaylogo=False, responsive=True)) + "</div>")
            first = False
        out.append("</div></section>")
    # top table
    cols = [k for k in ("E_out_uJ", "peak_kW", "gain_dB", "pulse_frac_pct", "sbs_pct", "B_rad", "ase_fwd_W", "dT_K")]
    if an["top"]:
        th = "".join(f"<th>{html.escape(AXES[k][0])}</th>" for k in an["axes"]) + "".join(f"<th>{METRICS[k][1]}</th>" for k in cols)
        tb = ""
        for i, r in enumerate(an["top"]):
            tb += f"<tr class='{'top' if i == 0 else ''}'>" + "".join(f"<td>{_fmt(r['values'].get(k))}</td>" for k in an["axes"]) \
                + "".join(f"<td>{_fmt(r['metrics'].get(k))}</td>" for k in cols) + "</tr>"
        out.append(f"<section><h2>Top {len(an['top'])} cases</h2><div class='tw'><table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div></section>")
    # all cases
    recs = table(rows, spec)
    keys = list(dict.fromkeys(k for r in recs for k in r if k not in ("warnings",)))
    th = "".join(f"<th>{html.escape(AXES[k][0] if k in AXES else (METRICS[k][1] if k in METRICS else k))}</th>" for k in keys)
    tb = "".join(f"<tr class='{'' if r.get('meets constraints') else 'fail'}'>" + "".join(f"<td>{_fmt(r.get(k))}</td>" for k in keys)
                 + "</tr>" for r in recs[:2000])
    out.append(f"<section><h2>All cases</h2><p class='lede' style='margin:0 0 8px'>Grey rows fail a constraint or did not finish."
               f"{' First 2000 shown.' if len(recs) > 2000 else ''}</p><div class='tw' style='max-height:560px'><table><thead><tr>{th}</tr></thead>"
               f"<tbody>{tb}</tbody></table></div></section>")
    if an["warnings"]:
        out.append("<section><h2>Warnings raised</h2><ul class='obs'>" + "".join(
            f"<li>{n}× {html.escape(w)}</li>" for w, n in an["warnings"][:15]) + "</ul></section>")
    cfg = "".join(f"<div>{html.escape(k)}: <b>{html.escape(str(v))}</b></div>" for k, v in _base_lines(spec))
    out.append(f"<section><h2>Base configuration</h2><div class='card cfg'>{cfg}</div></section>")
    out.append("<footer>Generated by fibamp. Every case is a full periodic-steady-state amplifier simulation; cross sections, "
               "concentrations and the clustering fraction are the user's inputs and set the accuracy.</footer></div></body></html>")
    return "".join(out)


# ----------------------------------------------------------------- Markdown
def markdown_report(spec, rows, an=None) -> str:
    an = an or analyse(spec, rows)
    obj = spec.objective
    L = [f"# {spec.name}", "",
         f"fibamp batch, {spec.created or time.strftime('%Y-%m-%d %H:%M')}, {an['n_cases']} cases "
         f"({an['n_ok']} finished, {an['n_pass']} meet the constraints).", "",
         f"**Objective:** {'maximise' if spec.maximize else 'minimise'} {METRICS[obj][1]} ({METRICS[obj][2]}), "
         f"subject to {_constraint_text(spec)}.", "", "## Swept parameters", ""]
    for k, vals in spec.axes.items():
        if vals:
            L.append(f"- {AXES[k][0]} ({AXES[k][1] or '-'}): {', '.join(_fmt(v) for v in vals)}")
    L += ["", "## Base configuration", ""] + [f"- {k}: {v}" for k, v in _base_lines(spec)]
    best = an["best"]
    L += ["", "## Best case", ""]
    if best:
        L += [f"- {AXES[k][0]}: {_fmt(v)} {AXES[k][1]}" for k, v in best["values"].items()]
        L += [f"- {METRICS[k][1]}: {_fmt(v)} {METRICS[k][2]}" for k, v in best["metrics"].items() if k in METRICS and v is not None]
    else:
        L.append("No case meets the constraints.")
    L += ["", "## Observations", ""] + [f"- {o}" if not o.startswith("  ") else f"  - {o.strip()}" for o in an["observations"]]
    for ax, pts in an["sensitivity"].items():
        if len(pts) < 2:
            continue
        cols = ["E_out_uJ", "peak_kW", "pulse_frac_pct", "sbs_pct", "B_rad"]
        L += ["", f"### Along {AXES[ax][0]} (other axes at their best values)", "",
              f"| {AXES[ax][0]} ({AXES[ax][1]}) | " + " | ".join(f"{METRICS[c][1]} ({METRICS[c][2]})" for c in cols) + " | meets constraints |",
              "|" + "---|" * (len(cols) + 2)]
        for p in pts:
            L.append(f"| {_fmt(p['values'][ax])} | " + " | ".join(_fmt(p["metrics"].get(c)) for c in cols)
                     + f" | {'yes' if passes(p, spec.constraints) else 'no'} |")
    if an["top"]:
        cols = ["E_out_uJ", "peak_kW", "gain_dB", "pulse_frac_pct", "sbs_pct", "B_rad"]
        L += ["", "## Top cases", "", "| # | " + " | ".join(AXES[k][0] for k in an["axes"]) + " | "
              + " | ".join(f"{METRICS[c][1]} ({METRICS[c][2]})" for c in cols) + " |",
              "|" + "---|" * (1 + len(an["axes"]) + len(cols))]
        for i, r in enumerate(an["top"]):
            L.append(f"| {i+1} | " + " | ".join(_fmt(r["values"].get(k)) for k in an["axes"]) + " | "
                     + " | ".join(_fmt(r["metrics"].get(c)) for c in cols) + " |")
    if an["warnings"]:
        L += ["", "## Warnings raised", ""] + [f"- {n}× {w}" for w, n in an["warnings"][:15]]
    L += ["", "Full per-case data: `report.json` / `results.csv` in the same folder.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------- JSON
def json_report(spec, rows, an=None) -> dict:
    """Self-describing report for LLM-aided analysis: setup, definitions, every case, analysis."""
    an = an or analyse(spec, rows)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("pulse", "spectrum", "trace")}
    return {
        "about": ("fibamp batch report. Each case is a full periodic-steady-state simulation of a pulsed Er/Er:Yb fibre "
                  "amplifier (rate equations with spectrally resolved ASE between pulses, time-resolved saturated gain, "
                  "dispersion, SPM, Raman and a simulated SBS Stokes wave during the pulse). 'values' are the swept inputs; "
                  "'metrics' are the outputs. A case 'meets_constraints' when it converged, has no SBS runaway and satisfies "
                  "every entry of 'constraints'."),
        "name": spec.name, "created": spec.created,
        "objective": {"metric": spec.objective, "maximize": spec.maximize},
        "constraints": {k: {"op": op, "value": v, "label": METRICS[k][1], "unit": METRICS[k][2]} for k, (op, v) in spec.constraints.items()},
        "axes": {k: {"label": AXES[k][0], "unit": AXES[k][1], "values": v} for k, v in spec.axes.items() if v},
        "metric_definitions": {k: {"label": lab, "unit": u, "source": src} for k, (src, lab, u) in METRICS.items()},
        "base_configuration": {"fibre": spec.fiber, "pump": spec.pump, "seed": spec.seed, "numerics": spec.sim,
                               "readable": dict(_base_lines(spec))},
        "summary": {"cases": an["n_cases"], "finished": an["n_ok"], "failed": an["n_err"], "meet_constraints": an["n_pass"],
                    "best": strip(an["best"]) if an["best"] else None,
                    "top": [strip(r) for r in an["top"]],
                    "observations": [o.strip() for o in an["observations"]],
                    "excluded_better_cases_by_limit": an["blockers"],
                    "warnings": [{"text": w, "count": n} for w, n in an["warnings"]]},
        "sensitivity_through_best": {ax: [{"value": p["values"][ax], "meets_constraints": passes(p, spec.constraints),
                                           **{k: p["metrics"].get(k) for k in METRICS}} for p in pts]
                                     for ax, pts in an["sensitivity"].items()},
        "cases": [dict(strip(r), meets_constraints=passes(r, spec.constraints)) for r in rows],
    }


def csv_text(spec, rows) -> str:
    import csv
    import io
    recs = table(rows, spec)
    keys = list(dict.fromkeys(k for r in recs for k in r))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    for r in recs:
        w.writerow(r)
    return buf.getvalue()


def write_reports(spec, rows, folder) -> dict:
    an = analyse(spec, rows)
    os.makedirs(folder, exist_ok=True)
    paths = {}
    for name, text in (("report.html", html_report(spec, rows, an)), ("report.md", markdown_report(spec, rows, an)),
                       ("report.json", json.dumps(json_report(spec, rows, an), indent=1, default=float)),
                       ("results.csv", csv_text(spec, rows))):
        p = os.path.join(folder, name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        paths[name] = p
    return paths
