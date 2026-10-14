"""Plotly figures for the UI (also usable from scripts / notebooks)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

C_IN, C_OUT, C_R, C_AF, C_AB, C_P = "#7f7f7f", "#1f4e9c", "#7b3fa0", "#2a9d8f", "#e07b39", "#d62728"
LAYOUT = dict(template="plotly_white", margin=dict(l=60, r=20, t=40, b=50), height=420,
              legend=dict(orientation="h", y=1.08, x=0))


def _trim(t, *ys, frac=1e-4, pad=0.15):
    P = np.max(np.vstack([np.abs(y) for y in ys]), axis=0)
    k = np.where(P > frac * P.max())[0]
    if k.size == 0:
        return slice(None)
    a, b = k[0], k[-1]
    w = b - a
    return slice(max(0, a - int(pad * w)), min(t.size, b + int(pad * w) + 1))


def _dec(x, n=6000):
    return x[:: max(1, x.size // n)]


def pulse_figure(res, normalised=False, log=False, extra=None):
    t = res.t * 1e9
    Pi, Po = np.abs(res.A_in) ** 2, np.abs(res.A_out) ** 2
    sl = _trim(t, Pi, Po)
    PRall = np.atleast_2d(res.P_raman).sum(axis=0) if np.size(res.P_raman) else np.zeros(t.size)
    t, Pi, Po, PR = t[sl], Pi[sl], Po[sl], PRall[sl]
    fig = go.Figure()
    if normalised:
        fig.add_trace(go.Scatter(x=_dec(t), y=_dec(Pi / Pi.max()), name="input (norm.)", line=dict(color=C_IN)))
        fig.add_trace(go.Scatter(x=_dec(t), y=_dec(Po / Po.max()), name="output (norm.)", line=dict(color=C_OUT)))
        ylab = "normalised power"
    else:
        fig.add_trace(go.Scatter(x=_dec(t), y=_dec(Pi / 1e3), name="input", line=dict(color=C_IN)))
        fig.add_trace(go.Scatter(x=_dec(t), y=_dec(Po / 1e3), name="output", line=dict(color=C_OUT)))
        if PR.max() > 1e-6 * Po.max():
            fig.add_trace(go.Scatter(x=_dec(t), y=_dec(PR / 1e3), name="Raman Stokes", line=dict(color=C_R)))
        Am = res.pulse.get("A_modes")
        if Am is not None and Am.shape[0] > 1:
            for nm_, a in zip(res.pulse["mode_names"], Am):
                Pm = np.abs(a[sl]) ** 2
                if Pm.max() > 1e-3 * Po.max():
                    fig.add_trace(go.Scatter(x=_dec(t), y=_dec(Pm / 1e3), name=nm_, line=dict(width=1, dash="dot")))
        ylab = "power (kW)"
    for lab, (tt, P) in (extra or {}).items():
        fig.add_trace(go.Scatter(x=_dec(tt * 1e9), y=_dec(P / (P.max() if normalised else 1e3)), name=lab,
                                 line=dict(dash="dot")))
    fig.update_layout(**LAYOUT, xaxis_title="time (ns)", yaxis_title=ylab)
    if log:
        fig.update_yaxes(type="log")
    return fig


def spectrum_figure(res, db=True, floor_dB=-80, show=("signal_out", "signal_in", "ase_fwd", "ase_bwd", "raman")):
    sp = res.spectra()
    names = dict(signal_out="signal out", signal_in="signal in", ase_fwd="forward ASE (output end)",
                 ase_bwd="backward ASE (input end)", raman="Raman Stokes", sbs_bwd="SBS Stokes (input end)")
    cols = dict(signal_out=C_OUT, signal_in=C_IN, ase_fwd=C_AF, ase_bwd=C_AB, raman=C_R, sbs_bwd="#b8860b")
    peak = max(float(np.max(v[1])) for k, v in sp.items() if len(v[1]) and k in show) or 1.0
    fig = go.Figure()
    for k in show:
        l, s = sp[k]
        if len(l) == 0 or np.max(s) <= 0:
            continue
        if k.startswith("signal"):
            # smooth the fine structure of a single-shot field to a readable density
            n = max(1, len(s) // 3000)
            s = np.convolve(s, np.ones(n) / n, mode="same")
            l, s = _dec(l, 8000), _dec(s, 8000)
        y = 10 * np.log10(np.maximum(s, peak * 10 ** (floor_dB / 10))) if db else s
        mode = "lines+markers" if k.startswith("ase") else "lines"
        fig.add_trace(go.Scatter(x=l, y=y, name=names[k], mode=mode, marker=dict(size=3),
                                 line=dict(color=cols[k])))
    fig.update_layout(**LAYOUT, xaxis_title="wavelength (nm)",
                      yaxis_title="average PSD (dBm/nm)" if db else "average PSD (W/nm)")
    if db:
        for tr in fig.data:
            tr.y = np.asarray(tr.y) + 30.0
    return fig


def profile_figure(res):
    z = res.z_nodes
    zc = res.z_cells
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.12, horizontal_spacing=0.12,
                        subplot_titles=("Er inversion (all ions, ring average)", "pulse energy and peak power",
                                        "pump and ASE (average)", "Raman energy"))
    fig.add_trace(go.Scatter(x=zc, y=res.nE_pre, name="before pulse", line=dict(color=C_OUT)), 1, 1)
    fig.add_trace(go.Scatter(x=zc, y=res.nE_post, name="after pulse", line=dict(color=C_P)), 1, 1)
    ntr = float(res.fiber.transparency(res.seed.lam0))
    fig.add_trace(go.Scatter(x=[0, z[-1]], y=[ntr, ntr], name="transparency", line=dict(color="gray", dash="dot")), 1, 1)
    if res.fiber.has_yb:
        fig.add_trace(go.Scatter(x=zc, y=res.nY_pre, name="Yb inversion", line=dict(color=C_AB)), 1, 1)
    fig.add_trace(go.Scatter(x=z, y=res.E_z * 1e6, name="pulse energy (uJ)", line=dict(color=C_OUT)), 1, 2)
    fig.add_trace(go.Scatter(x=z, y=res.Ppk_z / 1e3, name="peak power (kW)", line=dict(color=C_AF, dash="dash")), 1, 2)
    b, Pz = res.bands, res.avg["P_z"]
    for mask, lab, col in ((b.is_pump & (b.direction > 0), "co-pump", C_P), (b.is_pump & (b.direction < 0), "counter-pump", "#8c2d04"),
                           ((b.kind == "ase") & (b.direction > 0), "forward ASE", C_AF),
                           ((b.kind == "ase") & (b.direction < 0), "backward ASE", C_AB)):
        if mask.any():
            fig.add_trace(go.Scatter(x=z, y=Pz[:, mask].sum(axis=1), name=lab + " (W)", line=dict(color=col)), 2, 1)
    ER = np.atleast_2d(res.ER_z)
    for k in range(ER.shape[0]):
        fig.add_trace(go.Scatter(x=z, y=ER[k] * 1e6, name=f"Raman order {k+1} (uJ)",
                                 line=dict(color=C_R, dash=["solid", "dash", "dot", "dashdot"][k % 4])), 2, 2)
    fig.update_layout(**{**LAYOUT, "height": 620})
    fig.update_xaxes(title_text="z (m)", row=2)
    return fig


def trace_figure(res):
    tr = res.trace
    t = tr["t"] * 1e6
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=t, y=tr["nE_mean"], name="mean Er inversion", line=dict(color=C_OUT)), secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=tr["ase_f"], name="forward ASE (W)", line=dict(color=C_AF)), secondary_y=True)
    fig.add_trace(go.Scatter(x=t, y=tr["ase_b"], name="backward ASE (W)", line=dict(color=C_AB)), secondary_y=True)
    fig.add_trace(go.Scatter(x=t, y=tr["pump_abs"], name="pump absorbed (W)", line=dict(color=C_P, dash="dot")), secondary_y=True)
    fig.update_layout(**LAYOUT, xaxis_title="time after the pulse (us)")
    fig.update_yaxes(title_text="inversion", secondary_y=False)
    fig.update_yaxes(title_text="power (W)", secondary_y=True)
    return fig


def xs_figure(fiber):
    fig = go.Figure()
    for xs, lab in ((fiber.xs_er, "Er"), (fiber.xs_yb, "Yb")):
        if xs is None:
            continue
        fig.add_trace(go.Scatter(x=xs.lam * 1e9, y=xs.sa, name=f"{lab} absorption"))
        fig.add_trace(go.Scatter(x=xs.lam * 1e9, y=xs.se, name=f"{lab} emission", line=dict(dash="dash")))
    fig.update_layout(**LAYOUT, xaxis_title="wavelength (nm)", yaxis_title="cross section (m^2)")
    fig.update_yaxes(type="log", range=[-28, -23])
    return fig


def gain_spectrum_figure(fiber, n_levels=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    lam = np.linspace(1480e-9, 1640e-9, 400)
    fig = go.Figure()
    for n in n_levels:
        g = fiber.small_signal_coeffs(lam, n, n) * 4.342944819 * fiber.L
        fig.add_trace(go.Scatter(x=lam * 1e9, y=g, name=f"inversion {n:.1f}"))
    fig.update_layout(**LAYOUT, xaxis_title="wavelength (nm)", yaxis_title="single-pass gain, whole fibre (dB)")
    return fig


def qswitch_pulse_figure(p):
    fig = go.Figure()
    if p.get("ok"):
        fig.add_trace(go.Scatter(x=_dec(p["t"] * 1e9), y=_dec(p["P"] / 1e3), name="output", line=dict(color=C_OUT)))
    fig.update_layout(**LAYOUT, xaxis_title="time after the switch opens (ns)", yaxis_title="power (kW)")
    return fig


def qswitch_map_figure(Ls, Rs, FW, NR, mark=None):
    fig = go.Figure()
    fig.add_trace(go.Contour(x=Ls, y=Rs, z=np.log10(FW * 1e9), colorscale="Viridis", name="FWHM",
                             colorbar=dict(title="FWHM (ns)", tickvals=[0, 1, 2, 3],
                                           ticktext=["1", "10", "100", "1000"]),
                             contours=dict(showlabels=False)))
    fig.add_trace(go.Contour(x=Ls, y=Rs, z=NR, showscale=False, contours=dict(start=3, end=3, size=1,
                             coloring="none", showlabels=True), line=dict(color="red", width=2, dash="dash"),
                             name="3 round trips"))
    if mark:
        fig.add_trace(go.Scatter(x=[mark[0]], y=[mark[1]], mode="markers", marker=dict(size=12, color="white",
                                 line=dict(color="black", width=2)), name="current design"))
    fig.update_layout(**LAYOUT, xaxis_title="active fibre length (m)", yaxis_title="output coupler R",
                      title="Degnan estimate; right of / below the red line the output spikes")
    return fig


# ------------------------------------------------------------------ v2 figures
def osa_figure(res, rbw_nm=0.1, end="output", floor_dB=-70, components=True, span=None):
    """What an OSA with resolution rbw_nm would show at the output (or input) end, in dBm per RBW."""
    g, tot, parts = res.osa(rbw_nm, end=end, span=span)
    ref = max(float(tot.max()), 1e-300)
    fl = ref * 10 ** (floor_dB / 10)
    dbm = lambda v: 10 * np.log10(np.maximum(v, fl)) + 30
    fig = go.Figure()
    names = dict(signal_out="signal (pulse window)", ase_fwd="forward ASE", raman="Raman Stokes",
                 ase_bwd="backward ASE", sbs_bwd="SBS Stokes")
    cols = dict(signal_out=C_OUT, ase_fwd=C_AF, raman=C_R, ase_bwd=C_AB, sbs_bwd="#b8860b")
    fig.add_trace(go.Scatter(x=g, y=dbm(tot), name="total", line=dict(color="#e8710a", width=2)))
    if components:
        for k, v in parts.items():
            if v.max() > fl:
                fig.add_trace(go.Scatter(x=g, y=dbm(v), name=names.get(k, k),
                                         line=dict(color=cols.get(k), width=1, dash="dot")))
    fig.update_layout(**LAYOUT, xaxis_title="wavelength (nm)",
                      yaxis_title=f"average power (dBm / {rbw_nm:g} nm)",
                      title=f"{'output' if end == 'output' else 'input'} end, RBW {rbw_nm:g} nm")
    return fig


def _heat(z, x, M, xlab, zlab, title, log=False, floor_dB=-40):
    M = np.asarray(M, float)
    if log:
        M = 10 * np.log10(np.maximum(M / max(M.max(), 1e-300), 10 ** (floor_dB / 10)))
    fig = go.Figure(go.Heatmap(x=x, y=z, z=M, colorscale="Viridis",
                               colorbar=dict(title="dB" if log else "")))
    fig.update_layout(**{**LAYOUT, "height": 460}, xaxis_title=xlab, yaxis_title=zlab, title=title)
    return fig


def pulse_evolution_figure(res, log=False):
    P = res.pulse["snap_P"]
    t = res.pulse["snap_t"] * 1e9
    sl = _trim(t, P[0], P[-1], frac=1e-3)
    Pn = P[:, sl] / np.maximum(P[:, sl].max(axis=1, keepdims=True), 1e-300) if not log else P[:, sl]
    return _heat(res.pulse["snap_z"], t[sl], Pn, "time (ns)", "z (m)",
                 "pulse power along the fibre" + (" (dB)" if log else " (normalised per position)"), log=log)


def spectrum_evolution_figure(res, floor_dB=-50):
    S = res.pulse["snap_S"].astype(float)
    lam = res.pulse["snap_lam"] * 1e9
    o = np.argsort(lam)
    S = S[:, o] / np.maximum(S[:, o].max(axis=1, keepdims=True), 1e-300)
    sl = _trim(lam[o], S[0], S[-1], frac=10 ** (floor_dB / 10), pad=0.3)
    return _heat(res.pulse["snap_z"], lam[o][sl], S[:, sl], "wavelength (nm)", "z (m)",
                 "signal-window spectrum along the fibre (dB, normalised per position)", log=True, floor_dB=floor_dB)


def ase_evolution_figure(res, direction=+1, floor_dB=-40):
    b = res.bands
    m = (b.kind == "ase") & (b.direction == direction)
    lam = b.lam[m] * 1e9
    o = np.argsort(lam)
    psd = res.avg["P_z"][:, m][:, o] / (b.dlam[m][o] * 1e9)
    return _heat(res.z_nodes, lam[o], psd, "wavelength (nm)", "z (m)",
                 f"{'forward' if direction > 0 else 'backward'} ASE spectral density along the fibre (dB rel. max)",
                 log=True, floor_dB=floor_dB)


def ring_inversion_figure(res):
    f = res.fiber
    rc = 0.5 * (f.r_edges[1:] + f.r_edges[:-1]) * 1e6
    fig = make_subplots(rows=1, cols=2, subplot_titles=("before the pulse", "after the pulse"), shared_yaxes=True)
    for i, st in enumerate((res.st_pre, res.st_post)):
        exc = (f.N_s * st.n + f.N_p * st.x) / f.N_er
        fig.add_trace(go.Heatmap(x=rc, y=res.z_cells, z=exc, zmin=0, zmax=1, colorscale="Viridis",
                                 showscale=i == 1, colorbar=dict(title="excited Er")), 1, i + 1)
    fig.update_xaxes(title_text="radius (um)")
    fig.update_yaxes(title_text="z (m)", col=1)
    fig.update_layout(**{**LAYOUT, "height": 460})
    return fig


def mode_energy_figure(res):
    Em = res.pulse["Em_z"]
    fig = go.Figure()
    tot = Em.sum(axis=1)
    for k, nm_ in enumerate(res.pulse["mode_names"]):
        fig.add_trace(go.Scatter(x=res.z_nodes, y=Em[:, k] / np.maximum(tot, 1e-300) * 100, name=nm_))
    fig.update_layout(**LAYOUT, xaxis_title="z (m)", yaxis_title="share of pulse energy (%)",
                      title="modal content along the fibre")
    return fig


def sbs_figure(res):
    s = res.sbs
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if s is None:
        return fig
    Po = np.abs(res.A_out) ** 2
    fig.add_trace(go.Scatter(x=_dec(res.t * 1e9), y=_dec(Po / 1e3), name="output pulse (kW)",
                             line=dict(color=C_OUT)), secondary_y=False)
    fig.add_trace(go.Scatter(x=_dec(s["t"] * 1e9), y=_dec(s["P"]), name="SBS Stokes leaving the input end (W)",
                             line=dict(color="#b8860b")), secondary_y=True)
    ttl = (f"SBS RUNAWAY: effective gain exponent {s['G_eff']:.0f} (threshold ~21), not simulated" if s["runaway"] else
           f"SBS: {s['E']*1e6:.3g} uJ reflected; gain exponent {s['G_eff']:.1f} with linewidth relief "
           f"({s['G_B']:.0f} for a narrowband pulse)")
    fig.update_layout(**LAYOUT, xaxis_title="time at the input end (ns; the pulse is shown as launched)", title=ttl)
    fig.update_yaxes(title_text="pulse (kW)", secondary_y=False)
    fig.update_yaxes(title_text="Stokes (W)", secondary_y=True)
    return fig


def thermal_figure(res):
    th = res.thermal
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=th["z"], y=th["heat_W_m"], name="heat load (W/m)", line=dict(color=C_P)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=th["z"], y=th["dT_core"], name="core temperature rise (K)",
                             line=dict(color=C_OUT, dash="dash")), secondary_y=True)
    fig.update_layout(**LAYOUT, xaxis_title="z (m)", title="heat and core temperature (average over the period)")
    fig.update_yaxes(title_text="W/m", secondary_y=False)
    fig.update_yaxes(title_text="K", secondary_y=True)
    return fig
