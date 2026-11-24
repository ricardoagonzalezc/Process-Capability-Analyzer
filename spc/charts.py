import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

import theme
from theme import AMBER, SKY, MINT, CORAL, LAVENDER, INK, INK_DIM, LINE_STRONG
from spc.metrics import qq_points

# Fixed roles so colours mean the same thing on every chart:
#   data = sky · within/short-term = amber · overall/long-term = lavender
#   alarm points = coral diamonds · mean = mint


def build_histogram(data, usl, lsl, m):
    data = np.asarray(data, dtype=float)
    mean, s_w, s_o = m["mean"], m["sigma_within"], m["sigma_overall"]
    pad = 0.5 * (usl - lsl)
    lo, hi = min(data.min(), lsl) - 0.15 * pad, max(data.max(), usl) + 0.15 * pad
    x = np.linspace(lo, hi, 400)

    fig = go.Figure()
    fig.add_vrect(x0=lo, x1=lsl, fillcolor=CORAL, opacity=0.10, line_width=0, layer="below")
    fig.add_vrect(x0=usl, x1=hi, fillcolor=CORAL, opacity=0.10, line_width=0, layer="below")
    fig.add_trace(go.Histogram(
        x=data, histnorm="probability density", name="Measurements",
        marker=dict(color=SKY, line=dict(color=theme.PANEL, width=2)), opacity=0.55,
        hovertemplate="%{x}<br>density %{y:.2f}<extra></extra>"))
    if np.isfinite(s_w):
        fig.add_trace(go.Scatter(x=x, y=norm.pdf(x, mean, s_w), mode="lines",
                                 name="Within σ (Cp/Cpk)", line=dict(color=AMBER, width=2),
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=norm.pdf(x, mean, s_o), mode="lines",
                             name="Overall σ (Pp/Ppk)", line=dict(color=LAVENDER, width=2, dash="dash"),
                             hoverinfo="skip"))
    for v, name in ((lsl, "LSL"), (usl, "USL")):
        fig.add_vline(x=v, line=dict(color=INK, width=1.5, dash="dash"),
                      annotation_text=name, annotation_font=dict(family=theme.FONT_MONO, color=INK))
    fig.add_vline(x=mean, line=dict(color=MINT, width=1.5, dash="dot"),
                  annotation_text="Mean", annotation_position="bottom",
                  annotation_font=dict(family=theme.FONT_MONO, color=MINT))
    fig.update_layout(title="Distribution vs. specification limits", height=430,
                      xaxis_title="Measurement", yaxis_title="Density", xaxis_range=[lo, hi])
    return fig


def index_chart(m, target):
    rows = [("Pp", m["Pp"], LAVENDER), ("Ppk", m["Ppk"], LAVENDER),
            ("Cp", m["Cp"], AMBER), ("Cpk", m["Cpk"], AMBER)]
    fig = go.Figure()
    for name, idx, col in rows:
        if not np.isfinite(idx.value):
            continue
        fig.add_trace(go.Scatter(
            x=[idx.value], y=[name], mode="markers", showlegend=False,
            marker=dict(color=col, size=12, line=dict(color=theme.PANEL, width=2)),
            error_x=dict(type="data", symmetric=False, array=[idx.hi - idx.value],
                         arrayminus=[idx.value - idx.lo], color=col, thickness=2, width=7),
            hovertemplate=f"{name} = {idx.value:.2f}<br>{m['conf']:.0%} CI: {idx.lo:.2f} – {idx.hi:.2f}<extra></extra>"))
    fig.add_vline(x=target, line=dict(color=INK, width=1.5, dash="dash"),
                  annotation_text=f"target {target:g}", annotation_position="top",
                  annotation_font=dict(family=theme.FONT_MONO, color=INK))
    fig.add_vline(x=1.0, line=dict(color=INK_DIM, width=1, dash="dot"))
    fig.update_layout(title=f"Capability indices with {m['conf']:.0%} confidence intervals", height=300,
                      xaxis_title="Index value", yaxis=dict(categoryorder="array",
                      categoryarray=["Pp", "Ppk", "Cp", "Cpk"]), margin=dict(l=64, r=24, t=64, b=48))
    return fig


def control_chart_figure(c):
    """I-MR (n=1) or Xbar-R (n>1) with alarms (coral diamonds) and pattern signals (amber rings)."""
    idx = np.arange(1, len(c.x) + 1)
    top = "Individual value" if c.kind == "I-MR" else "Subgroup mean"
    bot = "Moving range" if c.kind == "I-MR" else "Subgroup range"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        row_heights=[0.62, 0.38])

    fig.add_trace(go.Scatter(x=idx, y=c.x, mode="lines+markers", name=top,
                             line=dict(color=SKY, width=1.5), marker=dict(color=SKY, size=5),
                             hovertemplate="#%{x}: %{y:.4f}<extra></extra>"), row=1, col=1)
    sig = [i for i in c.signals if i not in c.alarms]
    if sig:
        fig.add_trace(go.Scatter(x=idx[sig], y=c.x[sig], mode="markers", name="Pattern signal",
                                 marker=dict(color="rgba(0,0,0,0)", size=11,
                                             line=dict(color=AMBER, width=2)),
                                 hovertemplate="#%{x}: %{y:.4f} (pattern)<extra></extra>"), row=1, col=1)
    if c.alarms:
        fig.add_trace(go.Scatter(x=idx[c.alarms], y=c.x[c.alarms], mode="markers", name="Beyond limits",
                                 marker=dict(color=CORAL, size=11, symbol="diamond",
                                             line=dict(color=theme.PANEL, width=1.5)),
                                 hovertemplate="#%{x}: %{y:.4f} (beyond limits)<extra></extra>"), row=1, col=1)
    for y, nm, dash in ((c.x_ucl, "UCL", "dash"), (c.x_cl, "CL", "solid"), (c.x_lcl, "LCL", "dash")):
        fig.add_hline(y=y, row=1, col=1, line=dict(color=INK_DIM, width=1, dash=dash),
                      annotation_text=nm, annotation_position="right",
                      annotation_font=dict(family=theme.FONT_MONO, size=10, color=INK_DIM))

    d_idx = np.arange(2, len(c.disp) + 2) if c.kind == "I-MR" else idx
    fig.add_trace(go.Scatter(x=d_idx, y=c.disp, mode="lines+markers", name=bot,
                             line=dict(color=LAVENDER, width=1.5), marker=dict(color=LAVENDER, size=4),
                             hovertemplate="#%{x}: %{y:.4f}<extra></extra>"), row=2, col=1)
    if c.disp_alarms:
        da = np.array(c.disp_alarms)
        fig.add_trace(go.Scatter(x=d_idx[da], y=c.disp[da], mode="markers", name="Beyond limits",
                                 showlegend=not c.alarms,
                                 marker=dict(color=CORAL, size=11, symbol="diamond",
                                             line=dict(color=theme.PANEL, width=1.5)),
                                 hovertemplate="#%{x}: %{y:.4f} (beyond limits)<extra></extra>"), row=2, col=1)
    for y, nm, dash in ((c.disp_ucl, "UCL", "dash"), (c.disp_cl, "CL", "solid"), (c.disp_lcl, "LCL", "dash")):
        if y > 0 or nm == "CL":
            fig.add_hline(y=y, row=2, col=1, line=dict(color=INK_DIM, width=1, dash=dash),
                          annotation_text=nm, annotation_position="right",
                          annotation_font=dict(family=theme.FONT_MONO, size=10, color=INK_DIM))
    fig.update_yaxes(title_text=top, row=1, col=1)
    fig.update_yaxes(title_text=bot, row=2, col=1)
    fig.update_xaxes(title_text="Observation" if c.kind == "I-MR" else "Subgroup", row=2, col=1)
    fig.update_layout(title=f"{c.kind} control chart", height=560, margin=dict(l=64, r=56, t=64, b=48))
    return fig


def qq_figure(data, m):
    q, y, slope, intercept = qq_points(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q, y=y, mode="markers", name="Data",
                             marker=dict(color=SKY, size=6, line=dict(color=theme.PANEL, width=1)),
                             hovertemplate="z %{x:.2f}<br>value %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[q.min(), q.max()], y=[intercept + slope * q.min(), intercept + slope * q.max()],
                             mode="lines", name="Normal reference", line=dict(color=AMBER, width=2)))
    fig.update_layout(title=f"Normal Q-Q plot (Anderson-Darling p = {m['normality']['ad_p']:.3f})",
                      height=420, xaxis_title="Theoretical quantile (z)", yaxis_title="Sample quantile")
    return fig
