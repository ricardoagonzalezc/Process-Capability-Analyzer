import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

def build_histogram(data, usl, lsl):
    mean = np.mean(data)
    std  = np.std(data, ddof=1)

    x_range = np.linspace(min(data) - 3*std, max(data) + 3*std, 300)
    y_curve  = norm.pdf(x_range, mean, std)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data,
        histnorm='probability density',
        name='Measurements',
        marker_color='steelblue',
        opacity=0.6
    ))

    fig.add_trace(go.Scatter(
        x=x_range, y=y_curve,
        mode='lines',
        name='Normal Curve',
        line=dict(color='navy', width=2)
    ))

    fig.add_vline(x=usl, line_color="red",   line_dash="dash", annotation_text="USL")
    fig.add_vline(x=lsl, line_color="red",   line_dash="dash", annotation_text="LSL")
    fig.add_vline(x=mean, line_color="green", line_dash="dot",  annotation_text="Mean")

    fig.update_layout(
        title="Process Distribution vs. Spec Limits",
        xaxis_title="Measurement",
        yaxis_title="Density",
        legend=dict(x=0.01, y=0.99),
        bargap=0.05
    )

    return fig
