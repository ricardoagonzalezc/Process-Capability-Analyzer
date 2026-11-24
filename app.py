import numpy as np
import pandas as pd
import streamlit as st

import theme
from spc.metrics import calculate_metrics, get_verdict, MIN_RECOMMENDED_N
from spc.charts import build_histogram, index_chart, control_chart_figure, qq_figure

st.set_page_config(page_title="Process Capability Analyzer", layout="wide")
theme.apply_theme()
theme.header(
    "Process Capability Analyzer",
    "Cp, Cpk, Pp and Ppk with confidence intervals, a control-chart stability check and "
    "normality tests — upload your measurements or explore the built-in sample.",
    eyebrow="Project · Statistical process control",
)


def f(v, d=2):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def ppm_fmt(v):
    return "—" if not np.isfinite(v) else (f"{v:,.0f}" if v >= 1 else f"{v:.2f}")


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("1. Data source")
data_source = st.sidebar.radio("Choose data source:", ["Use sample data", "Upload CSV"],
                               label_visibility="collapsed")

df = None
if data_source == "Use sample data":
    df = pd.read_csv("data/sample_data.csv")
    st.sidebar.success(f"Sample data loaded ({len(df)} measurements)")
else:
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"{len(df)} rows loaded")

numeric_cols = [] if df is None else list(df.select_dtypes("number").columns)
col = None
if df is not None:
    if not numeric_cols:
        st.error("No numeric column found in this file.")
        st.stop()
    col = numeric_cols[0] if len(numeric_cols) == 1 else st.sidebar.selectbox("Measurement column", numeric_cols)

st.sidebar.header("2. Specification limits")
usl = st.sidebar.number_input("Upper spec limit (USL)", value=10.15, format="%.4f")
lsl = st.sidebar.number_input("Lower spec limit (LSL)", value=9.85, format="%.4f")

st.sidebar.header("3. Analysis settings")
subgroup = st.sidebar.slider(
    "Subgroup size", 1, 10, 1,
    help="1 = individual measurements (moving-range sigma). 2-10 = consecutive rows grouped "
         "into subgroups (range-based sigma). Use the size your sampling plan actually collects.")
conf = st.sidebar.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, format_func=lambda c: f"{c:.0%}")
target = st.sidebar.selectbox("Cpk target", [1.00, 1.33, 1.67, 2.00], index=1,
                              help="1.33 is the common minimum for an established process; "
                                   "1.67 is typical for new processes / safety-critical features.")

# ── Main ─────────────────────────────────────────────────────────────────────
if df is None:
    st.info("Select a data source to get started.")
    st.stop()

data = df[col].dropna().to_numpy(dtype=float)

if usl <= lsl:
    st.error("USL must be greater than LSL. Please fix the spec limits in the sidebar.")
    st.stop()
if len(data) < 2 * subgroup + 1 or len(data) < 10:
    st.error(f"Not enough data: need at least {max(10, 2 * subgroup + 1)} measurements for these settings.")
    st.stop()
if np.ptp(data) == 0:
    st.error("All measurements are identical, so process variation cannot be estimated.")
    st.stop()

m = calculate_metrics(data, usl, lsl, subgroup_size=subgroup, conf=conf)
status, icon, msg = get_verdict(m["Cpk"], target)
{"PASS": st.success, "WARNING": st.warning}.get(status, st.error)(f"{icon} {msg}")

if m["n"] < MIN_RECOMMENDED_N:
    st.warning(f"Only {m['n']} measurements: capability estimates are very uncertain below "
               f"~{MIN_RECOMMENDED_N}. Look at the confidence intervals, not just the point values.")
_ch = m["chart"]
_expected = 0.0027 * (len(_ch.x) + len(_ch.disp))          # chance alarms for an in-control process
if _ch.n_alarms > max(2, 3 * _expected):
    st.warning(f"⚠️ The process does not look statistically stable ({_ch.n_alarms} points beyond control "
               f"limits; ~{_expected:.1f} expected by chance). Cp/Cpk assume a stable process — treat them "
               f"with caution and see the Stability tab.")

# KPI tiles
cl = f"{m['conf']:.0%} CI"
tiles = [
    ("Cp", f(m["Cp"].value), f"{cl} {f(m['Cp'].lo)}–{f(m['Cp'].hi)}", ""),
    ("Cpk", f(m["Cpk"].value), f"{cl} {f(m['Cpk'].lo)}–{f(m['Cpk'].hi)}",
     {"PASS": "ok", "WARNING": "warn"}.get(status, "bad")),
    ("Pp", f(m["Pp"].value), f"{cl} {f(m['Pp'].lo)}–{f(m['Pp'].hi)}", ""),
    ("Ppk", f(m["Ppk"].value), f"{cl} {f(m['Ppk'].lo)}–{f(m['Ppk'].hi)}", ""),
    ("Sigma level", f(m["z_bench"]), "Z bench · overall σ", ""),
    ("Expected PPM", ppm_fmt(sum(m["ppm_overall"])), f"observed {ppm_fmt(sum(m['ppm_observed']))}", ""),
]
cols = st.columns(len(tiles))
for c, (k, v, s, stt) in zip(cols, tiles):
    c.markdown(theme.tile(k, v, s, stt), unsafe_allow_html=True)
st.write("")

tab_cap, tab_stab, tab_norm, tab_detail = st.tabs(["Capability", "Stability", "Normality", "Details"])

# ── Capability ───────────────────────────────────────────────────────────────
with tab_cap:
    st.plotly_chart(build_histogram(data, usl, lsl, m), width="stretch")
    st.plotly_chart(index_chart(m, target), width="stretch")

    theme.label("What this says")
    mid = (usl + lsl) / 2
    offset = (m["mean"] - mid) / (usl - lsl) * 100
    bullets = []
    gap_c = m["Cp"].value - m["Cpk"].value
    if abs(gap_c) < 0.05:
        bullets.append(f"**Centering:** the process is well centred (mean {m['mean']:.4f}, "
                       f"{offset:+.1f}% of the tolerance from mid-spec), so Cp ≈ Cpk.")
    else:
        bullets.append(f"**Centering:** the mean sits {offset:+.1f}% of the tolerance away from mid-spec. "
                       f"Recentring alone could lift Cpk from {m['Cpk'].value:.2f} up toward Cp "
                       f"({m['Cp'].value:.2f}).")
    gap_p = m["Cpk"].value - m["Ppk"].value
    if gap_p > 0.15:
        bullets.append(f"**Short vs long term:** Cpk exceeds Ppk by {gap_p:.2f}, meaning the process "
                       f"drifts or shifts over time. Its short-term potential is better than what customers "
                       f"actually receive.")
    else:
        bullets.append("**Short vs long term:** Cpk and Ppk agree closely, so there is little drift "
                       "between subgroups.")
    lo_ppm, hi_ppm = m["ppm_overall"]
    side = "lower" if lo_ppm > hi_ppm else "upper"
    bullets.append(f"**Defects:** about {ppm_fmt(lo_ppm + hi_ppm)} PPM expected out of spec "
                   f"(mostly past the {side} limit), against {m['n_out_of_spec']} observed in {m['n']} measurements.")
    if m["Cpk"].lo < target <= m["Cpk"].value:
        bullets.append("**Sample size:** the point estimate clears the target but the interval does not; "
                       "collecting more data would tighten it.")
    st.markdown("\n".join(f"- {b}" for b in bullets))

# ── Stability ────────────────────────────────────────────────────────────────
with tab_stab:
    ch = m["chart"]
    st.plotly_chart(control_chart_figure(ch), width="stretch")
    expected = 0.0027 * (len(ch.x) + len(ch.disp))
    if ch.stable:
        st.success("✅ No points beyond the control limits — no evidence of special-cause variation.")
    else:
        where = " and ".join(x for x in (
            f"{len(ch.alarms)} on the {'individuals' if ch.kind == 'I-MR' else 'mean'} chart" if ch.alarms else "",
            f"{len(ch.disp_alarms)} on the {'moving-range' if ch.kind == 'I-MR' else 'range'} chart" if ch.disp_alarms else "") if x)
        st.warning(f"⚠️ {ch.n_alarms} point(s) beyond control limits ({where}). About {expected:.1f} would be "
                   f"expected by chance alone across these charts, so a lone point may be noise — check "
                   f"whether it has an assignable cause before relying on capability numbers.")
    if ch.signals:
        st.info(f"{len(ch.signals)} point(s) are part of a run or trend pattern (amber rings). These tests "
                f"produce occasional false alarms in long series, so treat them as prompts to look closer.")
    with st.expander("Rules applied"):
        st.markdown(
            "- **Beyond limits**: a point outside the ±3σ limits on the mean/individuals chart, or above the upper limit of the range chart.\n"
            "- **8 in a row** on the same side of the centre line.\n"
            "- **6 in a row** steadily increasing or decreasing.\n\n"
            "Control limits come from the process's own short-term variation "
            "(moving range ÷ 1.128, or average range ÷ d2), *not* from the spec limits.")

# ── Normality ────────────────────────────────────────────────────────────────
with tab_norm:
    nt = m["normality"]
    left, right = st.columns([3, 2])
    left.plotly_chart(qq_figure(data, m), width="stretch")
    with right:
        theme.label("Tests (H₀: data are normal)")
        st.markdown(theme.tile("Anderson-Darling p", f(nt["ad_p"], 3), f"A² = {f(nt['ad_stat'], 3)}",
                               "ok" if nt["is_normal"] else "bad"), unsafe_allow_html=True)
        st.write("")
        st.markdown(theme.tile("Shapiro-Wilk p", f(nt["sw_p"], 3), f"W = {f(nt['sw_stat'], 3)}",
                               "" if not np.isfinite(nt["sw_p"]) else ("ok" if nt["sw_p"] >= 0.05 else "bad")),
                    unsafe_allow_html=True)
        st.write("")
        st.caption(f"Skewness {nt['skew']:+.2f} · excess kurtosis {nt['excess_kurtosis']:+.2f}")
    if nt["is_normal"]:
        st.success("✅ No evidence against normality (p ≥ 0.05). The Cp/Cpk/Pp/Ppk formulas and "
                   "confidence intervals above are appropriate.")
    else:
        st.warning("⚠️ The data deviate from a normal distribution (p < 0.05), so the normal-based indices, "
                   "PPM estimates and confidence intervals may mislead.")
        c1, c2 = st.columns(2)
        c1.markdown(theme.tile("Pp (percentile method)", f(m["Pp_np"]), "non-parametric"), unsafe_allow_html=True)
        c2.markdown(theme.tile("Ppk (percentile method)", f(m["Ppk_np"]), "uses 0.135 / 50 / 99.865 %"),
                    unsafe_allow_html=True)
        st.caption("The percentile method (ISO 22514) doesn't assume a distribution, but it needs a large "
                   "sample (hundreds of points) to estimate the extreme tails reliably.")

# ── Details ──────────────────────────────────────────────────────────────────
with tab_detail:
    table = pd.DataFrame([
        ("Sample size (N)", m["n"], ""),
        ("Subgroup size", m["subgroup_size"], "1 = individuals"),
        ("Mean", m["mean"], ""),
        ("σ within (short-term)", m["sigma_within"], "MR̄/d2 or R̄/d2"),
        ("σ overall (long-term)", m["sigma_overall"], "sample std dev"),
        ("Cp", m["Cp"].value, f"{cl}: {f(m['Cp'].lo)} – {f(m['Cp'].hi)}"),
        ("Cpl / Cpu", f"{f(m['Cpl'])} / {f(m['Cpu'])}", "one-sided, within σ"),
        ("Cpk", m["Cpk"].value, f"{cl}: {f(m['Cpk'].lo)} – {f(m['Cpk'].hi)}"),
        ("Pp", m["Pp"].value, f"{cl}: {f(m['Pp'].lo)} – {f(m['Pp'].hi)}"),
        ("Ppl / Ppu", f"{f(m['Ppl'])} / {f(m['Ppu'])}", "one-sided, overall σ"),
        ("Ppk", m["Ppk"].value, f"{cl}: {f(m['Ppk'].lo)} – {f(m['Ppk'].hi)}"),
        ("PPM < LSL / > USL (within σ)", f"{ppm_fmt(m['ppm_within'][0])} / {ppm_fmt(m['ppm_within'][1])}", "expected"),
        ("PPM < LSL / > USL (overall σ)", f"{ppm_fmt(m['ppm_overall'][0])} / {ppm_fmt(m['ppm_overall'][1])}", "expected"),
        ("PPM < LSL / > USL (observed)", f"{ppm_fmt(m['ppm_observed'][0])} / {ppm_fmt(m['ppm_observed'][1])}", "actual data"),
        ("Z bench", m["z_bench"], "from overall σ"),
    ], columns=["Statistic", "Value", "Note"])
    table["Value"] = table["Value"].map(lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v))
    st.dataframe(table, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3)
    st.download_button("Download results (CSV)", table.to_csv(index=False), "capability_results.csv", "text/csv")

    with st.expander("Raw data preview"):
        st.dataframe(df.head(50), width="stretch")
    with st.expander("Method notes"):
        st.markdown(
            "- **Cp / Cpk** use short-term σ estimated *within* subgroups; **Pp / Ppk** use the overall "
            "sample standard deviation.\n"
            "- **Confidence intervals**: chi-square interval for Cp/Pp; Bissell's approximation for Cpk/Ppk. "
            "Both assume independent, normally distributed data and use N as the sample size (approximate "
            "for the within-σ indices).\n"
            "- **Normality**: Anderson-Darling (p-value via D'Agostino & Stephens) and Shapiro-Wilk.\n"
            "- **PPM** values assume a normal distribution; compare them with the observed count.\n"
            "- **Verdict** uses the lower confidence bound of Cpk, so a small sample can't earn a pass on luck.")
