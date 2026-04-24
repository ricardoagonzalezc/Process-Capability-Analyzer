import streamlit as st
import pandas as pd
import numpy as np
from spc.metrics import calculate_metrics, get_verdict
from spc.charts import build_histogram

st.set_page_config(page_title="Process Capability Analyzer", layout="wide")
st.title("⚙️ Process Capability Analyzer")
st.markdown("Upload your process data or use the built-in sample to calculate Cp, Cpk, Pp, and Ppk.")

st.sidebar.header("1. Data Source")
data_source = st.sidebar.radio("Choose data source:", ["Use Sample Data", "Upload CSV"])

df = None

if data_source == "Use Sample Data":
    df = pd.read_csv("data/sample_data.csv")
    st.sidebar.success("Sample data loaded (200 measurements)")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"{len(df)} rows loaded")

st.sidebar.header("2. Spec Limits")
usl = st.sidebar.number_input("Upper Spec Limit (USL)", value=10.15)
lsl = st.sidebar.number_input("Lower Spec Limit (LSL)", value=9.85)

if df is not None:
    col = df.columns[0]
    data = df[col].dropna().values

    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    if usl <= lsl:
        st.error("USL must be greater than LSL. Please fix the spec limits in the sidebar.")
    else:
        metrics = calculate_metrics(data, usl, lsl)
        verdict, icon = get_verdict(metrics["Cpk"])

        if verdict == "PASS":
            st.success(f"{icon} Process is CAPABLE — Cpk = {metrics['Cpk']}")
        elif verdict == "WARNING":
            st.warning(f"{icon} Marginal capability — Cpk = {metrics['Cpk']} (target ≥ 1.33)")
        else:
            st.error(f"{icon} Process is NOT CAPABLE — Cpk = {metrics['Cpk']}")

        st.subheader("📊 Capability Metrics")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Cp",      metrics["Cp"])
        c2.metric("Cpk",     metrics["Cpk"])
        c3.metric("Pp",      metrics["Pp"])
        c4.metric("Ppk",     metrics["Ppk"])
        c5.metric("Mean",    metrics["Mean"])
        c6.metric("Std Dev", metrics["Std Dev"])

        st.subheader("📈 Distribution Chart")
        fig = build_histogram(data.tolist(), usl, lsl)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Select a data source in the sidebar to get started.")