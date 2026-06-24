"""Module E — interactive model-reliability monitoring dashboard.

Wraps the model-agnostic monitor (module A) in a Streamlit UI: pick the model/score under test
and the detector settings, and watch rolling discrimination, drift alarms, and per-feature
signal decay over time. This is the system's "front end" — monitoring is the natural product
form of a reliability project, so the UI isn't decorative.

    micromamba run -n dist-shift-diagnosis streamlit run app.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import pipeline as P
from src.monitor import monitor

st.set_page_config(page_title="Model-Reliability Monitor", layout="wide")


@st.cache_data
def load(symbol):
    return P.load_dataset(traded_only=True, gex_only=True, symbol=symbol).reset_index()


@st.cache_data
def run_monitor(symbol, score_source, window, k_sd, h_sd, band):
    df = load(symbol).copy()
    if score_source == "Gamma-sign rule":
        df["score"] = (-np.sign(df["gex_prev"]) + 1) / 2
    else:  # trained logistic (walk-forward OOF probabilities)
        oof = P.walk_forward_oof(load(symbol).set_index("date"), P.ALL_FEATURES)
        df["score"] = oof.to_numpy()
    rep = monitor(df, time_col="date", score_col="score", label_col="y",
                  feature_cols=["gex_prev", "rvol_prev", "ret_prev"], window=window,
                  n_boot=150 if band else 0, cusum_k_sd=k_sd, cusum_h_sd=h_sd)
    return rep


# ---------------------------------------------------------------------------- sidebar
st.sidebar.title("Monitor settings")
instrument = st.sidebar.selectbox("Instrument", ["SPY", "QQQ"], index=0)
score_source = st.sidebar.radio("Model under test", ["Gamma-sign rule", "Trained logistic"])
window = st.sidebar.slider("Rolling window (traded days)", 80, 250, 150, 10)
k_sd = st.sidebar.slider("CUSUM slack (×σ)", 0.0, 3.0, 1.5, 0.1,
                         help="Higher = ignore more noise before alarming")
h_sd = st.sidebar.slider("CUSUM threshold (×σ)", 2.0, 10.0, 5.0, 0.5,
                         help="Higher = only flag deeper / more sustained drops")
band = st.sidebar.checkbox("Show bootstrap uncertainty band (slower)", value=False)

# ---------------------------------------------------------------------------- header
st.title("🔎 Model-Reliability Monitor")
st.caption("Detecting when a deployed predictive model's discrimination drifts — and dating the "
           "break. Case: a dealer-gamma signal for an intraday strategy, through the 2022 0DTE shift.")

rep = run_monitor(instrument, score_source, window, k_sd, h_sd, band)
s = rep.summary

# ---------------------------------------------------------------------------- KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Overall AUC", f"{s['overall_auc']:.3f}")
c2.metric("Baseline AUC", f"{s['baseline_auc']:.3f}")
c3.metric("Worst AUC", f"{s['worst_auc']:.3f}",
          delta=f"{s['worst_auc'] - s['baseline_auc']:+.3f}", delta_color="inverse")
c4.metric("Drift alarms", f"{s['n_alarms']}")
c5.metric("First alarm", s["first_alarm"].date().isoformat() if s["first_alarm"] is not None else "—")

if s["n_alarms"]:
    st.warning(f"⚠️ Drift alarm: discrimination degraded from baseline {s['baseline_auc']:.3f}; "
               f"first flagged **{s['first_alarm'].date()}**, trough {s['worst_auc']:.3f} on "
               f"**{s['worst_auc_date'].date()}**. (Onset precedes the 2022 inversion — an early warning.)")
else:
    st.success("No sustained discrimination drift detected at these settings.")

# ---------------------------------------------------------------------------- main chart
st.subheader("Rolling discrimination + drift alarms")
fig, ax = plt.subplots(figsize=(11, 4.4))
rep.plot(ax=ax)
st.pyplot(fig)

# ---------------------------------------------------------------------------- two panels
left, right = st.columns(2)
with left:
    st.subheader("Per-feature signal over time")
    sig_cols = [c for c in rep.rolling.columns if c.startswith("sig_")]
    if sig_cols:
        st.line_chart(rep.rolling[sig_cols].rename(columns=lambda c: c[4:]))
        st.caption("Rolling |rank-corr| of each feature with the outcome. The gamma feature's "
                   "signal fades while base features hold — the drift is feature-specific.")
with right:
    st.subheader("Drift alarms")
    if len(rep.alarms):
        st.dataframe(rep.alarms[["date", "statistic", "baseline", "note"]], use_container_width=True)
    else:
        st.write("— none at current settings —")

with st.expander("How to read this / honest caveats"):
    st.markdown(
        "- **AUC** = ranking/discrimination of the model's score; 0.5 = no better than chance, "
        "**below 0.5 = inverted** (anti-predictive).\n"
        "- The detector is a **CUSUM** on rolling AUC vs an in-control **baseline** (first windows). "
        "It is tuned conservatively: minor dips are *not* flagged.\n"
        "- **Honest reading of this case:** discrimination erodes from ~2020 and *inverts* in "
        "2022–23 as 0DTE options surge, then partially recovers. 0DTE coincides with the inversion "
        "phase but the erosion began earlier — so this is an **early-warning** story, not a clean "
        "single-cause break.\n"
        "- Effect sizes are modest (AUC ~0.5–0.6) and the sample is small; the value is the "
        "diagnosis, not a strong predictor.")
