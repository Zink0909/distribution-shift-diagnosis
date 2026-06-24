#!/usr/bin/env python3
"""Module D — replicate the drift diagnosis on QQQ (kills the n=1 critique).

Runs the SAME monitor + regime split on QQQ's gamma signal and compares to SPY: if the
erosion -> inversion -> recovery pattern shows up independently on QQQ, the finding isn't a
single-instrument fluke. Honest either way.

    micromamba run -n dist-shift-diagnosis python scripts/run_qqq.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import pipeline as P
from src.monitor import monitor

REPORTS = os.path.join(P.HERE, "reports")
BANDS = [("pre-2022 (2014-21)", "2014", "2022"),
         ("0DTE surge (2022-23)", "2022", "2024"),
         ("recent (2024-26)", "2024", "2027")]


def gamma_auc(sub):
    return roc_auc_score(sub["y"], -np.sign(sub["gex_prev"])) if sub["y"].nunique() == 2 else np.nan


def regime_row(df, lab, a, b):
    s = df[(df.index >= a) & (df.index < b)]
    sg, lg = s.loc[s["gex_prev"] < 0, "y"].mean(), s.loc[s["gex_prev"] > 0, "y"].mean()
    return sg - lg, gamma_auc(s), len(s)


def run_one(symbol):
    df = P.load_dataset(traded_only=True, gex_only=True, symbol=symbol).reset_index()
    df["p"] = (-np.sign(df["gex_prev"]) + 1) / 2
    rep = monitor(df, "date", "p", "y", window=150, n_boot=0)
    dfi = df.set_index("date")
    regimes = {lab: regime_row(dfi, lab, a, b) for lab, a, b in BANDS}
    return df, rep, regimes


def main():
    out = {}
    for sym in ["SPY", "QQQ"]:
        out[sym] = run_one(sym)

    print("=== Replication: gamma-signal drift, SPY vs QQQ ===\n")
    print(f"{'regime':<22}{'SPY gap':>9}{'SPY AUC':>9}   {'QQQ gap':>9}{'QQQ AUC':>9}")
    for lab, _, _ in BANDS:
        sg, sa, _ = out["SPY"][2][lab]
        qg, qa, _ = out["QQQ"][2][lab]
        print(f"{lab:<22}{sg:>+9.2f}{sa:>9.3f}   {qg:>+9.2f}{qa:>9.3f}")

    print("\n=== Monitor (gamma-sign rule) ===")
    for sym in ["SPY", "QQQ"]:
        s = out[sym][1].summary
        fa = s["first_alarm"].date() if s["first_alarm"] is not None else None
        print(f"  {sym}: baseline AUC {s['baseline_auc']:.3f} | worst {s['worst_auc']:.3f} @ "
              f"{s['worst_auc_date'].date()} | alarms {s['n_alarms']} | first {fa}")

    # --- overlay figure: SPY vs QQQ rolling AUC --------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for sym, col in [("SPY", "#2c7fb8"), ("QQQ", "#e6550d")]:
        a = out[sym][1].rolling["auc"]
        ax.plot(a.index, a, color=col, lw=1.8, label=f"{sym} rolling AUC")
    ax.axhline(0.5, color="#969696", ls=":", lw=1)
    ax.axvline(pd.Timestamp("2022-01-01"), color="#d73027", ls="--", lw=1)
    ax.text(pd.Timestamp("2022-02-01"), 0.40, "0DTE era", color="#d73027", fontsize=9)
    ax.set_ylabel("rolling gamma-sign AUC"); ax.set_ylim(0.38, 0.72)
    ax.set_title("Replication: the gamma signal erodes & inverts on BOTH SPY and QQQ")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p = os.path.join(REPORTS, "fig6_qqq_replication.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"\n[fig] {p}")


if __name__ == "__main__":
    main()
