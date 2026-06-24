#!/usr/bin/env python3
"""Figures for the writeup: the decay of the gamma feature's discriminating power over time,
and pre/post calibration. Saves PNGs to reports/.

    micromamba run -n dist-shift-diagnosis python scripts/make_figures.py
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
from src import pipeline as P                                                   # noqa: E402

REPORTS = os.path.join(P.HERE, "reports")
os.makedirs(REPORTS, exist_ok=True)
BLUE, GREY, RED = "#2c7fb8", "#969696", "#d73027"


def winrate_gap(sub):
    sg, lg = sub.loc[sub["gex_prev"] < 0, "y"], sub.loc[sub["gex_prev"] > 0, "y"]
    if len(sg) < 5 or len(lg) < 5:
        return np.nan
    return sg.mean() - lg.mean()


def gamma_auc(sub):
    if sub["y"].nunique() < 2:
        return np.nan
    return roc_auc_score(sub["y"], -np.sign(sub["gex_prev"]))


def fig_winrate_by_year(df):
    yrs = sorted(df.index.year.unique())
    sg, lg = [], []
    for y in yrs:
        s = df[df.index.year == y]
        sg.append(s.loc[s["gex_prev"] < 0, "y"].mean())
        lg.append(s.loc[s["gex_prev"] > 0, "y"].mean())
    x = np.arange(len(yrs))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.2, sg, 0.4, label="short-gamma days (GEX<0)", color=BLUE)
    ax.bar(x + 0.2, lg, 0.4, label="long-gamma days (GEX>0)", color=GREY)
    ax.axvline(yrs.index(2022) - 0.5, color=RED, ls="--", lw=1)
    ax.text(yrs.index(2022) - 0.45, 0.62, "0DTE era", color=RED, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(yrs, rotation=45, ha="right")
    ax.set_ylabel("short-leg win rate"); ax.set_ylim(0, 0.78)
    ax.set_title("Win rate by gamma regime — a reliable edge before 2022, erratic after")
    ax.legend(frameon=False, fontsize=9); fig.tight_layout()
    p = os.path.join(REPORTS, "fig1_winrate_by_year.png"); fig.savefig(p, dpi=120); plt.close(fig)
    return p


def fig_rolling_decay(df, W=150):
    end_dates, aucs, gaps = [], [], []
    for i in range(W, len(df) + 1):
        sub = df.iloc[i - W:i]
        end_dates.append(sub.index[-1]); aucs.append(gamma_auc(sub)); gaps.append(winrate_gap(sub))
    end_dates = pd.DatetimeIndex(end_dates)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(end_dates, aucs, color=BLUE, lw=1.8, label="gamma-sign rule AUC")
    ax.axhline(0.5, color=GREY, lw=1, ls=":")
    ax.axvline(pd.Timestamp("2022-01-01"), color=RED, ls="--", lw=1)
    ax.text(pd.Timestamp("2022-02-01"), 0.40, "0DTE era", color=RED, fontsize=9)
    ax.set_ylabel("rolling AUC (gamma-sign rule)", color=BLUE); ax.set_ylim(0.38, 0.72)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax.twinx()
    ax2.plot(end_dates, gaps, color="#444", lw=1.2, alpha=0.7, label="win-rate gap (short−long γ)")
    ax2.axhline(0.0, color="#444", lw=0.6, ls=":")
    ax2.set_ylabel("win-rate gap (short−long γ)", color="#444"); ax2.set_ylim(-0.15, 0.35)
    ax.set_title(f"Gamma's discrimination breaks down in the 0DTE surge (2022–23),\n"
                 f"then only partially recovers (rolling {W} traded days)")
    fig.tight_layout()
    p = os.path.join(REPORTS, "fig2_rolling_decay.png"); fig.savefig(p, dpi=120); plt.close(fig)
    pd.DataFrame({"date": end_dates, "rolling_auc": aucs, "winrate_gap": gaps}).to_csv(
        os.path.join(P.HERE, "data", "rolling_metrics.csv"), index=False)
    return p


def fig_calibration():
    oof = pd.read_csv(os.path.join(P.HERE, "data", "oof_predictions.csv"))
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1)
    for era, col, lab in [("pre", BLUE, "pre-2022 (OOF)"), ("post", RED, "2022+ (deployed)")]:
        s = oof[(oof["era"] == era) & oof["p_full"].notna()]
        q = pd.qcut(s["p_full"], 5, duplicates="drop")
        g = s.groupby(q, observed=True).agg(pred=("p_full", "mean"), obs=("y", "mean"))
        ax.plot(g["pred"], g["obs"], "o-", color=col, lw=1.6, label=lab)
    ax.set_xlabel("predicted win probability"); ax.set_ylabel("observed win rate")
    ax.set_title("Calibration of base+gamma model"); ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8); fig.tight_layout()
    p = os.path.join(REPORTS, "fig3_calibration.png"); fig.savefig(p, dpi=120); plt.close(fig)
    return p


def three_regime_summary(df):
    """The honest structure the rolling view reveals: strong -> broken -> partial recovery."""
    bands = [("pre-2022 (2014-21)", "2014", "2022"),
             ("0DTE surge (2022-23)", "2022", "2024"),
             ("recent (2024-26)", "2024", "2027")]
    print(f"\n{'regime':<22}{'n':>5}{'win|short-γ':>13}{'win|long-γ':>12}{'gap':>8}{'γ-sign AUC':>12}")
    for lab, a, b in bands:
        s = df[(df.index >= a) & (df.index < b)]
        sg, lg = s.loc[s['gex_prev'] < 0, 'y'].mean(), s.loc[s['gex_prev'] > 0, 'y'].mean()
        print(f"{lab:<22}{len(s):>5}{sg:>13.2f}{lg:>12.2f}{sg - lg:>+8.2f}{gamma_auc(s):>12.3f}")


def main():
    df = P.load_dataset(traded_only=True, gex_only=True)
    print("[fig]", fig_winrate_by_year(df))
    print("[fig]", fig_rolling_decay(df))
    print("[fig]", fig_calibration())
    three_regime_summary(df)


if __name__ == "__main__":
    main()
