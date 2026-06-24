#!/usr/bin/env python3
"""Module C — external-data attribution: does the discrimination decay track 0DTE adoption?

Overlays a real (public, Cboe-sourced) 0DTE-share-of-SPX-volume series on the monitor's rolling
AUC, and quantifies the association — honestly. The point is NOT to "prove" 0DTE caused it; it's
to test the mechanism and surface the nuance the rolling view already implied: the erosion begins
~2020 (while 0DTE is still low), but the *inversion* (AUC < 0.5) lands in 2022-23 exactly as 0DTE
surges. So 0DTE coincides with / deepens the inversion rather than being the sole cause.

0DTE data: `data/zdte_share.csv`. Anchored on Cboe figures (2016 ≈ 5%; Tue/Thu SPX expiries added
2022; mid-2023 ≈ 43%; 2024 ≈ 48%; 2025 monthly records 56-62%); intermediate years interpolated
and labeled as such. Annual granularity is sufficient for an overlay (spec allows this).
Sources: cboe.com/insights (0DTE share posts), cboe.com/markets/us/options/market-statistics.

    micromamba run -n dist-shift-diagnosis python scripts/run_attribution.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import pipeline as P
from src.monitor import monitor

REPORTS = os.path.join(P.HERE, "reports")
BLUE, RED = "#2c7fb8", "#d73027"


def zdte_daily(index: pd.DatetimeIndex) -> pd.Series:
    """Annual 0DTE share -> a daily series aligned to `index` (placed mid-year, interpolated)."""
    z = pd.read_csv(os.path.join(P.HERE, "data", "zdte_share.csv"))
    anchor = pd.Series(z["zdte_share_pct"].values,
                       index=pd.to_datetime(z["year"].astype(str) + "-07-01"))
    full = anchor.reindex(anchor.index.union(index)).interpolate("time").ffill().bfill()
    return full.reindex(index)


def main():
    df = P.load_dataset(traded_only=True, gex_only=True).reset_index()
    df["p"] = (-np.sign(df["gex_prev"]) + 1) / 2
    rep = monitor(df, time_col="date", score_col="p", label_col="y", window=150, n_boot=0)

    auc = rep.rolling["auc"].dropna()
    z = zdte_daily(auc.index)
    J = pd.concat([auc.rename("auc"), z.rename("zdte")], axis=1).dropna()

    def sp(a, b):
        r, p = stats.spearmanr(a, b)
        return r, p

    r_all, p_all = sp(J["auc"], J["zdte"])
    pre = J[J.index < "2022-01-01"]
    post = J[J.index >= "2022-01-01"]
    r_pre, _ = sp(pre["auc"], pre["zdte"])
    r_post, _ = sp(post["auc"], post["zdte"])

    print("=== 0DTE attribution (rolling AUC vs 0DTE share) ===")
    print(f"  Spearman corr, full overlap   : {r_all:+.2f}  (p={p_all:.1e}, n={len(J)})")
    print(f"  Spearman corr, pre-2022        : {r_pre:+.2f}")
    print(f"  Spearman corr, 2022+           : {r_post:+.2f}")
    print()
    inv = J[J["auc"] < 0.5]
    rec = post[post.index > inv.index.max()]   # after the last sub-0.5 window = the recovery
    print("  Honest decomposition (why -0.72 must NOT be read as clean causation):")
    print(f"    onset  : AUC {pre['auc'].iloc[0]:.2f}->{pre['auc'].iloc[-1]:.2f} over "
          f"{pre.index.min().year}-{pre.index.max().year} while 0DTE only "
          f"{pre['zdte'].iloc[0]:.0f}%->{pre['zdte'].iloc[-1]:.0f}%  -> erosion STARTS before the 0DTE surge")
    print(f"    inversion: AUC<0.5 from {inv.index.min().date()}, 0DTE then "
          f"{J.loc[inv.index.min(),'zdte']:.0f}%+  -> worst phase COINCIDES with the 0DTE surge")
    if len(rec):
        print(f"    recovery : AUC {inv['auc'].min():.2f}->{rec['auc'].iloc[-1]:.2f} while 0DTE keeps "
              f"rising to {rec['zdte'].iloc[-1]:.0f}%  -> 0DTE canNOT explain the rebound (post-2022 corr {r_post:+.2f})")
    print("  Verdict: 0DTE is coincident with the inversion, NOT a proven monotone cause "
          "(it explains neither the 2020 onset nor the 2024+ recovery).")

    # --- figure: overlay -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(J.index, J["auc"], color=BLUE, lw=1.8, label="rolling AUC (gamma signal)")
    ax.axhline(0.5, color="#969696", ls=":", lw=1)
    ax.set_ylabel("rolling AUC", color=BLUE); ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_ylim(0.40, 0.72)
    ax2 = ax.twinx()
    ax2.plot(J.index, J["zdte"], color=RED, lw=1.6, ls="--", label="0DTE share of SPX volume")
    ax2.set_ylabel("0DTE share of SPX options volume (%)", color=RED); ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(0, 70)
    ax.axvline(pd.Timestamp("2020-07-01"), color="#444", ls="-", lw=0.8)
    ax.text(pd.Timestamp("2020-08-01"), 0.70, "erosion onset\n(0DTE still ~19%)", fontsize=8, color="#444")
    ax.set_title(f"Discrimination decay vs 0DTE adoption  (Spearman {r_all:+.2f})")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    out = os.path.join(REPORTS, "fig5_0dte_attribution.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n[fig] {out}")


if __name__ == "__main__":
    main()
