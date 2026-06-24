#!/usr/bin/env python3
"""Smoke test + module-B demo for the reliability monitor (module A).

Feeds the gamma-sign rule's prediction stream through the model-agnostic `monitor()` and checks
the change-point detector raises a drift alarm around the 2022 0DTE break. Saves the monitor
figure to reports/.

    micromamba run -n dist-shift-diagnosis python scripts/run_monitor.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import pipeline as P          # noqa: E402  (only to load the dataset)
from src.monitor import monitor        # noqa: E402

REPORTS = os.path.join(P.HERE, "reports")


def main():
    df = P.load_dataset(traded_only=True, gex_only=True).reset_index()
    # "model" under test = the gamma-sign rule: short-gamma (gex<0) -> predict win.
    df["p"] = (-np.sign(df["gex_prev"]) + 1) / 2

    rep = monitor(df, time_col="date", score_col="p", label_col="y",
                  feature_cols=["gex_prev", "rvol_prev", "ret_prev"], window=150)

    s = rep.summary
    print("=== monitor summary ===")
    for k in ["n_obs", "n_windows", "overall_auc", "baseline_auc", "worst_auc",
              "worst_auc_date", "n_alarms", "first_alarm"]:
        print(f"  {k:<15}: {s[k]}")
    print("\n=== alarms ===")
    print(rep.alarms[["date", "statistic", "baseline", "note"]].to_string(index=False)
          if len(rep.alarms) else "  (none)")

    # smoke assertions (honest): a time series is produced; the detector raises an early-warning
    # alarm by 2021 (degradation onset precedes the 0DTE inversion); the trough lands in 2022-23.
    assert len(rep.rolling) > 100, "monitor produced too few windows"
    assert len(rep.alarms) >= 1, "expected at least one drift alarm"
    assert rep.summary["first_alarm"].year <= 2021, "onset alarm should be an early warning (<=2021)"
    assert rep.summary["worst_auc_date"].year in (2022, 2023), "trough should be in the 0DTE break"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = rep.plot()
    plt.tight_layout()
    out = os.path.join(REPORTS, "fig4_monitor.png")
    plt.savefig(out, dpi=120)
    print(f"\n[ok] smoke test passed; figure -> {out}")


if __name__ == "__main__":
    main()
