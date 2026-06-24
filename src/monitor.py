"""Module A — the model-reliability monitor (the system's core, MODEL-AGNOSTIC).

This is the locked interface every other module plugs into. It knows nothing about gamma,
finance, or any specific model: feed it a time-ordered stream of (timestamp, predicted score,
true label, optional features) and it returns a `DriftReport` — rolling discrimination with
uncertainty bands, rolling calibration error, per-feature rolling signal, and timestamped
change-point ALARMS on the model's discrimination.

    from src.monitor import monitor
    report = monitor(df, time_col="date", score_col="p", label_col="y",
                     feature_cols=["x1", "x2"], window=150)
    report.summary            # dict overview
    report.alarms             # DataFrame of timestamped drift alarms
    report.rolling            # DataFrame time series (AUC band, calib, per-feature signal)
    report.plot()             # the monitoring chart
    report.to_frame()         # == report.rolling

DESIGN NOTE (the contract — do not break without bumping all downstream modules):
- INPUT  : one tidy row per observation; `score_col` is a higher-is-more-likely-positive score
           (probability or any monotone score — AUC only needs the ranking); `label_col` is 0/1.
- OUTPUT : the `DriftReport` dataclass below. Columns/fields named here are the stable API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# small self-contained stats (kept here so the monitor is a standalone module)
# --------------------------------------------------------------------------- #
def _auc(y: np.ndarray, s: np.ndarray) -> float:
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def _auc_ci(y: np.ndarray, s: np.ndarray, n_boot: int, block: int, rng) -> tuple[float, float]:
    """Moving-block bootstrap CI for AUC (respects autocorrelation). n_boot<=0 -> skip (fast)."""
    n = len(y)
    if n_boot <= 0 or n < 2 * block or len(np.unique(y)) < 2:
        return (np.nan, np.nan)
    n_blocks = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(st, st + block) for st in starts])[:n]
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) == 2:
            out.append(roc_auc_score(yb, sb))
    return (float(np.percentile(out, 5)), float(np.percentile(out, 95))) if out else (np.nan, np.nan)


def _cusum_down(series: np.ndarray, baseline: float, k: float, h: float, rearm_at: float):
    """One-sided CUSUM detecting *sustained downward* shifts below `baseline`.

    Emits one change-point per downward episode: after firing it stays "alarmed" (silent) until
    the series recovers back to `rearm_at`, then re-arms. This gives discrete change-points
    instead of an alarm every window. `k` = slack (deviations below k don't accumulate),
    `h` = decision threshold.
    """
    S, hits, alarmed = 0.0, [], False
    for i, v in enumerate(series):
        if not np.isfinite(v):
            continue
        S = max(0.0, S + (baseline - v - k))
        if not alarmed and S > h:
            hits.append(i); alarmed = True; S = 0.0
        elif alarmed and v >= rearm_at:        # recovered to in-control -> re-arm
            alarmed = False; S = 0.0
    return hits


# --------------------------------------------------------------------------- #
# the report object (stable output API)
# --------------------------------------------------------------------------- #
@dataclass
class DriftReport:
    rolling: pd.DataFrame          # index = window-end timestamp; cols: n, auc, auc_lo, auc_hi,
                                   #   calib_err, sig_<feature> ...
    alarms: pd.DataFrame           # cols: date, signal, statistic, baseline, note
    summary: dict                  # overall/baseline AUC, n_alarms, first_alarm, worst ...
    meta: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return self.rolling

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4.4))
        r = self.rolling
        ax.plot(r.index, r["auc"], color="#2c7fb8", lw=1.8, label="rolling AUC")
        if {"auc_lo", "auc_hi"}.issubset(r.columns):
            ax.fill_between(r.index, r["auc_lo"], r["auc_hi"], color="#2c7fb8", alpha=0.15)
        ax.axhline(0.5, color="#969696", ls=":", lw=1)
        if np.isfinite(self.summary.get("baseline_auc", np.nan)):
            ax.axhline(self.summary["baseline_auc"], color="#444", ls="--", lw=0.8, label="baseline")
        for d in self.alarms["date"]:
            ax.axvline(d, color="#d73027", ls="--", lw=1.1)
        if len(self.alarms):
            ax.axvline(self.alarms["date"].iloc[0], color="#d73027", ls="--", lw=1.1, label="drift alarm")
        ax.set_ylabel("rolling AUC"); ax.set_title("Model-reliability monitor")
        ax.legend(frameon=False, fontsize=9)
        return ax


# --------------------------------------------------------------------------- #
# THE INTERFACE (locked) — module A entry point
# --------------------------------------------------------------------------- #
def monitor(df: pd.DataFrame,
            time_col: str,
            score_col: str,
            label_col: str,
            feature_cols: list[str] | None = None,
            window: int = 150,
            min_class: int = 5,
            n_boot: int = 150,
            block: int = 10,
            baseline_n: int = 60,
            cusum_k_sd: float = 1.5,     # slack ~1.5 baseline-sigma: ignore noise dips
            cusum_h_sd: float = 5.0,     # threshold ~5 baseline-sigma: only sustained drops

            seed: int = 0) -> DriftReport:
    """Run the reliability monitor on a time-ordered prediction stream.

    Parameters
    ----------
    df            tidy table; one row per observation.
    time_col      timestamp column (used only to order + label the output).
    score_col     model score, higher = more likely positive (prob or any monotone score).
    label_col     binary outcome 0/1.
    feature_cols  optional features to track for per-feature signal drift.
    window        rolling window length, in number of observations.
    n_boot,block  moving-block bootstrap settings for the AUC band.
    baseline_n    number of initial rolling points that define the in-control baseline.
    cusum_*_sd    CUSUM slack/threshold as multiples of the baseline AUC std.

    Returns a `DriftReport`.
    """
    d = df[[time_col, score_col, label_col] + (feature_cols or [])].dropna(
        subset=[time_col, score_col, label_col]).copy()
    d = d.sort_values(time_col).reset_index(drop=True)
    t = pd.DatetimeIndex(d[time_col])
    y = d[label_col].to_numpy(int)
    s = d[score_col].to_numpy(float)
    feats = {f: d[f].to_numpy(float) for f in (feature_cols or [])}
    rng = np.random.default_rng(seed)
    n = len(d)

    rows = []
    for i in range(window, n + 1):
        a, b = i - window, i
        yy, ss = y[a:b], s[a:b]
        if min(np.bincount(yy, minlength=2)[:2]) < min_class:
            continue
        lo, hi = _auc_ci(yy, ss, n_boot, block, rng)
        rec = {"date": t[b - 1], "n": b - a, "auc": _auc(yy, ss),
               "auc_lo": lo, "auc_hi": hi,
               "calib_err": abs(ss.mean() - yy.mean())}        # calibration-in-the-large
        for f, arr in feats.items():
            ff = arr[a:b]
            # rank-correlation strength of feature vs label (fast, robust on small windows)
            rec[f"sig_{f}"] = abs(pd.Series(ff).corr(pd.Series(yy), method="spearman"))
        rows.append(rec)

    rolling = pd.DataFrame(rows).set_index("date")

    # --- change-point alarms on the rolling AUC ---------------------------------------------
    auc_series = rolling["auc"].to_numpy(float)
    base_vals = auc_series[:baseline_n]
    base_vals = base_vals[np.isfinite(base_vals)]
    baseline = float(np.mean(base_vals)) if len(base_vals) else np.nan
    sd = float(np.std(base_vals)) if len(base_vals) > 1 else np.nan
    sd = sd if (np.isfinite(sd) and sd > 1e-6) else 0.02
    k = cusum_k_sd * sd
    hit_idx = _cusum_down(auc_series, baseline, k, cusum_h_sd * sd, rearm_at=baseline - k) \
        if np.isfinite(baseline) else []
    alarms = pd.DataFrame([{
        "date": rolling.index[i], "signal": "auc_drop",
        "statistic": float(auc_series[i]), "baseline": baseline,
        "note": f"rolling AUC fell to {auc_series[i]:.3f} vs baseline {baseline:.3f}",
    } for i in hit_idx])
    if alarms.empty:
        alarms = pd.DataFrame(columns=["date", "signal", "statistic", "baseline", "note"])

    summary = {
        "n_obs": n, "n_windows": len(rolling), "window": window,
        "overall_auc": _auc(y, s), "baseline_auc": baseline,
        "worst_auc": float(np.nanmin(auc_series)) if len(auc_series) else np.nan,
        "worst_auc_date": (rolling.index[int(np.nanargmin(auc_series))]
                           if len(auc_series) and np.isfinite(np.nanmin(auc_series)) else None),
        "n_alarms": len(alarms),
        "first_alarm": (alarms["date"].iloc[0] if len(alarms) else None),
    }
    meta = {"time_col": time_col, "score_col": score_col, "label_col": label_col,
            "feature_cols": feature_cols or [], "window": window}
    return DriftReport(rolling=rolling, alarms=alarms, summary=summary, meta=meta)
