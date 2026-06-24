"""Reusable pieces for the distribution-shift case study.

Everything here operates on the small daily table (`data/dataset.csv`) only — no dependency on
the upstream study. Design choices that matter for *trustworthy* evaluation:

- **Leakage-free**: every feature is lagged to the prior close (built upstream); models are only
  ever fit on data strictly before the day they predict (expanding-window walk-forward).
- **Honest uncertainty**: the sample is small and autocorrelated, so AUC confidence intervals
  use a moving-block bootstrap rather than an i.i.d. one.
- **Drift localization**: features are split into a `gamma` group (the signal under study) and a
  `base` group, so we can measure the gamma feature's *incremental* power and watch it decay.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(HERE, "data", "dataset.csv")

GAMMA_FEATURES = ["gex_prev", "gex_prev_absmag"]
BASE_FEATURES = ["rvol_prev", "ret_prev", "absret_prev", "range_prev",
                 "mom5_prev", "vol5_prev", "vol20_prev", "dow", "month"]
ALL_FEATURES = BASE_FEATURES + GAMMA_FEATURES


def load_dataset(traded_only: bool = True, gex_only: bool = True, symbol: str = "SPY") -> pd.DataFrame:
    """Load the daily table. Default to the alpha question: traded days with a gamma reading."""
    path = DATASET if symbol.upper() == "SPY" else os.path.join(HERE, "data", f"dataset_{symbol.lower()}.csv")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if traded_only:
        df = df[df["traded"] == 1]
    if gex_only:
        df = df[df["gex_prev"].notna()]
    df = df.dropna(subset=ALL_FEATURES + ["y"])
    df["y"] = df["y"].astype(int)
    return df


def make_model() -> Pipeline:
    """Regularized logistic regression — the honest choice for a small, noisy sample."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000)),
    ])


def walk_forward_oof(df: pd.DataFrame, features: list[str],
                     min_train: int = 250, step: int = 40) -> pd.Series:
    """Expanding-window walk-forward out-of-fold probabilities (leakage-free).

    Fit on all rows strictly before each test block, predict the block, slide forward. Returns a
    probability series aligned to the dates that ever sat in a test block.
    """
    X, y = df[features].to_numpy(float), df["y"].to_numpy(int)
    n = len(df)
    oof = np.full(n, np.nan)
    start = min_train
    while start < n:
        end = min(start + step, n)
        if y[:start].min() != y[:start].max():          # need both classes to fit
            m = make_model().fit(X[:start], y[:start])
            oof[start:end] = m.predict_proba(X[start:end])[:, 1]
        start = end
    return pd.Series(oof, index=df.index, name="p")


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> pd.Series:
    """Train on `train`, score `test` — for an explicit pre/post split."""
    m = make_model().fit(train[features].to_numpy(float), train["y"].to_numpy(int))
    return pd.Series(m.predict_proba(test[features].to_numpy(float))[:, 1],
                     index=test.index, name="p")


def auc(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    ok = ~np.isnan(p)
    return roc_auc_score(y[ok], p[ok]) if ok.sum() > 10 and len(set(y[ok])) == 2 else np.nan


def block_bootstrap_auc_ci(y, p, n_boot: int = 2000, block: int = 10, seed: int = 0):
    """Moving-block bootstrap CI for AUC — respects autocorrelation in the daily series."""
    y, p = np.asarray(y), np.asarray(p)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    n = len(y)
    if n < 30 or len(set(y)) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    aucs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        yb, pb = y[idx], p[idx]
        if len(set(yb)) == 2:
            aucs.append(roc_auc_score(yb, pb))
    return (float(np.percentile(aucs, 5)), float(np.percentile(aucs, 95))) if aucs else (np.nan, np.nan)


def score_block(y, p) -> dict:
    """Full scorecard on one set of predictions."""
    y, p = np.asarray(y), np.asarray(p)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    lo, hi = block_bootstrap_auc_ci(y, p)
    return {"n": len(y), "base_rate": float(y.mean()), "auc": auc(y, p),
            "auc_lo": lo, "auc_hi": hi, "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1])}
