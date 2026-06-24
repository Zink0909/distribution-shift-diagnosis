# Model-Reliability Monitor — detecting & diagnosing distribution shift

*A small, model-agnostic system that watches a deployed predictive model's discrimination over
time, **raises a timestamped alarm when it drifts**, localizes the drift to a feature, and tests
candidate causes against external data — demonstrated on a real market signal through the 2022
"0DTE" structural break.*

This started as a question every deployed model eventually forces: **it stopped working — why?**
(overfitting? input drift? did the input→target relationship itself change?) The repo turns that
question into a reusable monitoring tool plus an honest case study.

## What it does

1. **Monitors** — `monitor()` takes any time-ordered `(timestamp, score, label)` stream and returns
   rolling discrimination (AUC with block-bootstrap bands), calibration drift, per-feature signal
   decay, and **CUSUM change-point alarms**.
2. **Detects early** — on the case study it raises its drift alarm in **mid-2020**, ~2 years before
   the signal hit its worst.
3. **Attributes honestly** — overlays a real 0DTE-adoption series and *resists* the easy causal story.
4. **Dashboards it** — a Streamlit front end to explore any model under test interactively.

## Headline finding (the case study)

A dealer-gamma signal predicted an intraday strategy's win/loss with real (if modest)
discrimination — **AUC ≈ 0.60 through 2016–2019** — then **eroded from ~2020, inverted below 0.5
in 2022–23** (the feature briefly became anti-predictive), and **partially recovered** by 2024–26.

Is it the 0DTE options boom? The overall correlation of discrimination vs 0DTE share is a tempting
**−0.72** — but it doesn't survive scrutiny: the erosion **began before** 0DTE took off, and
discrimination **recovered while 0DTE kept climbing** (post-2022 correlation flips to **+0.76**).
**Verdict: 0DTE coincides with the inversion but explains neither the onset nor the recovery** — a
deliberately un-tidy, honest conclusion. The model itself is simple and the effect modest; the
point is the *diagnosis*, done without fooling myself. The same erosion→inversion→recovery shape
**replicates independently on QQQ** (weaker and noisier, but same signs) — so it isn't a one-ticker fluke.

## Quickstart

```bash
micromamba env create -f environment.yml

# (one-time) build the modelling table from the upstream study's cached data
micromamba run -n intraday-momentum python scripts/build_dataset.py

# run the whole analysis pipeline (numbers + figures)
micromamba run -n dist-shift-diagnosis python scripts/run_all.py

# launch the interactive monitor
micromamba run -n dist-shift-diagnosis streamlit run app.py
```

## Use it on your own model

The monitor is model-agnostic — point it at any prediction log:

```python
from src.monitor import monitor
report = monitor(df, time_col="ts", score_col="pred_prob", label_col="outcome",
                 feature_cols=["f1", "f2"], window=150)
report.summary      # overall/baseline/worst AUC, alarms, first-alarm date
report.alarms       # timestamped drift change-points
report.plot()       # the monitoring chart
```

## Layout

```
src/monitor.py        the model-agnostic monitor (module A — the locked core)
src/pipeline.py       leakage-free walk-forward CV, block-bootstrap, feature groups
scripts/build_dataset.py   ETL: upstream study -> data/dataset.csv (leakage-free daily table)
scripts/run_baseline.py    the diagnosis numbers (drift vs overfit/crowding)
scripts/make_figures.py    win-rate-by-year / rolling-decay / calibration figures
scripts/run_monitor.py     monitor smoke test + figure
scripts/run_attribution.py 0DTE external-data attribution (the honest causal test)
scripts/run_qqq.py         independent replication on QQQ (SPY vs QQQ)
scripts/run_all.py         runs the analysis pipeline end-to-end
app.py                Streamlit dashboard (module E)
reports/report.html   the writeup (self-contained); report.md is the source
data/                 dataset.csv (committed, small) + zdte_share.csv; raw 1-min data NOT redistributed
```

## Data provenance

`data/dataset.csv` is derived once (via `scripts/build_dataset.py`) from a separate intraday-
momentum study on real Interactive Brokers 1-minute bars + a dealer-gamma feature from free
QuantConnect options data; the licensed raw data is not redistributed. `data/zdte_share.csv` is a
0DTE-share-of-SPX-volume series anchored on public Cboe figures (2016≈5%, 2023≈43%, 2024≈48%,
2025 monthly 56–62%) with intermediate years interpolated and labeled.

## Honest limitations

Single market, daily, small sample (~700 traded days; a few hundred per regime); AUCs sit in
0.5–0.6 with confidence intervals that brush 0.5. The drift detector applies standard methods
(rolling metrics + CUSUM) — the contribution is the evaluation discipline and the honest
diagnosis, not model complexity or a novel algorithm.
