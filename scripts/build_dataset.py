#!/usr/bin/env python3
"""Build the daily modelling table for the distribution-shift case study.

This is the ETL/provenance bridge from the upstream intraday-momentum study. It reads that
repo's cached 1-minute SPY bars + dealer-gamma (GEX) feed, constructs a **leakage-free** daily
table — target known only after the day, every feature known by the *prior* close — and writes
`data/dataset.csv` (small, committed). The ML pipeline downstream depends only on that file.

Target
------
y_t = 1 if the intraday-momentum SHORT leg is profitable on day t  (short_ret_t > 0).

Features (all lagged to the prior close — no look-ahead)
--------
gex_prev          prior-day end-of-day dealer gamma (signed)         [the signal under study]
gex_prev_sign     sign of gex_prev (the classic regime label)
gex_prev_absmag   |gex_prev|
rvol_prev         prior-day intraday realized vol (std of 1-min log returns)
ret_prev          prior-day close-to-close return
absret_prev       |ret_prev|
range_prev        prior-day (high-low)/close
mom5_prev         prior 5-day momentum
vol5_prev         prior 5-day mean of rvol
vol20_prev        prior 20-day mean of rvol
dow, month        calendar (known in advance)

Run (needs the upstream repo's env + cached data):
    micromamba run -n intraday-momentum python scripts/build_dataset.py
"""
import os
import sys
import argparse
import dataclasses
import numpy as np
import pandas as pd

# --- bridge to the upstream study (ETL only; the ML pipeline never imports this) ----------
INTRADAY_REPO = "/Users/mmmm/projects/SequoAlpha/intraday_momentum"
sys.path.insert(0, INTRADAY_REPO)
from intramom import data as datamod          # noqa: E402
from intramom import strategy as strat        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _naive_day(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx).tz_localize(None).normalize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY", help="instrument (needs cached 1-min bars + <SYM>_gex.csv)")
    sym = ap.parse_args().symbol.upper()
    gex_csv = os.path.join(INTRADAY_REPO, "qc", f"{sym}_gex.csv")
    out = os.path.join(HERE, "data", "dataset.csv" if sym == "SPY" else f"dataset_{sym.lower()}.csv")

    # --- raw 1-min bars -> daily aggregates ------------------------------------------------
    df = datamod.get_data("ibkr", sym, "1min", use_cache=True)
    day = df.index.normalize()
    g = df.groupby(day)
    o, h, l, c = g["open"].first(), g["high"].max(), g["low"].min(), g["close"].last()
    rvol = g["close"].apply(lambda s: np.log(s).diff().std())     # intraday realized vol
    for x in (o, h, l, c, rvol):
        x.index = _naive_day(x.index)

    daily = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "rvol": rvol})
    daily["ret"] = daily["close"].pct_change()
    daily["range"] = (daily["high"] - daily["low"]) / daily["close"]
    daily["mom5"] = daily["close"] / daily["close"].shift(5) - 1.0
    daily["vol5"] = daily["rvol"].rolling(5).mean()
    daily["vol20"] = daily["rvol"].rolling(20).mean()

    # --- target: profitable SHORT-leg day --------------------------------------------------
    short_cfg = dataclasses.replace(strat.PRESETS["V3_full"], side="short")
    sret = strat.backtest(df, short_cfg).daily["ret"].copy()
    sret.index = _naive_day(sret.index)

    # --- dealer-gamma feed -----------------------------------------------------------------
    gex = pd.read_csv(gex_csv)
    gex["date"] = pd.to_datetime(gex["date"]).dt.tz_localize(None).dt.normalize()
    gex = gex.set_index("date")["gex_standard"].sort_index()

    # --- assemble (everything lagged to prior close) ---------------------------------------
    F = pd.DataFrame(index=daily.index)
    F["short_ret"] = sret
    # The short leg only acts on ~30% of days; on the rest short_ret == 0 (no trade). The honest
    # classification target is win-vs-loss *among traded days* (the alpha question, where dealer
    # gamma should discriminate) — not profitable-vs-all, which would dump no-trade zeros into the
    # loss class. `traded` is a sample filter, not a feature (it's intraday-determined); every
    # feature below is lagged to the prior close, so the conditional analysis stays leakage-free.
    F["traded"] = (sret != 0).astype("Int64")
    F["y"] = (sret > 0).astype("Int64")          # on traded days this is win(1)/loss(0)
    F["gex_prev"] = gex.reindex(daily.index).shift(1)
    F["gex_prev_sign"] = np.sign(F["gex_prev"])
    F["gex_prev_absmag"] = F["gex_prev"].abs()
    F["rvol_prev"] = daily["rvol"].shift(1)
    F["ret_prev"] = daily["ret"].shift(1)
    F["absret_prev"] = daily["ret"].shift(1).abs()
    F["range_prev"] = daily["range"].shift(1)
    F["mom5_prev"] = daily["mom5"].shift(1)
    F["vol5_prev"] = daily["vol5"].shift(1)
    F["vol20_prev"] = daily["vol20"].shift(1)
    F["dow"] = F.index.dayofweek
    F["month"] = F.index.month

    F = F.dropna(subset=["short_ret"])           # keep all tradable days; GEX may be NaN pre-2014
    F.index.name = "date"

    os.makedirs(os.path.dirname(out), exist_ok=True)
    F.to_csv(out)

    gex_rows = F["gex_prev"].notna().sum()
    print(f"[{sym}] {len(F):,} rows  {F.index.min().date()} -> {F.index.max().date()}")
    print(f"[target ] profitable-short rate: {F['y'].mean():.3f}")
    print(f"[gex    ] rows with gamma feature: {gex_rows:,}  "
          f"({F.loc[F['gex_prev'].notna()].index.min().date()} -> {F.index.max().date()})")
    print(f"[write  ] {out}")


if __name__ == "__main__":
    main()
