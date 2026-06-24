#!/usr/bin/env python3
"""Headline result: a predictive signal that works, then fails out-of-distribution after 2022 —
and a diagnosis that localizes the failure to the dealer-gamma feature (distribution shift),
not the base features and not overfitting.

    micromamba run -n dist-shift-diagnosis python scripts/run_baseline.py
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import pipeline as P                                                   # noqa: E402

SPLIT = "2022-01-01"
OOF_OUT = os.path.join(P.HERE, "data", "oof_predictions.csv")


def show(tag, s):
    print(f"  {tag:<22} n={s['n']:>4}  base={s['base_rate']:.2f}  "
          f"AUC={s['auc']:.3f} [{s['auc_lo']:.2f},{s['auc_hi']:.2f}]  "
          f"PR-AUC={s['pr_auc']:.3f}  Brier={s['brier']:.3f}")


def main():
    df = P.load_dataset(traded_only=True, gex_only=True)
    pre, post = df[df.index < SPLIT], df[df.index >= SPLIT]
    print(f"Target: win vs loss among traded short days (gamma era).")
    print(f"Sample: {len(df)} traded days  | pre-2022 {len(pre)} (win {pre['y'].mean():.2f}) "
          f"| 2022+ {len(post)} (win {post['y'].mean():.2f})\n")

    # --- 1. Does the signal predict, out-of-sample, in the pre-2022 era? --------------------
    print("=== 1. Walk-forward OOF within pre-2022 (does it work at all?) ===")
    oof_pre_base = P.walk_forward_oof(pre, P.BASE_FEATURES)
    oof_pre_full = P.walk_forward_oof(pre, P.ALL_FEATURES)
    show("base features", P.score_block(pre["y"], oof_pre_base))
    show("base + gamma", P.score_block(pre["y"], oof_pre_full))

    # --- 2. Freeze a pre-2022 model, deploy on 2022+ (the failure) -------------------------
    print("\n=== 2. Train on pre-2022, test on 2022+ (deployment) ===")
    p_post_base = P.fit_predict(pre, post, P.BASE_FEATURES)
    p_post_full = P.fit_predict(pre, post, P.ALL_FEATURES)
    show("base features", P.score_block(post["y"], p_post_base))
    show("base + gamma", P.score_block(post["y"], p_post_full))
    print("  -> gamma's incremental AUC: "
          f"pre-2022 (OOF) {P.auc(pre['y'], oof_pre_full) - P.auc(pre['y'], oof_pre_base):+.3f}"
          f"  vs  2022+ {P.auc(post['y'], p_post_full) - P.auc(post['y'], p_post_base):+.3f}")

    # --- 3. Single-rule baseline: the raw gamma sign (ties to the upstream finding) ---------
    print("\n=== 3. Raw gamma-sign rule (short-gamma -> predict win) ===")
    for tag, sub in [("pre-2022", pre), ("2022+", post)]:
        # short-gamma (gex<0) historically the 'good' regime: use -sign as the score
        score = -np.sign(sub["gex_prev"])
        print(f"  {tag:<10} AUC={P.auc(sub['y'], score):.3f}   "
              f"win|short-gamma={sub.loc[sub['gex_prev'] < 0, 'y'].mean():.2f}  "
              f"win|long-gamma={sub.loc[sub['gex_prev'] > 0, 'y'].mean():.2f}")

    # --- 4. Drift localization: mutual information per feature, pre vs post -----------------
    print("\n=== 4. Feature<->target mutual information (bits), pre vs post ===")
    print(f"  {'feature':<16}{'pre-2022':>10}{'2022+':>9}{'change':>9}")
    mi_pre = mutual_info_classif(pre[P.ALL_FEATURES], pre["y"], random_state=0, discrete_features=False)
    mi_post = mutual_info_classif(post[P.ALL_FEATURES], post["y"], random_state=0, discrete_features=False)
    for f, a, b in sorted(zip(P.ALL_FEATURES, mi_pre, mi_post), key=lambda t: -(t[1] - t[2])):
        star = "  <- gamma" if f in P.GAMMA_FEATURES else ""
        print(f"  {f:<16}{a:>10.4f}{b:>9.4f}{b - a:>+9.4f}{star}")

    # --- save OOF (pre walk-forward + post deployment) for later drift plots ----------------
    oof = pd.concat([
        pd.DataFrame({"p_full": oof_pre_full, "p_base": oof_pre_base, "y": pre["y"], "era": "pre"}),
        pd.DataFrame({"p_full": p_post_full, "p_base": p_post_base, "y": post["y"], "era": "post"}),
    ])
    oof.to_csv(OOF_OUT)
    print(f"\n[write] {OOF_OUT}")


if __name__ == "__main__":
    main()
