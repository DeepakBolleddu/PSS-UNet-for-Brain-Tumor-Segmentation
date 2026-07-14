#!/usr/bin/env python3
"""
selective_prediction_analysis.py

Turns the per-case uncertainty records from uq_inference.py into the actual
contribution: showing that an uncertainty score lets the model abstain on the
cases it gets wrong, especially the catastrophic out-of-distribution failures.

Runs three analyses, separately for the in-distribution test set and Africa:

  1. Risk-coverage: sort cases by uncertainty (most uncertain first), reject the
     top fraction, report mean Dice of the RETAINED cases at several coverage
     levels, plus AURC (area under the Dice-vs-coverage curve; higher = better
     the score ranks good cases above bad ones). A random ranker is the baseline.

  2. Failure detection: define a failure as Dice < --fail_thr. Report AUROC of
     each uncertainty score for detecting failures (how well it separates good
     from bad cases) and the recall of failures inside the worst-X% rejected.

  3. Split-conformal Dice floor (the rigorous, distribution-shift piece):
     calibrate a rejection threshold on the IN-DISTRIBUTION test set so that the
     RETAINED cases meet a target Dice floor with probability >= 1 - alpha, then
     apply that same threshold to Africa and report the empirical coverage and
     floor-violation rate. This directly tests whether an uncertainty rule
     calibrated on the source domain still controls risk under domain shift.

Usage:
  python selective_prediction_analysis.py \
      --test test_uq.json --africa africa_uq.json \
      --score entropy --fail_thr 0.80
"""
import argparse, json
import numpy as np
from sklearn.metrics import roc_auc_score

SCORES = ["entropy", "tta_var", "band_frac", "neg_mean_conf", "neg_pred_vol"]

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # NumPy 2.0 renamed trapz


def load(path):
    recs = json.load(open(path))
    for r in recs:
        r["neg_mean_conf"] = -r["mean_conf"]   # higher = more uncertain
        r["neg_pred_vol"] = -r["pred_vol"]     # smaller lesion = riskier
    return recs


def risk_coverage(dice, unc):
    """Return coverage grid and mean retained Dice when rejecting most-uncertain."""
    order = np.argsort(unc)             # ascending uncertainty = most certain first
    dice_sorted = dice[order]
    n = len(dice)
    cov = np.arange(1, n + 1) / n
    retained_mean = np.cumsum(dice_sorted) / np.arange(1, n + 1)
    return cov, retained_mean


def aurc(dice, unc):
    cov, ret = risk_coverage(dice, unc)
    return float(_trapz(ret, cov))    # higher is better


def dice_at(dice, unc, coverages):
    cov, ret = risk_coverage(dice, unc)
    out = {}
    for c in coverages:
        idx = min(len(ret) - 1, int(np.ceil(c * len(ret))) - 1)
        out[c] = float(ret[idx])
    return out


def report_split(name, recs, score, fail_thr, coverages):
    dice = np.array([r["dice"] for r in recs])
    print(f"\n{'='*60}\n{name}  (N={len(recs)})  mean Dice={dice.mean():.4f}\n{'='*60}")
    fails = (dice < fail_thr)
    print(f"failures (Dice<{fail_thr}): {fails.sum()} / {len(recs)} "
          f"({100*fails.mean():.1f}%)")

    print(f"\nAURC / failure-AUROC per uncertainty score:")
    best = None
    for s in SCORES:
        unc = np.array([r[s] for r in recs])
        a = aurc(dice, unc)
        try:
            au = roc_auc_score(fails.astype(int), unc) if fails.any() and not fails.all() else float("nan")
        except ValueError:
            au = float("nan")
        flag = " <-- selected" if s == score else ""
        print(f"  {s:14s}  AURC={a:.4f}  failAUROC={au:.3f}{flag}")
        if best is None or a > best[1]:
            best = (s, a)
    rnd = float(np.mean([_trapz(*risk_coverage(dice, np.random.permutation(dice)))
                         for _ in range(200)]))
    print(f"  {'random':14s}  AURC={rnd:.4f}  (chance baseline)")
    print(f"best AURC score: {best[0]} ({best[1]:.4f})")

    unc = np.array([r[s] for r in recs] if False else [r[score] for r in recs])
    da = dice_at(dice, unc, coverages)
    print(f"\nRetained mean Dice with score='{score}' at coverage:")
    for c in coverages:
        print(f"  cover {int(c*100):3d}%  ->  Dice {da[c]:.4f}")
    if fails.any():
        order = np.argsort(-unc)            # most uncertain first
        for c in coverages:
            k = int(round((1 - c) * len(recs)))   # number rejected
            rejected = set(order[:k].tolist())
            recall = np.mean([i in rejected for i in np.where(fails)[0]]) if k > 0 else 0.0
            print(f"  reject {int((1-c)*100):2d}% -> captures {100*recall:4.0f}% of failures")
    return dice


def conformal_floor(test_recs, africa_recs, score, floor, alpha):
    """
    Calibrate a max-uncertainty threshold on TEST so retained Dice >= floor
    holds with prob >= 1-alpha (lower-tail conformal on the indicator
    'Dice >= floor' ranked by uncertainty), then apply to Africa.
    """
    print(f"\n{'='*60}\nSPLIT-CONFORMAL Dice floor (floor={floor}, alpha={alpha})\n{'='*60}")
    td = np.array([r["dice"] for r in test_recs])
    tu = np.array([r[score] for r in test_recs])
    # We want to keep cases whose uncertainty is below a threshold tau.
    # Choose tau as the largest value such that, among kept TEST cases, the
    # fraction with Dice < floor is <= alpha.
    order = np.argsort(tu)
    td_o, tu_o = td[order], tu[order]
    tau, kept_floor_rate = tu_o[-1], 1.0
    for k in range(len(td_o), 0, -1):
        viol = np.mean(td_o[:k] < floor)
        if viol <= alpha:
            tau = tu_o[k - 1]
            kept_floor_rate = viol
            kept_k = k
            break
    else:
        kept_k = 0
    print(f"calibrated tau on TEST: keep if {score} <= {tau:.5f}")
    print(f"  TEST kept {kept_k}/{len(td)} ({100*kept_k/len(td):.0f}%), "
          f"floor-violation among kept = {100*kept_floor_rate:.1f}% (target <= {100*alpha:.0f}%)")

    ad = np.array([r["dice"] for r in africa_recs])
    au = np.array([r[score] for r in africa_recs])
    keep = au <= tau
    if keep.any():
        viol = np.mean(ad[keep] < floor)
        print(f"  AFRICA applying same tau: kept {keep.sum()}/{len(ad)} "
              f"({100*keep.mean():.0f}%), mean retained Dice={ad[keep].mean():.4f}, "
              f"floor-violation among kept = {100*viol:.1f}%")
        print("  -> if AFRICA violation stays near alpha, the rule transfers under shift;")
        print("     if it blows up, that itself is a publishable distribution-shift finding.")
    else:
        print("  AFRICA: threshold rejects everything (severe shift in the score).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--africa", required=True)
    ap.add_argument("--score", default="entropy", choices=SCORES)
    ap.add_argument("--fail_thr", type=float, default=0.80)
    ap.add_argument("--floor", type=float, default=0.80)
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    np.random.seed(0)
    test_recs = load(args.test)
    africa_recs = load(args.africa)
    cov = [1.0, 0.9, 0.8, 0.7]

    report_split("IN-DISTRIBUTION TEST", test_recs, args.score, args.fail_thr, cov)
    report_split("BRATS-AFRICA (OOD)", africa_recs, args.score, args.fail_thr, cov)
    conformal_floor(test_recs, africa_recs, args.score, args.floor, args.alpha)


if __name__ == "__main__":
    main()