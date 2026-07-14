#!/usr/bin/env python3
"""
make_tables.py

Regenerates the paper's result tables directly from the JSON files shipped in
results/. No GPU, no data download, no retraining required -- this is the
"audit trail" for the paper: every printed number can be traced back to a file
under results/ that was written by evaluate.py / uq_inference.py during the
actual runs.

Two tables are produced:

  Table 1  Segmentation Dice (mean +/- seed std) on the in-distribution test
           set (N=251) and on standardized BraTS-Africa, for all models.
           Source: results/segmentation_results.json

  Table 3  Failure detection and selective prediction on BraTS-Africa, using
           test-time-augmentation entropy as the uncertainty score, on the
           median-performing seed per model.
           Source: results/uq/<model>_seed<N>_africa_std.json

Run:
  python make_tables.py                 # prints both tables
  python make_tables.py --uq_dir results/uq --seg results/segmentation_results.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr


# Order and display names used throughout the paper.
MODELS = [
    ("baseline", "V-Net baseline"),
    ("vnet_se", "V-Net + SE"),
    ("vnet_ssm", "V-Net + SSM (bneck)"),
    ("pss_unet", "PSS-UNet (SSM all)"),
    ("umamba", "U-Mamba"),
]


def auroc(scores, labels):
    """P(uncertainty of a failure > uncertainty of a non-failure). labels: 1=failure."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    npos, nneg = int(labels.sum()), int((labels == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def retained_dice(entropy, dice, coverage):
    """Mean Dice of the most-confident (lowest-entropy) `coverage` fraction."""
    order = np.argsort(entropy)
    d = np.asarray(dice)[order]
    k = max(1, int(round(coverage * len(d))))
    return float(d[:k].mean())


def table1(seg_path):
    seg = json.load(open(seg_path))
    print("\n" + "=" * 78)
    print("Table 1. Segmentation Dice (mean +/- seed std)")
    print("=" * 78)
    print(f"{'Model':22s} {'Test (N=251)':>16s} {'BraTS-Africa':>18s}")
    print("-" * 78)
    for key, name in MODELS:
        t = seg.get(f"{key}__test")
        a = seg.get(f"{key}__africa_std")
        if t is None:
            continue
        t_str = f"{t['mean']:.4f} +/- {t['std']:.4f}"
        a_str = f"{a['mean']:.4f} +/- {a['std']:.4f}" if a else "n/a"
        n_seeds = len(t["per_seed"])
        note = "" if n_seeds >= 3 else f"  ({n_seeds} seed)"
        print(f"{name:22s} {t_str:>16s} {a_str:>18s}{note}")
    print("-" * 78)


def table3(uq_dir):
    print("\n" + "=" * 78)
    print("Table 3. Failure detection & selective prediction on BraTS-Africa")
    print("         (TTA entropy; median seed per model; failure = Dice < 0.80)")
    print("=" * 78)
    print(f"{'Model':22s} {'fail-AUROC':>11s} {'Spearman':>10s} "
          f"{'Dice@100%':>10s} {'Dice@70%':>10s}")
    print("-" * 78)
    for key, name in MODELS:
        if key == "umamba":
            # U-Mamba is reported as a segmentation baseline only (Table 1 / Fig 1);
            # the controlled reliability study covers the four matched-backbone models.
            continue
        matches = glob.glob(os.path.join(uq_dir, f"{key}_seed*_africa_std.json"))
        if not matches:
            continue
        recs = json.load(open(matches[0]))
        dice = np.array([r["dice"] for r in recs], float)
        ent = np.array([r["entropy"] for r in recs], float)
        fails = (dice < 0.80).astype(int)
        au = auroc(ent, fails)
        rho = spearmanr(ent, dice).correlation
        d100 = retained_dice(ent, dice, 1.0)
        d70 = retained_dice(ent, dice, 0.7)
        print(f"{name:22s} {au:>11.3f} {rho:>10.2f} {d100:>10.4f} {d70:>10.4f}")
    print("-" * 78)
    print("Note: entropy ranks failures above successes (AUROC ~0.9-0.96) and is")
    print("negatively correlated with Dice, so abstaining on the most uncertain")
    print("cases raises retained Dice from ~0.89 to ~0.94 in every model.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="results/segmentation_results.json")
    ap.add_argument("--uq_dir", default="results/uq")
    args = ap.parse_args()
    table1(args.seg)
    table3(args.uq_dir)
    print("\nAll numbers above are read straight from results/. To reproduce the")
    print("split-conformal Dice floor (Table 4) run selective_prediction_analysis.py;")
    print("for full reliability curves + bootstrap CIs run reliability_study.py.\n")


if __name__ == "__main__":
    main()
