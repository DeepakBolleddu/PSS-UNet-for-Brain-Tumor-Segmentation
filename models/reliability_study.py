#!/usr/bin/env python3
"""
reliability_study.py

Extra analyses for the controlled-study paper, computed entirely from the
existing UQ JSONs (no GPU, no retraining). For each model it reports, on both
the in-distribution test set and standardized BraTS-Africa:

  1. Reliability curve: cases binned into deciles of uncertainty (entropy),
     showing mean Dice per bin. A monotonic drop = uncertainty tracks error.
     Also Spearman corr(entropy, Dice) and a calibration-style gap metric.
  2. Failure detection at TWO thresholds (Dice<0.80 and Dice<0.70):
     failure rate and failure-detection AUROC, with bootstrap 95% CIs.
  3. Selective prediction: retained Dice at 100/90/80/70% coverage.
  4. Africa subgroup: glioma vs other-neoplasm mean Dice, mean entropy,
     and failure rate (true categories read from raw africa_per_patient.json,
     because the africa_std UQ files carry a placeholder 'test' category).

Also writes reliability_curves.json (bin data) for plotting.

Run:
  python reliability_study.py --uq_dir uq --results_root runs_multiseed
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import rankdata, spearmanr

THRESHOLDS = [0.80, 0.70]
N_BOOT = 2000
RNG = np.random.default_rng(0)


def auroc(scores, labels):
    """P(score of a failure > score of a non-failure). labels: 1=failure."""
    labels = np.asarray(labels); scores = np.asarray(scores, float)
    npos = int(labels.sum()); nneg = int((labels == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return (r[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def boot_ci(fn, *arrays, n=N_BOOT):
    arrays = [np.asarray(a) for a in arrays]
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, m, m)
        v = fn(*[a[idx] for a in arrays])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def reliability_bins(entropy, dice, nbins=10):
    """Quantile bins of entropy; report mean dice per bin (low ent -> high dice)."""
    order = np.argsort(entropy)
    dice_sorted = np.asarray(dice)[order]
    ent_sorted = np.asarray(entropy)[order]
    bins = np.array_split(np.arange(len(dice)), nbins)
    rows = []
    for b in bins:
        if len(b) == 0:
            continue
        rows.append({"n": int(len(b)),
                     "entropy_mean": float(ent_sorted[b].mean()),
                     "dice_mean": float(dice_sorted[b].mean())})
    return rows


def selective_dice(entropy, dice):
    """Retained mean Dice keeping the (coverage) most-confident (lowest entropy)."""
    order = np.argsort(entropy)  # ascending entropy = descending confidence
    d = np.asarray(dice)[order]
    out = {}
    for cov in (1.0, 0.9, 0.8, 0.7):
        k = max(1, int(round(cov * len(d))))
        out[cov] = float(d[:k].mean())
    return out


def load_uq(path):
    recs = json.load(open(path))
    dice = np.array([r["dice"] for r in recs], float)
    ent = np.array([r["entropy"] for r in recs], float)
    pids = [r["patient_id"] for r in recs]
    return pids, dice, ent


def category_map(results_root):
    """pid -> true category from any raw africa_per_patient.json."""
    for p in glob.glob(os.path.join(results_root, "*", "*", "africa_per_patient.json")):
        try:
            data = json.load(open(p))
            m = {d["patient_id"]: d.get("category", "?") for d in data}
            if m:
                return m
        except Exception:
            continue
    return {}


def analyze_split(name, pids, dice, ent, catmap=None):
    print(f"\n  --- {name}  (N={len(dice)}, mean Dice={dice.mean():.4f}) ---")
    rho = spearmanr(ent, dice).correlation
    print(f"  Spearman corr(entropy, Dice) = {rho:.3f}  "
          f"(negative = higher uncertainty -> lower Dice, as desired)")

    # reliability curve
    rows = reliability_bins(ent, dice)
    # calibration-style gap: how much Dice drops from most- to least-confident decile
    gap = rows[0]["dice_mean"] - rows[-1]["dice_mean"]
    print(f"  reliability (entropy deciles): most-confident Dice={rows[0]['dice_mean']:.3f}"
          f"  least-confident Dice={rows[-1]['dice_mean']:.3f}  drop={gap:.3f}")

    # failure detection at both thresholds, with bootstrap CI on AUROC
    for thr in THRESHOLDS:
        fail = (dice < thr).astype(int)
        nf = int(fail.sum())
        a = auroc(ent, fail)
        lo, hi = boot_ci(lambda e, f: auroc(e, f), ent, fail)
        print(f"  Dice<{thr:.2f}: {nf}/{len(dice)} failures ({100*nf/len(dice):.1f}%)"
              f"  fail-AUROC={a:.3f}  95%CI[{lo:.3f},{hi:.3f}]")

    # selective prediction
    sd = selective_dice(ent, dice)
    print("  retained Dice @ coverage: "
          + "  ".join(f"{int(c*100)}%={sd[c]:.4f}" for c in (1.0, 0.9, 0.8, 0.7)))

    # bootstrap CI on mean Dice
    mlo, mhi = boot_ci(lambda d: float(d.mean()), dice)
    print(f"  mean Dice 95%CI [{mlo:.4f}, {mhi:.4f}]")

    # subgroup (africa only)
    if catmap:
        cats = {}
        for pid, d, e in zip(pids, dice, ent):
            c = catmap.get(pid, "unknown")
            cats.setdefault(c, {"d": [], "e": []})
            cats[c]["d"].append(d); cats[c]["e"].append(e)
        print("  subgroup breakdown:")
        for c in sorted(cats):
            d = np.array(cats[c]["d"]); e = np.array(cats[c]["e"])
            fr = 100 * (d < 0.80).mean()
            print(f"    {c:20s} n={len(d):3d}  Dice={d.mean():.4f}"
                  f"  entropy={e.mean():.4f}  fail%(<.80)={fr:.1f}")

    # ==========================================
    # FIX: Sanitize NaNs for strict JSON compliance
    # ==========================================
    clean_rho = None if np.isnan(rho) else float(rho)
    for r in rows:
        if np.isnan(r["entropy_mean"]): 
            r["entropy_mean"] = None
        if np.isnan(r["dice_mean"]): 
            r["dice_mean"] = None

    return {"reliability_bins": rows, "spearman": clean_rho}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uq_dir", default="uq")
    ap.add_argument("--results_root", default="runs_multiseed")
    args = ap.parse_args()

    catmap = category_map(args.results_root)
    if catmap:
        print(f"loaded {len(catmap)} true categories for Africa subgroup analysis")
    else:
        print("WARNING: no category map found; subgroup analysis will show 'unknown'")

    tests = sorted(glob.glob(os.path.join(args.uq_dir, "*_test.json")))
    curves = {}
    for tpath in tests:
        key = os.path.basename(tpath)[:-len("_test.json")]   # e.g. baseline_seed2
        apath = os.path.join(args.uq_dir, key + "_africa_std.json")
        print("\n" + "=" * 66)
        print(f"MODEL: {key}")
        print("=" * 66)

        pids, dice, ent = load_uq(tpath)
        curves.setdefault(key, {})["test"] = analyze_split("IN-DIST TEST", pids, dice, ent)

        if os.path.exists(apath):
            pids, dice, ent = load_uq(apath)
            curves[key]["africa"] = analyze_split("BRATS-AFRICA (std)", pids, dice, ent, catmap)
        else:
            print(f"  (no africa_std file for {key})")

    json.dump(curves, open("reliability_curves.json", "w"), indent=2)
    print("\nwrote reliability_curves.json (bin data for plotting)")


if __name__ == "__main__":
    main()