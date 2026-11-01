#!/usr/bin/env python3
"""
revision_reliability_analysis.py

Re-analysis of the per-case TTA uncertainty files for the BSPC revision.
No GPU, no retraining. Reads <model>_<seed>_test.json and <model>_<seed>_africa_std.json
(the files written by uq_inference.py) and produces:

  A. Reproduction of Table 5 (fail-AUROC, bootstrap CI, Spearman, Dice@100/@70)
     and Table 6 (plug-in Dice-floor calibration on test, applied to Africa).
  B. Finite-sample selective-risk calibration (Reviewer #6, comment 2):
     candidate thresholds are the entropy quantiles at a PRE-SPECIFIED grid of
     coverage levels (100, 98, ..., 50 %). For each candidate, the exact-binomial
     (Clopper-Pearson) upper confidence bound at level 1 - delta/|grid| (Bonferroni)
     on the floor-violation rate among the RETAINED calibration cases is computed,
     and the most permissive candidate whose bound is <= alpha is selected.
     Guarantee (exchangeable data): P(selective risk of chosen tau > alpha) <= delta.
     Refs: Geifman & El-Yaniv 2017 (selective guaranteed risk);
           Bates et al. 2021 (RCPS); Angelopoulos et al. 2021 (Learn-then-Test).
  C. Repeated split-half calibration on the 251 test cases (Reviewer #6, comment 1):
     calibrate tau on a random half, evaluate coverage and violation on the other
     half (out-of-sample in-distribution) and on Africa; 1000 repeats.
  D. Mamba U-Net failure-AUROC with the two undefined-entropy cases included
     under worst-case (entropy := min) and best-case (entropy := max) assignments.
  E. Africa subgroup breakdown using the true categories carried by the umamba file.

Usage:
  python revision_reliability_analysis.py --uq_dir . --floor 0.80 --alpha 0.10 --delta 0.10
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import beta, rankdata, spearmanr

RNG = np.random.default_rng(0)
MODEL_NAMES = {"baseline": "V-Net baseline", "vnet_se": "V-Net + SE",
               "vnet_ssm": "V-Net + SSM (bottleneck)", "pss_unet": "PSS-UNet",
               "umamba": "Mamba U-Net"}


# ---------------------------------------------------------------- helpers
def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    npos, nneg = labels.sum(), (labels == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def boot_ci(fn, *arrays, n=2000):
    arrays = [np.asarray(a) for a in arrays]; m = len(arrays[0]); vals = []
    for _ in range(n):
        idx = RNG.integers(0, m, m)
        v = fn(*[a[idx] for a in arrays])
        if not np.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def retained_dice_at(dice, ent, cov):
    order = np.argsort(ent); k = max(1, int(round(cov * len(dice))))
    return float(dice[order][:k].mean())


def cp_upper(v, k, delta):
    """Exact (Clopper-Pearson) one-sided upper 1-delta confidence bound on a
    binomial proportion with v successes in k trials."""
    if k == 0:
        return 1.0
    return 1.0 if v == k else float(beta.ppf(1 - delta, v + 1, k - v))


def calibrate_plugin(dice, ent, floor, alpha):
    """Original rule: largest tau s.t. empirical violation among kept <= alpha."""
    order = np.argsort(ent); d, e = dice[order], ent[order]
    for k in range(len(d), 0, -1):
        if np.mean(d[:k] < floor) <= alpha:
            return float(e[k - 1]), k
    return -np.inf, 0


COVERAGE_GRID = (1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50)


def calibrate_cp(dice, ent, floor, alpha, delta, grid=COVERAGE_GRID):
    """Finite-sample rule on a pre-specified coverage grid with Bonferroni split of
    delta: most permissive grid level whose CP upper (1 - delta/|grid|) bound on the
    retained violation rate is <= alpha. Returns (tau, kept_k, violations)."""
    order = np.argsort(ent); d, e = dice[order], ent[order]; n = len(d)
    dp = delta / len(grid)
    for cov in grid:                           # most permissive first
        k = int(round(cov * n)); v = int((d[:k] < floor).sum())
        if cp_upper(v, k, dp) <= alpha:
            return float(e[k - 1]), k, v
    return -np.inf, 0, 0


def apply_tau(dice, ent, tau, floor):
    keep = ent <= tau
    if keep.sum() == 0:
        return 0, float("nan"), float("nan")
    return int(keep.sum()), float(dice[keep].mean()), float(np.mean(dice[keep] < floor))


def load(path):
    recs = json.load(open(path))
    dice = np.array([r["dice"] for r in recs], float)
    ent = np.array([np.nan if r["entropy"] is None else r["entropy"] for r in recs], float)
    cats = [r.get("category", "?") for r in recs]
    pids = [r["patient_id"] for r in recs]
    return pids, dice, ent, cats


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uq_dir", default=".")
    ap.add_argument("--floor", type=float, default=0.80)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--n_splits", type=int, default=1000)
    ap.add_argument("--out", default="revision_reliability_results.json")
    a = ap.parse_args()

    # true Africa categories: take them from any file that carries them
    catmap = {}
    for p in glob.glob(os.path.join(a.uq_dir, "*_africa_std.json")):
        pids, _, _, cats = load(p)
        for pid, c in zip(pids, cats):
            if c not in ("test", "?"):
                catmap[pid] = c
    print(f"category map: {len(catmap)} Africa cases with true subtype\n")

    results = {}
    for tpath in sorted(glob.glob(os.path.join(a.uq_dir, "*_test.json"))):
        key = os.path.basename(tpath)[:-len("_test.json")]
        apath = os.path.join(a.uq_dir, key + "_africa_std.json")
        if not os.path.exists(apath):
            continue
        model = key.rsplit("_seed", 1)[0]
        _, td, te, _ = load(tpath)
        apids, ad, ae, _ = load(apath)
        name = MODEL_NAMES.get(model, model)
        R = results[key] = {}
        print("=" * 72); print(f"{name}  ({key})"); print("=" * 72)

        # ----- D. undefined entropies (Mamba U-Net) -----
        nan_mask = np.isnan(ae)
        if nan_mask.any():
            print(f"  {nan_mask.sum()} Africa case(s) with undefined entropy: "
                  + ", ".join(f"{apids[i]} (Dice={ad[i]:.3f})" for i in np.where(nan_mask)[0]))
            fail = (ad < a.floor).astype(int)
            ok = ~nan_mask
            au_excl = auroc(ae[ok], fail[ok])
            ae_worst = ae.copy(); ae_worst[nan_mask] = np.nanmin(ae) - 1e-9   # ranked most confident
            ae_best = ae.copy(); ae_best[nan_mask] = np.nanmax(ae) + 1e-9     # ranked least confident
            print(f"  fail-AUROC  excluded={au_excl:.3f}  worst-case(ent:=min)={auroc(ae_worst, fail):.3f}"
                  f"  best-case(ent:=max)={auroc(ae_best, fail):.3f}")
            R["nan_cases"] = {"excluded": au_excl, "worst": auroc(ae_worst, fail), "best": auroc(ae_best, fail)}
            # for the remaining analyses treat undefined entropy as most confident (conservative)
            ae = ae_worst
        if np.isnan(te).any():
            te = np.where(np.isnan(te), np.nanmin(te) - 1e-9, te)

        # ----- A. Table 5 reproduction -----
        fail = (ad < a.floor).astype(int)
        au = auroc(ae, fail); lo, hi = boot_ci(auroc, ae, fail)
        rho = spearmanr(ae, ad).correlation
        d100, d70 = retained_dice_at(ad, ae, 1.0), retained_dice_at(ad, ae, 0.7)
        print(f"  [Table 5] Africa fail-AUROC={au:.3f} [{lo:.2f},{hi:.2f}]  Spearman={rho:.2f}"
              f"  Dice@100={d100:.3f}  Dice@70={d70:.3f}  failures={fail.sum()}/{len(ad)}")
        R["table5"] = {"auroc": au, "ci": [lo, hi], "spearman": float(rho), "dice100": d100, "dice70": d70}

        # ----- A. Table 6 reproduction (plug-in rule) -----
        tau_p, k_p = calibrate_plugin(td, te, a.floor, a.alpha)
        kept, rd, viol = apply_tau(ad, ae, tau_p, a.floor)
        print(f"  [Table 6, plug-in] tau={tau_p:.4f}  test kept {k_p}/{len(td)} (in-sample)"
              f"  -> Africa kept {kept}/{len(ad)}  retained Dice={rd:.4f}  violation={100*viol:.1f}%")
        R["table6_plugin"] = {"tau": tau_p, "test_kept": k_p, "africa_kept": kept,
                              "retained_dice": rd, "violation": viol}

        # ----- B. finite-sample (Clopper-Pearson) rule -----
        tau_c, k_c, v_c = calibrate_cp(td, te, a.floor, a.alpha, a.delta)
        kept, rd, viol = apply_tau(ad, ae, tau_c, a.floor)
        ub = cp_upper(v_c, k_c, a.delta / len(COVERAGE_GRID))
        print(f"  [NEW Table 6, CP delta={a.delta}] tau={tau_c:.4f}  test kept {k_c}/{len(td)}"
              f" (violations {v_c}, CP upper bound {100*ub:.1f}%)"
              f"  -> Africa kept {kept}/{len(ad)} ({100*kept/len(ad):.1f}%)  retained Dice={rd:.4f}"
              f"  violation={100*viol:.1f}%")
        R["table6_cp"] = {"tau": tau_c, "test_kept": k_c, "test_violations": v_c, "cp_upper": ub,
                          "africa_kept": kept, "retained_dice": rd, "violation": viol}

        # ----- C. repeated split-half calibration -----
        n = len(td); half = n // 2
        rows = {"plugin": {"id_cov": [], "id_viol": [], "af_cov": [], "af_viol": []},
                "cp":     {"id_cov": [], "id_viol": [], "af_cov": [], "af_viol": []}}
        for _ in range(a.n_splits):
            perm = RNG.permutation(n); A, B = perm[:half], perm[half:]
            for rule in ("plugin", "cp"):
                if rule == "plugin":
                    tau, _ = calibrate_plugin(td[A], te[A], a.floor, a.alpha)
                else:
                    tau, _, _ = calibrate_cp(td[A], te[A], a.floor, a.alpha, a.delta)
                kB, _, vB = apply_tau(td[B], te[B], tau, a.floor)
                kA_, _, vA_ = apply_tau(ad, ae, tau, a.floor)
                rows[rule]["id_cov"].append(kB / len(B)); rows[rule]["id_viol"].append(vB)
                rows[rule]["af_cov"].append(kA_ / len(ad)); rows[rule]["af_viol"].append(vA_)
        R["split_half"] = {}
        for rule in ("plugin", "cp"):
            s = {k: np.array(v, float) for k, v in rows[rule].items()}
            q = lambda x: (float(np.nanmedian(x)), float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5)))
            exceed = float(np.nanmean(s["id_viol"] > a.alpha))
            print(f"  [split-half x{a.n_splits}, {rule:6s}] held-out test: coverage {100*q(s['id_cov'])[0]:.1f}%"
                  f"  violation median {100*q(s['id_viol'])[0]:.1f}% [{100*q(s['id_viol'])[1]:.1f},{100*q(s['id_viol'])[2]:.1f}]"
                  f"  P(viol>alpha)={100*exceed:.1f}%   | Africa: coverage {100*q(s['af_cov'])[0]:.1f}%"
                  f"  violation median {100*q(s['af_viol'])[0]:.1f}% [{100*q(s['af_viol'])[1]:.1f},{100*q(s['af_viol'])[2]:.1f}]")
            R["split_half"][rule] = {"id_cov": q(s["id_cov"]), "id_viol": q(s["id_viol"]),
                                     "p_id_viol_exceeds_alpha": exceed,
                                     "af_cov": q(s["af_cov"]), "af_viol": q(s["af_viol"])}

        # ----- B2. floor sweep (where does the rule bind, and does it transfer?) -----
        R["floor_sweep"] = {}
        for fl in (0.80, 0.85, 0.90):
            tp, kp = calibrate_plugin(td, te, fl, a.alpha); p_ = apply_tau(ad, ae, tp, fl)
            tc, kc, vc = calibrate_cp(td, te, fl, a.alpha, a.delta); c_ = apply_tau(ad, ae, tc, fl)
            print(f"  [floor {fl:.2f}, alpha {a.alpha}] plug-in: test cov {100*kp/len(td):5.1f}%  Africa cov {100*p_[0]/len(ad):5.1f}%"
                  f"  Dice {p_[1]:.4f}  viol {100*p_[2]:4.1f}% {'ok' if p_[2] <= a.alpha else 'EXCEEDS'}"
                  f"   | finite-sample: test cov {100*kc/len(td):5.1f}%  Africa cov {100*c_[0]/len(ad):5.1f}%"
                  f"  Dice {c_[1]:.4f}  viol {100*c_[2]:4.1f}% {'ok' if c_[2] <= a.alpha else 'EXCEEDS'}")
            R["floor_sweep"][fl] = {"plugin": {"test_kept": kp, "africa_kept": p_[0], "dice": p_[1], "viol": p_[2]},
                                    "cp": {"test_kept": kc, "africa_kept": c_[0], "dice": c_[1], "viol": c_[2]}}

        # ----- E. subgroup -----
        if catmap:
            R["subgroup"] = {}
            for c in sorted(set(catmap.values())):
                idx = np.array([catmap.get(p) == c for p in apids])
                dd, ee = ad[idx], ae[idx]
                print(f"  [subgroup] {c:18s} n={idx.sum():3d}  Dice={dd.mean():.4f}  entropy={np.mean(ee):.4f}"
                      f"  fail%(<{a.floor})={100*np.mean(dd < a.floor):.1f}")
                R["subgroup"][c] = {"n": int(idx.sum()), "dice": float(dd.mean()),
                                    "entropy": float(np.mean(ee)), "fail_rate": float(np.mean(dd < a.floor))}
        print()

    json.dump(results, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()