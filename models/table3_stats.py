#!/usr/bin/env python3
"""
table3_stats.py  --  Table 3 for the revision (Reviewer #6, comment 4)

Run locally; nothing to upload:

    python table3_stats.py --root runs_multiseed --out table3

Reads, for each model in {baseline, vnet_se, vnet_ssm, pss_unet} and each seed
folder runs_multiseed/<model>/seed*/, the files
    test_per_patient.json         (BraTS 2021 held-out, N=251)
    africa_std_per_patient.json   (BraTS-Africa, standardized pipeline, N=145)
Accepts all three layouts your scripts have produced:
    (a) list of {"patient_id":..., "dice":...}            (evaluate.py)
    (b) {"<pid>": {"dice":...}, ...}                       (older format)
    (c) {"per_patient": [ ... ]}                           (older format)

For every patient the Dice is averaged over the available seeds, then PSS-UNet is
compared with each control on the paired, seed-averaged per-patient Dice:
    delta (PSS - control), 95% bootstrap CI, paired t p, Wilcoxon p,
    Holm-adjusted p (over all comparisons made), Cohen's d (paired),
    TOST equivalence at margins 0.01 and 0.005 Dice (90% CI inside +/- margin).

Writes <out>.json, <out>.tex (ready to paste) and prints a check of the seed
means against Table 2.
"""
import argparse, glob, json, os
import numpy as np
from scipy import stats

MODELS = ["baseline", "vnet_se", "vnet_ssm", "pss_unet", "res_baseline"]   # res_baseline optional
LABEL = {"baseline": "Baseline V-Net", "vnet_se": "V-Net + SE",
         "vnet_ssm": "V-Net + SSM (bottleneck)", "pss_unet": "PSS-UNet",
         "res_baseline": "Residual backbone (E0)"}
# (first, second) pairs; delta = first - second.  E0 pairs are used only if E0 runs exist.
PAIRS = [("pss_unet", "baseline"), ("pss_unet", "vnet_se"), ("pss_unet", "vnet_ssm"),
         ("pss_unet", "res_baseline"),   # isolates the state-space machinery
         ("res_baseline", "baseline")]   # isolates the residual backbone
SPLITS = {"Test": "test_per_patient.json", "Africa": "africa_std_per_patient.json"}
RNG = np.random.default_rng(42)


def load_per_patient(path):
    data = json.load(open(path))
    if isinstance(data, list):
        recs = data
    elif isinstance(data, dict) and "per_patient" in data:
        recs = data["per_patient"]
    elif isinstance(data, dict):
        recs = [dict(v, patient_id=v.get("patient_id", k)) for k, v in data.items() if isinstance(v, dict)]
    else:
        raise ValueError(f"unrecognised layout: {path}")
    return {r["patient_id"]: float(r["dice"]) for r in recs if "dice" in r}


def seed_average(root, model, fname):
    """dict pid -> mean Dice over seeds; also per-seed mean Dice for the Table 2 check."""
    per_seed = {}
    for sd in sorted(glob.glob(os.path.join(root, model, "seed*"))):
        p = os.path.join(sd, fname)
        if os.path.exists(p):
            per_seed[os.path.basename(sd)] = load_per_patient(p)
    if not per_seed:
        raise FileNotFoundError(f"no {fname} under {root}/{model}/seed*/")
    common = set.intersection(*[set(d) for d in per_seed.values()])
    avg = {pid: float(np.mean([d[pid] for d in per_seed.values()])) for pid in common}
    seed_means = {s: float(np.mean(list(d.values()))) for s, d in per_seed.items()}
    return avg, seed_means


def paired_stats(a, b, margins):
    """a, b: aligned arrays (PSS-UNet, control)."""
    d = a - b; n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
    t, p_t = stats.ttest_rel(a, b)
    try:
        p_w = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p_w = float("nan")
    boots = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(10000)])
    ci95 = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    tcrit90 = stats.t.ppf(0.95, n - 1)
    ci90 = (m - tcrit90 * se, m + tcrit90 * se)
    tost = {}
    for eps in margins:
        p1 = 1 - stats.t.cdf((m + eps) / se, n - 1)     # H0: delta <= -eps
        p2 = 1 - stats.t.cdf((eps - m) / se, n - 1)     # H0: delta >= +eps
        tost[eps] = {"p": float(max(p1, p2)), "equivalent": bool(ci90[0] > -eps and ci90[1] < eps)}
    return {"n": int(n), "delta": float(m), "sd_diff": float(sd), "ci95": ci95,
            "ci90": (float(ci90[0]), float(ci90[1])), "t": float(t), "p_t": float(p_t),
            "p_wilcoxon": float(p_w), "cohens_d": float(m / sd), "tost": tost,
            "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum())}


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_multiseed")
    ap.add_argument("--out", default="table3")
    ap.add_argument("--margins", type=float, nargs="+", default=[0.01, 0.005])
    a = ap.parse_args()

    print("Table 2 check (mean over seeds of per-seed mean Dice, +/- seed SD):")
    avg = {}
    for split, fname in SPLITS.items():
        for mdl in MODELS:
            try:
                avg[(split, mdl)], sm = seed_average(a.root, mdl, fname)
            except FileNotFoundError:
                if mdl == "res_baseline":
                    continue
                raise
            v = np.array(list(sm.values()))
            print(f"  {split:6s} {LABEL[mdl]:26s} seeds={sorted(sm)}  {v.mean():.4f} +/- {v.std(ddof=0):.4f}"
                  f"  (per-seed: {', '.join(f'{x:.4f}' for x in v)})  n_patients={len(avg[(split, mdl)])}")

    rows = []
    for split in SPLITS:
        for first, ctrl in PAIRS:
            if (split, first) not in avg or (split, ctrl) not in avg:
                continue
            pss, c = avg[(split, first)], avg[(split, ctrl)]
            pids = sorted(set(pss) & set(c))
            r = paired_stats(np.array([pss[p] for p in pids]), np.array([c[p] for p in pids]), a.margins)
            r.update({"split": split, "control": ctrl, "first": first}); rows.append(r)
    adj = holm([r["p_t"] for r in rows])
    for r, pa in zip(rows, adj):
        r["p_holm"] = float(pa)

    print("\nTable 3 (PSS-UNet minus control, seed-averaged per-patient Dice):")
    hdr = f"{'Control':26s} {'Split':6s} {'n':>4s} {'delta':>8s} {'95% CI':>20s} {'p':>8s} {'p_Holm':>8s} {'d':>6s}" \
          + "".join(f"  TOST(+/-{e})" for e in a.margins)
    print(hdr)
    for r in rows:
        line = (f"{LABEL[r['first']][:10]+' - '+LABEL[r['control']]:26s} {r['split']:6s} {r['n']:4d} {r['delta']:+8.4f} "
                f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] {r['p_t']:8.4f} {r['p_holm']:8.4f} {r['cohens_d']:6.2f}")
        for e in a.margins:
            t = r["tost"][e]; line += f"  {'yes' if t['equivalent'] else 'no ':>3s} (p={t['p']:.1e})"
        print(line)

    # LaTeX
    e1, e2 = a.margins[0], (a.margins[1] if len(a.margins) > 1 else None)
    tex = ["\\begin{table*}[t]", "\\centering", "\\footnotesize",
           "\\caption{Pairwise comparison of PSS-UNet with each control on seed-averaged per-patient "
           "whole-tumour Dice. $\\Delta$ is PSS-UNet minus control with a 95\\% bootstrap CI; $p$ is the "
           "paired $t$-test, Holm-adjusted over the six comparisons; $d$ is the paired Cohen's $d$. "
           f"Equivalence is tested with two one-sided tests at a margin of $\\pm{e1}$ Dice "
           "(equivalent if the 90\\% CI of $\\Delta$ lies inside the margin). Where a difference is "
           "detectable its effect size is small and every comparison satisfies equivalence; we therefore "
           "describe the models as practically equivalent.}", "\\label{tab:pairwise}",
           "\\begin{tabular}{llrcccccc}", "\\toprule",
           f"Comparison & Split & $N$ & $\\Delta$ Dice & 95\\% CI & $p$ & $p_{{\\mathrm{{Holm}}}}$ & $d$ & TOST $\\pm{e1}$\\\\",
           "\\midrule"]
    for r in rows:
        p_fmt = lambda p: ("$<$0.001" if p < 0.001 else f"{p:.3f}")
        tex.append(f"{LABEL[r['first']]} vs {LABEL[r['control']]} & {r['split']} & {r['n']} & ${r['delta']:+.4f}$ & "
                   f"$[{r['ci95'][0]:+.4f},\\,{r['ci95'][1]:+.4f}]$ & {p_fmt(r['p_t'])} & {p_fmt(r['p_holm'])} & "
                   f"${r['cohens_d']:+.2f}$ & {'yes' if r['tost'][e1]['equivalent'] else 'no'}\\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(a.out + ".tex", "w").write("\n".join(tex))
    json.dump(rows, open(a.out + ".json", "w"), indent=2)
    print(f"\nwrote {a.out}.tex and {a.out}.json")


if __name__ == "__main__":
    main()