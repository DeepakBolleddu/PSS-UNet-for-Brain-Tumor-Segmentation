#!/usr/bin/env python3
"""
evaluate.py  (GADI version, matches your data_loader.py)

Evaluates a trained model on a list of patient IDs and writes per-patient
metrics (so ablation_significance.py can run paired tests).

Two modes:
  --split test    : reads the 'test' key (251 held-out) from --splits_file
  --ids_dir DIR   : evaluate every patient folder under DIR (for BraTS-Africa,
                    which is nested in 95_Glioma/ and 51_OtherNeoplasms/)

Uses YOUR BraTSMultimodalDataset (flair,t1,t1ce,t2 order; 155->152 crop).
This is the ONLY script that touches the test set.

Examples:
  # held-out 251 test
  python evaluate.py --model pss_unet \
     --checkpoint ../Results/fair/pss_unet/best.pth \
     --data_dir ../Data/BRATS2021_standardized \
     --splits_file ../Config/data_splits_heldout.json --split test \
     --out_dir ../Results/fair/pss_unet/test

  # BraTS-Africa (note: africa file naming differs; see --africa flag below)
  python evaluate.py --model pss_unet \
     --checkpoint ../Results/fair/pss_unet/best.pth \
     --data_dir ../Data/BRATS_Africa --africa \
     --out_dir ../Results/fair/pss_unet/africa
"""
import os, json, glob, argparse
import numpy as np
import torch
import nibabel as nib
from train_fair import build_model


# ---- Africa loader: file naming is t1n/t1c/t2w/t2f, nested folders ----------
# Map your model's expected order [flair, t1, t1ce, t2] to africa suffixes.
AFRICA_SUFFIX = {"flair": "t2f", "t1": "t1n", "t1ce": "t1c", "t2": "t2w"}


def load_case_standardized(folder, pid):
    mods = ["flair", "t1", "t1ce", "t2"]
    vols = []
    for m in mods:
        f = os.path.join(folder, f"{pid}_{m}.nii.gz")
        vols.append(nib.load(f).get_fdata().astype(np.float32))
    img = np.stack(vols, 0)
    seg = nib.load(os.path.join(folder, f"{pid}_seg.nii.gz")).get_fdata().astype(np.float32)
    return img, seg


def load_case_africa(folder, pid):
    mods = ["flair", "t1", "t1ce", "t2"]
    vols = []
    for m in mods:
        suf = AFRICA_SUFFIX[m]
        f = os.path.join(folder, f"{pid}-{suf}.nii.gz")
        vols.append(nib.load(f).get_fdata().astype(np.float32))
    img = np.stack(vols, 0)
    seg = nib.load(os.path.join(folder, f"{pid}-seg.nii.gz")).get_fdata().astype(np.float32)
    return img, seg


def crop_depth(img, seg, target=152):
    # match data_loader.py: center-crop depth 155 -> 152
    if img.shape[-1] == 155:
        s = (155 - target) // 2
        img = img[..., s:s + target]
        seg = seg[..., s:s + target]
    return img, seg


@torch.no_grad()
def metrics(logit, tgt, thr=0.5, sm=1e-7):
    p = (torch.sigmoid(logit) > thr).float()
    pf, tf = p.reshape(-1), tgt.reshape(-1)
    tp = (pf * tf).sum(); fp = (pf * (1 - tf)).sum()
    fn = ((1 - pf) * tf).sum(); tn = ((1 - pf) * (1 - tf)).sum()
    return {
        "dice": float((2 * tp + sm) / (2 * tp + fp + fn + sm)),
        "iou": float((tp + sm) / (tp + fp + fn + sm)),
        "sensitivity": float((tp + sm) / (tp + fn + sm)),
        "specificity": float((tn + sm) / (tn + fp + sm)),
        "precision": float((tp + sm) / (tp + fp + sm)),
    }


def discover_cases(args):
    """Return list of (folder, pid, loader_fn, category)."""
    cases = []
    if args.africa:
        # nested: <data_dir>/<category>/<pid>/<files>
        for cat in sorted(os.listdir(args.data_dir)):
            catp = os.path.join(args.data_dir, cat)
            if not os.path.isdir(catp):
                continue
            for pid in sorted(os.listdir(catp)):
                folder = os.path.join(catp, pid)
                if os.path.isdir(folder):
                    cases.append((folder, pid, load_case_africa, cat))
    else:
        with open(args.splits_file) as f:
            splits = json.load(f)
        ids = splits[args.split]
        print(f"Loaded {len(ids)} ids from '{args.split}' key")
        for pid in ids:
            cases.append((os.path.join(args.data_dir, pid), pid,
                          load_case_standardized, "test"))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["baseline", "vnet_se", "vnet_ssm", "pss_unet", "umamba"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--splits_file", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--africa", action="store_true")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model = build_model(args.model, deep_sup="none").to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    # tolerate checkpoints saved under 'model_state_dict'
    if isinstance(ck, dict) and "model_state_dict" in ck and "model" not in ck:
        state = ck["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    cases = discover_cases(args)
    print(f"Evaluating {args.model} on {len(cases)} cases "
          f"({'BraTS-Africa' if args.africa else 'held-out test'}).")

    per_patient, skipped = [], []
    with torch.no_grad():
        for i, (folder, pid, loader, cat) in enumerate(cases):
            try:
                img, seg = loader(folder, pid)
            except Exception as e:
                skipped.append((pid, str(e))); continue
            seg = (seg > 0).astype(np.float32)            # whole tumor, binary
            img, seg = crop_depth(img, seg, 152)
            x = torch.from_numpy(img).unsqueeze(0).to(device)
            t = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).to(device)
            with torch.cuda.amp.autocast():
                out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            m = metrics(out.float(), t)
            m["patient_id"] = pid
            m["category"] = cat
            per_patient.append(m)
            if i % 25 == 0:
                print(f"  {i}/{len(cases)} {pid} dice={m['dice']:.4f}")

    d = np.array([p["dice"] for p in per_patient])
    summary = {"model": args.model, "n": len(per_patient),
               "dice_mean": float(d.mean()), "dice_std": float(d.std()),
               "dice_median": float(np.median(d)), "skipped": skipped}
    # by-category (useful for Africa glioma vs other)
    cats = sorted(set(p["category"] for p in per_patient))
    if len(cats) > 1:
        summary["by_category"] = {
            c: {"n": int(sum(1 for p in per_patient if p["category"] == c)),
                "dice_mean": float(np.mean([p["dice"] for p in per_patient if p["category"] == c]))}
            for c in cats}

    json.dump(per_patient, open(os.path.join(args.out_dir, "per_patient.json"), "w"), indent=2)
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print(f"{args.model}: dice = {summary['dice_mean']:.4f} +/- {summary['dice_std']:.4f} "
          f"(n={summary['n']}, skipped={len(skipped)})")
    print(f"wrote {args.out_dir}/per_patient.json")


if __name__ == "__main__":
    main()

    
    
    
    
    
    
    