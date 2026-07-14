#!/usr/bin/env python3
"""
uq_inference.py

Per-case uncertainty via test-time augmentation (TTA). No retraining, no
dropout needed, so it works identically for all four models.

For each case: run the 8 axis-aligned flips, un-flip each probability map,
average into an ensemble probability, threshold at 0.5, and record:

  dice       Dice of the ensemble prediction vs ground truth
  tta_var    mean voxelwise variance across the 8 flips, over predicted FG
  entropy    mean voxelwise predictive entropy of the ensemble, over FG
  band_frac  fraction of voxels with ensemble prob in [0.3, 0.7]
  mean_conf  mean 2*|p-0.5| over FG  (1 = confident, 0 = unsure)
  pred_vol   predicted foreground voxel count (small lesions are risky)

Reuses evaluate.py case discovery and loaders, so preprocessing is
identical to your reported results. Same CLI shape as evaluate.py.

Examples:
  # held-out 251 test
  python uq_inference.py --model pss_unet --checkpoint runs/pss_unet/seed42/best.pth \
      --data_dir ../Data/BRATS2021_standardized \
      --splits_file ../Config/data_splits_heldout.json --split test \
      --out test_uq_pss_seed42.json

  # BraTS-Africa
  python uq_inference.py --model pss_unet --checkpoint runs/pss_unet/seed42/best.pth \
      --data_dir ../Data/BRATS_Africa --africa \
      --out africa_uq_pss_seed42.json
"""
import argparse, json
from types import SimpleNamespace
import numpy as np
import torch

from train_fair import build_model
from evaluate import discover_cases, crop_depth   # reuse your exact pipeline

FLIP_SETS = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]


@torch.no_grad()
def tta_predict(model, img, device):
    probs = []
    for axes in FLIP_SETS:
        x = torch.flip(img, dims=axes) if axes else img
        with torch.cuda.amp.autocast():
            out = model(x.to(device))
        if isinstance(out, (list, tuple)):
            out = out[0]
        p = torch.sigmoid(out.float())
        if axes:
            p = torch.flip(p, dims=axes)
        probs.append(p.cpu())
    stack = torch.stack(probs, 0)
    return stack.mean(0), stack


def dice_score(pred_bin, mask, sm=1e-7):
    p = pred_bin.reshape(-1)
    t = (mask.reshape(-1) > 0.5).float()
    inter = (p * t).sum()
    return float((2 * inter + sm) / (p.sum() + t.sum() + sm))


def case_scores(ens_prob, stack, mask):
    pred = (ens_prob > 0.5).float()
    fg = pred > 0.5
    n_fg = int(fg.sum().item())
    eps = 1e-7
    var_map = stack.var(0, unbiased=False)
    p = ens_prob.clamp(eps, 1 - eps)
    ent_map = -(p * p.log() + (1 - p) * (1 - p).log())
    if n_fg > 0:
        tta_var = float(var_map[fg].mean().item())
        entropy = float(ent_map[fg].mean().item())
        mean_conf = float((2 * (ens_prob[fg] - 0.5).abs()).mean().item())
    else:
        tta_var = float(var_map.mean().item())
        entropy = float(ent_map.mean().item())
        mean_conf = 0.0
    band = ((ens_prob >= 0.3) & (ens_prob <= 0.7)).float().mean().item()
    return {"dice": dice_score(pred, mask), "tta_var": tta_var, "entropy": entropy,
            "band_frac": float(band), "mean_conf": mean_conf, "pred_vol": n_fg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["baseline", "vnet_se", "vnet_ssm", "pss_unet", "umamba"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--splits_file", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--africa", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, deep_sup="none").to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=False)
    model.eval()

    # discover_cases reads args.africa / splits_file / split / data_dir
    cases = discover_cases(SimpleNamespace(
        africa=args.africa, data_dir=args.data_dir,
        splits_file=args.splits_file, split=args.split))
    tag = "africa" if args.africa else "test"
    print(f"{args.model}: {len(cases)} cases ({tag})")

    records, skipped = [], []
    for i, (folder, pid, loader, cat) in enumerate(cases):
        try:
            img, seg = loader(folder, pid)
        except Exception as e:
            skipped.append((pid, str(e))); continue
        seg = (seg > 0).astype(np.float32)
        img, seg = crop_depth(img, seg, 152)
        x = torch.from_numpy(img).unsqueeze(0)
        m = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0)
        ens, stack = tta_predict(model, x, device)
        rec = case_scores(ens, stack, m)
        rec.update({"patient_id": pid, "dataset": tag, "category": cat,
                    "model": args.model})
        records.append(rec)
        if i % 25 == 0:
            print(f"  {i}/{len(cases)} {pid} dice={rec['dice']:.4f} "
                  f"ent={rec['entropy']:.4f} conf={rec['mean_conf']:.3f}")

    json.dump(records, open(args.out, "w"), indent=2)
    print(f"wrote {len(records)} records -> {args.out} (skipped {len(skipped)})")


if __name__ == "__main__":
    main()
