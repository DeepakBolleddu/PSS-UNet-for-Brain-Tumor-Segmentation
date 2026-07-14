#!/usr/bin/env python3
"""
train_fair.py  (GADI version, matches your data_loader.py)

ONE identical protocol for all four models so Table 3 becomes a controlled
comparison (fixes R2.8). Early stopping uses the 100-case VAL split only.
The 251 TEST split is never loaded here.

Models:
  pss_unet  -> from pss_unet import PSSUNet          (canonical: learned state-propagation)
  baseline  -> from ablation_models_final import BaselineVNet
  vnet_se   -> from ablation_models_final import VNetWithSE
  vnet_ssm  -> from ablation_models_final import VNetWithSSM

Protocol: AdamW, CosineAnnealingLR (NO warm restarts), Dice+BCE,
fixed seed, early stop on VAL dice. deep_sup applied identically to all.

Uses data_loader.create_data_loaders, which reads 'train' and
'validation' from the splits file. Point --splits_file at the HELD-OUT
splits so 'validation' is the 100-case set (NOT the 251).
"""
import os, gc, json, argparse, random, logging
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as _mp
try:
    _mp.set_sharing_strategy("file_system")
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loader import create_data_loaders   # YOUR loader


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_model(name, deep_sup):
    name = name.lower()
    if name == "pss_unet":
        from pss_unet import PSSUNet
        try:
            return PSSUNet(in_channels=4, out_channels=1,
                           deep_supervision=(deep_sup == "all"))
        except TypeError:
            return PSSUNet(in_channels=4, out_channels=1)
    if name == "baseline":
        from ablation_models import BaselineVNet
        return BaselineVNet(in_channels=4, out_channels=1)
    if name == "vnet_se":
        from ablation_models import VNetWithSE
        return VNetWithSE(in_channels=4, out_channels=1)
    if name == "vnet_ssm":
        from ablation_models import VNetWithSSM
        return VNetWithSSM(in_channels=4, out_channels=1)
    if name == "umamba":
        from umamba_seg import UMambaSeg
        return UMambaSeg(in_channels=4, out_channels=1)
    raise ValueError(name)


class DiceBCE(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def _one(self, logit, tgt):
        bce = self.bce(logit, tgt)
        p = torch.sigmoid(logit).reshape(-1)
        t = tgt.reshape(-1)
        inter = (p * t).sum()
        dice = (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
        return 0.5 * bce + 0.5 * (1 - dice)

    def forward(self, out, tgt):
        # PSS-UNet in deep-sup training mode returns (main, [aux...], state_norm)
        if isinstance(out, (list, tuple)):
            main = out[0]
            loss = self._one(main, tgt)
            if len(out) > 1 and isinstance(out[1], (list, tuple)):
                for aux in out[1]:
                    a = F.interpolate(aux, size=tgt.shape[2:], mode="trilinear",
                                      align_corners=False)
                    loss = loss + 0.4 * self._one(a, tgt)
            return loss
        return self._one(out, tgt)


@torch.no_grad()
def dice_of(out, tgt, thr=0.5, sm=1e-7):
    if isinstance(out, (list, tuple)):
        out = out[0]
    p = (torch.sigmoid(out) > thr).float().reshape(-1)
    t = tgt.reshape(-1)
    inter = (p * t).sum()
    return ((2 * inter + sm) / (p.sum() + t.sum() + sm)).item()


def save_ckpt(path, model, opt, sched, scaler, epoch, best, best_ep, hist):
    sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({"epoch": epoch, "model": sd, "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                "best_dice": best, "best_epoch": best_ep, "history": hist}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["baseline", "vnet_se", "vnet_ssm", "pss_unet", "umamba"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--splits_file", required=True,
                    help="HELD-OUT splits: 'validation' must be the 100-case set")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--accumulation_steps", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--deep_sup", choices=["none", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir) / args.model
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(out / f"{args.model}.log"),
                                  logging.StreamHandler()])
    logging.info("=" * 70)
    logging.info(f"FAIR TRAIN {args.model} | deep_sup={args.deep_sup} "
                 f"| epochs={args.epochs} patience={args.patience} "
                 f"| splits={args.splits_file}")
    logging.info("Early stopping on VAL (100). TEST (251) NOT loaded here.")
    logging.info("=" * 70)
    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # YOUR loader: reads 'train' and 'validation' from the splits file.
    train_loader, val_loader = create_data_loaders(
        args.data_dir, args.splits_file, batch_size=1, num_workers=4)

    model = build_model(args.model, args.deep_sup).to(device)
    n = sum(p.numel() for p in model.parameters())
    logging.info(f"Parameters: {n:,} ({n/1e6:.2f}M)")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    crit = DiceBCE()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-7)
    scaler = GradScaler()

    start, best, best_ep = 1, 0.0, 0
    hist = {"train_loss": [], "val_dice": [], "lr": []}
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start = ck["epoch"] + 1; best = ck.get("best_dice", 0.0); best_ep = ck.get("best_epoch", 0)
        hist = ck.get("history", hist)
        logging.info(f"Resumed at epoch {start} (best {best:.4f}@{best_ep})")

    no_improve = 0
    for ep in range(start, args.epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        run, nb = 0.0, 0
        for i, b in enumerate(train_loader):
            img = b["image"].to(device, non_blocking=True)
            msk = b["mask"].to(device, non_blocking=True)
            with autocast():
                pred = model(img)
                loss = crit(pred, msk) / args.accumulation_steps
            if torch.isnan(loss):
                opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss).backward()
            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            run += loss.item() * args.accumulation_steps; nb += 1
            if i % 50 == 0:
                clear_mem()
        tl = run / max(nb, 1)

        model.eval(); vd, vn = 0.0, 0
        with torch.no_grad():
            for b in val_loader:
                img = b["image"].to(device, non_blocking=True)
                msk = b["mask"].to(device, non_blocking=True)
                with autocast():
                    pred = model(img)
                vd += dice_of(pred, msk); vn += 1
        val = vd / max(vn, 1)

        sched.step()
        lr = opt.param_groups[0]["lr"]
        hist["train_loss"].append(tl); hist["val_dice"].append(val); hist["lr"].append(lr)
        logging.info(f"Epoch {ep}/{args.epochs} | LR {lr:.2e} | train_loss {tl:.4f} | val_dice {val:.4f}")

        if val > best + 1e-4:
            best, best_ep, no_improve = val, ep, 0
            save_ckpt(out / "best.pth", model, opt, sched, scaler, ep, best, best_ep, hist)
            logging.info(f"  * new best val_dice {best:.4f}")
        else:
            no_improve += 1
        save_ckpt(out / "last.pth", model, opt, sched, scaler, ep, best, best_ep, hist)
        json.dump(hist, open(out / "history.json", "w"), indent=2)

        if no_improve >= args.patience:
            logging.info(f"Early stop at epoch {ep} (no val gain {args.patience} epochs).")
            break
        clear_mem()

    logging.info(f"DONE {args.model}: best val_dice {best:.4f} @ {best_ep}")


if __name__ == "__main__":
    main()
