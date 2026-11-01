#!/usr/bin/env python3
"""
profile_models.py  --  Table 5 (inference profiling), exactly as Section 3.8 describes.

For each model: batch size 1, input 1 x 4 x 240 x 240 x 152, FP16 autocast,
torch.no_grad(), 5 warm-up iterations, 20 timed iterations with
torch.cuda.synchronize() before and after every timed forward pass; time per
volume = mean +/- SD over the 20 iterations; throughput = 1/mean; peak memory =
torch.cuda.max_memory_allocated() after torch.cuda.reset_peak_memory_stats() at
the start of profiling. With --tta the "forward pass" is the full 8-flip
test-time-augmentation ensemble (8 forward passes + un-flipping + averaging), so
the end-to-end cost of the reliability layer is measured rather than estimated.

Weights do not affect timing; checkpoints are loaded only if --ckpt_root is given.

    python profile_models.py --out profiling.json
    python profile_models.py --tta --out profiling_tta.json
"""
import argparse, json, time, platform
import numpy as np
import torch
from train_fair import build_model

MODELS = ["baseline", "vnet_se", "vnet_ssm", "pss_unet", "umamba", "res_baseline"]
FLIPS = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]


@torch.no_grad()
def one_pass(model, x, tta):
    if not tta:
        with torch.cuda.amp.autocast():
            out = model(x)
        return out[0] if isinstance(out, (list, tuple)) else out
    acc = None
    for axes in FLIPS:
        xi = torch.flip(x, dims=axes) if axes else x
        with torch.cuda.amp.autocast():
            out = model(xi)
        out = out[0] if isinstance(out, (list, tuple)) else out
        p = torch.sigmoid(out.float())
        if axes:
            p = torch.flip(p, dims=axes)
        acc = p if acc is None else acc + p
    return acc / len(FLIPS)


def profile(name, device, tta, warmup, timed, ckpt=None):
    model = build_model(name, deep_sup="none").to(device).eval()
    if ckpt:
        ck = torch.load(ckpt, map_location=device)
        model.load_state_dict(ck["model"] if "model" in ck else ck, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 4, 240, 240, 152, device=device)
    for _ in range(warmup):
        one_pass(model, x, tta)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(timed):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        one_pass(model, x, tta)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated(device)
    t = np.array(times) * 1e3
    rec = {"model": name, "params_M": round(n_params / 1e6, 2), "tta": tta,
           "time_ms_mean": float(t.mean()), "time_ms_sd": float(t.std(ddof=1)),
           "throughput_vol_per_s": float(1000.0 / t.mean()), "peak_mem_GB": peak / 1024**3,
           "warmup": warmup, "timed": timed, "input": [1, 4, 240, 240, 152]}
    del model, x; torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=20)
    ap.add_argument("--ckpt_root", default=None,
                    help="optional: runs_multiseed root; loads <root>/<model>/seed42/best.pth")
    ap.add_argument("--out", default="profiling.json")
    a = ap.parse_args()
    device = torch.device("cuda")
    env = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0), "python": platform.python_version()}
    for pkg in ("mamba_ssm", "causal_conv1d"):
        try:
            env[pkg] = __import__("importlib.metadata").metadata.version(pkg.replace("_", "-"))
        except Exception:
            env[pkg] = "not installed"
    print("environment:", json.dumps(env))
    rows = []
    for m in a.models:
        ckpt = f"{a.ckpt_root}/{m}/seed42/best.pth" if a.ckpt_root else None
        try:
            r = profile(m, device, a.tta, a.warmup, a.timed, ckpt)
        except Exception as e:
            print(f"{m}: FAILED ({e})"); continue
        rows.append(r)
        print(f"{m:13s} {r['params_M']:6.2f}M  {r['peak_mem_GB']:5.2f} GB  "
              f"{r['time_ms_mean']:7.1f} +/- {r['time_ms_sd']:4.1f} ms  {r['throughput_vol_per_s']:5.2f} vol/s"
              + ("  [8-flip TTA]" if a.tta else ""))
    json.dump({"environment": env, "rows": rows}, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()