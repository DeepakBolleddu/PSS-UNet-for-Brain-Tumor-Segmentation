#!/usr/bin/env python3
"""
make_splits.py  -  Generate train / val / held-out-test splits ONCE.

The test set produced here is intended to be touched exactly once, at the very
end, for final reporting. Early stopping / model selection uses ONLY the val
set. This is the fix for the "test set == validation set" problem.

It discovers patient folders directly from the data directory so the counts
always match what is physically present (no 1351-vs-1251 arithmetic error).
"""
import os
import re
import json
import random
import argparse


def discover_patients(data_dir):
    """Return sorted list of patient IDs found as subdirectories."""
    ids = []
    for name in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, name)
        if os.path.isdir(p):
            ids.append(name)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=900)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test", type=int, default=251)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    patients = discover_patients(args.data_dir)
    n = len(patients)
    print(f"Discovered {n} patient folders in {args.data_dir}")

    need = args.train + args.val + args.test
    if need != n:
        print(f"WARNING: requested train+val+test = {need} but found {n}.")
        print("Adjust --train/--val/--test so they sum to the real count, "
              "or proceed and the script will clip to what is available.")

    rng = random.Random(args.seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)

    train = shuffled[:args.train]
    val = shuffled[args.train:args.train + args.val]
    test = shuffled[args.train + args.val:args.train + args.val + args.test]

    # Hard guarantee: no overlap between any split.
    assert not (set(train) & set(val)), "train/val overlap!"
    assert not (set(train) & set(test)), "train/test overlap!"
    assert not (set(val) & set(test)), "val/test overlap!"

    splits = {
        "seed": args.seed,
        "counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "note": "test set is held out; used ONCE for final reporting, never for early stopping",
        "train": sorted(train),
        "validation": sorted(val),
        "test": sorted(test),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"train={len(train)}  val={len(val)}  test={len(test)}  "
          f"(sum={len(train)+len(val)+len(test)})")
    print(f"Wrote {args.out}")
    print("Test IDs are frozen by seed; do not regenerate after training starts.")


if __name__ == "__main__":
    main()
