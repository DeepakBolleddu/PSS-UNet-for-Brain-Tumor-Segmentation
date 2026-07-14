# PSS-UNet for Brain Tumor Segmentation

Code, data splits, and results for our BSPC submission, a **controlled
reproducibility study of state-space models for 3D brain-tumor segmentation**,
paired with a retraining-free **reliability / selective-prediction** method that
we test under real cross-dataset shift (BraTS 2021 → BraTS-Africa).

This repository is deliberately small. It is meant to let a reviewer confirm
three things without a GPU:

1. **The evaluation is leak-free.** The 900 / 100 / 251 split is frozen in
   [`splits/heldout_900_100_251.json`](splits/heldout_900_100_251.json); the 251
   test IDs never appear in train or val, and only `evaluate.py` ever touches
   them.
2. **The numbers in the paper are the numbers we actually got.** Every table can
   be regenerated from the per-case JSON files in [`results/`](results) with one
   command — nothing is transcribed by hand.
3. **The comparison is fair.** All models are trained by the *same*
   `train_fair.py` under the *same* `configs/train_protocol.yaml` (one schedule,
   deep supervision off for everyone, flips-only augmentation, seeds 42/1/2,
   checkpoint chosen on the validation set only).



---

## What's in here

```
PSS-UNet-for-Brain-Tumor-Segmentation/
├── models/
│   ├── pss_unet.py                      # PSS-UNet: SE + genuine bidirectional selective SSM (S6)
│   ├── ablation_models.py               # BaselineVNet, VNetWithSE, VNetWithSSM (shared backbone)
│   ├── data_loader.py                   # BraTS dataset; flips-only aug; 155→152 depth crop
│   ├── train_fair.py                    # the ONE training protocol (all models, all seeds)
│   ├── evaluate.py                      # single-pass Dice on test / BraTS-Africa (the only script that reads test)
│   ├── uq_inference.py                  # per-case uncertainty via 8-flip test-time augmentation
│   ├── selective_prediction_analysis.py # risk–coverage, failure-AUROC, split-conformal Dice floor
│   ├── reliability_study.py             # reliability curves, bootstrap CIs, Africa subgroups
│   └── make_splits.py                   # regenerate the frozen split (seeded, with overlap asserts)
├── preprocessing/
│   ├── Preprocess_BRATS.py              # DICOM → NIfTI, resample to 1mm, per-modality processing
│   └── standardize_full_dataset.py      # resample every case to 240×240×155
├── splits/
│   └── heldout_900_100_251.json         # the frozen, leak-free split (seed 42)
├── configs/
│   └── train_protocol.yaml              # frozen hyper-parameters (mirrors train_fair.py defaults)
├── results/
│   ├── segmentation_results.json        # per-seed + mean/std Dice, all models, test + Africa
│   ├── reliability_curves.json          # entropy-decile Dice bins for the reliability figure
│   ├── training_config.json             # exact config emitted by a real training run
│   ├── uq/                              # per-case UQ records (median seed), test + africa_std
│   └── per_patient/                     # per-patient Dice for the paired significance tests
├── requirements.txt
└── LICENSE
```


**Data is not redistributed.** The BraTS 2021 and BraTS-Africa licenses forbid
it. Download the datasets yourself (links below) and point the scripts at them.

---

## The five models

| key in code | paper name | what it adds | params |
|---|---|---|---|
| `baseline` | V-Net baseline | encoder–decoder, InstanceNorm, no attention | 10.26 M |
| `vnet_se`  | V-Net + SE | squeeze-and-excitation channel attention | 10.28 M |
| `vnet_ssm` | V-Net + SSM (bottleneck) | one selective-scan block at the bottleneck | 9.41 M |
| `pss_unet` | PSS-UNet | SE + bidirectional selective SSM distributed across low-res stages + progressive state | 14.32 M |
| `umamba`   | U-Mamba | external SOTA SSM baseline (segmentation reference only) | 5.97 M |

`baseline`, `vnet_se`, `vnet_ssm` and `pss_unet` share the identical backbone,
filter count, normalization and activation, so the ablation is clean. `pss_unet`
uses a *real* Mamba-style S6 scan (state matrix, ZOH discretization,
input-dependent B/C/Δt, bidirectional), not a pooled channel gate — see the
module docstring in [`models/pss_unet.py`](models/pss_unet.py) for the exact
formulation and the honest note on where the scan is affordable.

> **On U-Mamba:** it is included as a segmentation baseline (Table 1 / Figure 1).
> Its model definition (`umamba_seg.py`) wraps the external U-Mamba/nnU-Net code
> and depends on a CUDA build of `mamba-ssm`, so we do not vendor it here; install
> [U-Mamba](https://github.com/bowang-lab/U-Mamba) and expose a `UMambaSeg` class
> to reproduce it. U-Mamba is **not** part of the controlled reliability study,
> which covers the four matched-backbone models.

---

## Setup

```bash
git clone https://github.com/DeepakBolleddu/PSS-UNet-for-Brain-Tumor-Segmentation.git
cd PSS-UNet-for-Brain-Tumor-Segmentation

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Reported runs used **PyTorch 2.2.0 + CUDA 12.1** on a Tesla V100-SXM2-32GB.
The fused `mamba-ssm` kernel is optional — `pss_unet.py` falls back to a correct
pure-PyTorch selective scan if it is not installed (just slower).

### Data

- **BraTS 2021** (training + in-distribution test): register at the
  [RSNA-ASNR-MICCAI BraTS 2021 challenge](https://www.synapse.org/brats2021).
- **BraTS-Africa** (out-of-distribution): from the
  [BraTS 2023 SSA challenge](https://www.synapse.org/brats2023).

Then preprocess to the layout the loaders expect (per-modality min–max to [0,1],
resampled to 240×240×155, files named `<pid>_<flair|t1|t1ce|t2|seg>.nii.gz`):

```bash
python preprocessing/Preprocess_BRATS.py          # DICOM → NIfTI, resample to 1mm
python preprocessing/standardize_full_dataset.py  # resample every case to 240×240×155
```

`data_loader.py` and `evaluate.py` center-crop depth 155 → 152 on the fly, and
augmentation is **random flips only** (`p=0.5` per axis) — nothing else. That is
the entire augmentation claim, and it lives in
[`models/data_loader.py`](models/data_loader.py).

---

## Reproduce the paper

Set `DATA` to your standardized BraTS 2021 root and `AFRICA` to your standardized
BraTS-Africa root, and run everything from inside `models/`:

```bash
cd models
DATA=/path/to/BRATS2021_standardized
AFRICA=/path/to/BRATS_Africa            # nested: 95_Glioma/ and 51_OtherNeoplasms/
SPLITS=../splits/heldout_900_100_251.json
```

### 1 — Train (one identical protocol, three seeds)

Early stopping uses the 100-case **validation** split only; the 251-case test
split is never loaded here.

```bash
for M in baseline vnet_se vnet_ssm pss_unet; do
  for S in 42 1 2; do
    python train_fair.py --model $M --seed $S \
        --data_dir $DATA --splits_file $SPLITS \
        --output_dir ../runs/$M/seed$S \
        --epochs 150 --lr 1e-4 --weight_decay 1e-5 \
        --accumulation_steps 8 --patience 30 --deep_sup none
  done
done
```

(These flags are the frozen protocol in
[`configs/train_protocol.yaml`](configs/train_protocol.yaml).)

### 2 — Evaluate → segmentation Dice (Table 1 / Figure 1)

`evaluate.py` is the **only** script that reads the test split. Run it per model
per seed on both the in-distribution test set and BraTS-Africa:

```bash
# in-distribution test (251)
python evaluate.py --model pss_unet --checkpoint ../runs/pss_unet/seed42/pss_unet/best.pth \
    --data_dir $DATA --splits_file $SPLITS --split test \
    --out_dir ../runs/pss_unet/seed42/test

# BraTS-Africa (out-of-distribution)
python evaluate.py --model pss_unet --checkpoint ../runs/pss_unet/seed42/pss_unet/best.pth \
    --data_dir $AFRICA --africa \
    --out_dir ../runs/pss_unet/seed42/africa
```

### 3 — Per-case uncertainty (test-time augmentation)

Eight axis-aligned flips, un-flipped and averaged; entropy is the uncertainty
score. No retraining, so it is identical for every model.

```bash
python uq_inference.py --model pss_unet --checkpoint ../runs/pss_unet/seed42/pss_unet/best.pth \
    --data_dir $DATA --splits_file $SPLITS --split test \
    --out ../results/uq/pss_unet_seed42_test.json

python uq_inference.py --model pss_unet --checkpoint ../runs/pss_unet/seed42/pss_unet/best.pth \
    --data_dir $AFRICA --africa \
    --out ../results/uq/pss_unet_seed42_africa_std.json
```

### 4 — Reliability & selective prediction (Tables 3–4, Figures 2–3)

```bash
# risk–coverage, failure-AUROC, and the split-conformal Dice floor (Table 4)
python selective_prediction_analysis.py \
    --test ../results/uq/pss_unet_seed42_test.json \
    --africa ../results/uq/pss_unet_seed42_africa_std.json \
    --score entropy --fail_thr 0.80 --floor 0.80 --alpha 0.10

# reliability curves, bootstrap 95% CIs, and the Africa subgroup breakdown
python reliability_study.py --uq_dir ../results/uq --results_root ../runs
```

### 5 — Regenerate the paper tables from the shipped JSONs (no GPU)

This is the one-command audit trail — it reads only `results/` and prints the
same numbers that are in the paper:

```bash
cd ..
python make_tables.py
```

---

## Results (what `make_tables.py` prints)

**Table 1 — Segmentation Dice (mean ± seed std).** Four models overlap in and
out of distribution; U-Mamba is level with them.

| Model | Test (N=251) | BraTS-Africa |
|---|---|---|
| V-Net baseline | 0.9304 ± 0.0003 | 0.8906 ± 0.0025 |
| V-Net + SE | 0.9311 ± 0.0015 | 0.8926 ± 0.0015 |
| V-Net + SSM (bneck) | 0.9316 ± 0.0013 | 0.8880 ± 0.0018 |
| PSS-UNet (SSM all) | 0.9327 ± 0.0009 | 0.8890 ± 0.0027 |
| U-Mamba | 0.9310 ± 0.0025 | 0.8894 |

**Table 3 — Failure detection & selective prediction on BraTS-Africa**
(TTA entropy, median seed, failure = Dice < 0.80). Entropy ranks failures above
successes and abstaining lifts retained Dice from ~0.89 to ~0.94.

| Model | fail-AUROC | Spearman(ent,Dice) | Dice @100% | Dice @70% |
|---|---|---|---|---|
| V-Net baseline | 0.945 | −0.86 | 0.896 | 0.944 |
| V-Net + SE | 0.962 | −0.87 | 0.897 | 0.939 |
| V-Net + SSM (bneck) | 0.922 | −0.85 | 0.896 | 0.936 |
| PSS-UNet | 0.899 | −0.78 | 0.896 | 0.934 |

The paired significance tests (PSS-UNet vs each alternative) come from the
per-patient Dice files in [`results/per_patient/`](results/per_patient); the
split-conformal Dice floor and the glioma-vs-other-neoplasm subgroup finding are
produced by the two analysis scripts above.

---

## The leak-free split, in one place

[`splits/heldout_900_100_251.json`](splits/heldout_900_100_251.json) contains the
frozen `train` (900), `validation` (100) and `test` (251) patient-ID lists,
generated with seed 42. To confirm there is no leakage:

```bash
python -c "import json; s=json.load(open('splits/heldout_900_100_251.json')); \
tr,va,te=set(s['train']),set(s['validation']),set(s['test']); \
print('sizes', len(tr),len(va),len(te)); \
print('overlaps', tr&va, tr&te, va&te)"
# -> sizes 900 100 251 ; overlaps set() set() set()
```

`make_splits.py` regenerates this exact file and asserts pairwise-empty overlaps
before writing.

---

## Citation

```bibtex
@article{bolleddu2026pssunet,
  title   = {A Controlled Study of State-Space Models for Brain Tumor
             Segmentation with Cross-Dataset Reliability},
  author  = {Bolleddu, Deepak and others},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is released under the [MIT License](LICENSE). BraTS data is **not**
included and remains under its original licenses; download it from the sources
above.
