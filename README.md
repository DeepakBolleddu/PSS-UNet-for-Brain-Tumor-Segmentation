# A Controlled Study of State-Space Models for Brain Tumor Segmentation, with a Retraining-Free Reliability Layer Evaluated Under Cross-Dataset Shift

Code, data split, per-case results and analysis scripts for the manuscript (Biomedical Signal Processing and Control, second revision). Trained checkpoints are on Zenodo: **DOI: 10.5281/zenodo.23850917**.

## What the study does

Six 3D whole-tumour segmentation networks are trained under one leak-free protocol on BraTS 2021 (900 / 100 / 251 split, three seeds each) and evaluated on the 251 held-out cases and on 145 BraTS-Africa cases that were never used for training:

| Model | Description | Params |
|---|---|---|
| E1 `baseline` | V-Net-style 3D encoder-decoder, plain blocks | 10.26M |
| E2 `vnet_se` | E1 + squeeze-and-excitation (ratio 8) | 10.28M |
| E3 `vnet_ssm` | E1 with a unidirectional selective-scan block at the bottleneck (d_state 4, reference kernel) | 9.41M |
| E0 `res_baseline` | PSS-UNet's residual backbone with every state module removed (matched-backbone control) | 12.67M |
| E4 `pss_unet` | PSS-UNet: bidirectional selective scan at five reduced-resolution stages + progressive state | 14.32M |
| `umamba` | Independently implemented U-Mamba (bottleneck variant, bidirectional Mamba block, d_state 16, fused kernel) | 5.97M |

E1–E3 share a macro-architecture with E4 but use plain rather than residual blocks; E0 isolates the state modules from that difference (E4 vs E0) and the backbone from the baseline (E0 vs E1).

Main results: all models reach 0.930–0.933 Dice in distribution and 0.884–0.893 on BraTS-Africa; PSS-UNet differs from its own backbone (E0) by +0.0001 Dice. A test-time-augmentation entropy score, applied after training, detects BraTS-Africa failures with AUROC 0.90–0.97 (three seeds); its one blind spot, the confident empty prediction, is closed by an empty-mask check.

## Layout

```
models/        model definitions and pipeline scripts
  pss_unet.py            PSS-UNet (class SelectiveSSM3D is the scan block); requires mamba-ssm, raises if absent
  ablation_models.py     E1-E3
  res_baseline.py        E0
  umamba_seg.py          Mamba U-Net
  train_fair.py          training (deep supervision off by default; explicit scan-stage set)
  evaluate.py            per-patient Dice (strict checkpoint loading)
  uq_inference.py        test-time-augmentation records (--single_pass for one view)
  preprocess_africa_faithful.py   BraTS-Africa standardisation with the training-set transform
  profile_models.py      Table 5 (add --tta for the TTA column)
  table3_stats.py        Table 4 (paired tests, Holm, TOST)
  revision_reliability_analysis.py, final_analysis.py   Tables 6-7 and Appendix C
splits/        heldout_900_100_251.json (MD5 9f79ac8435b19fa3ff2f0c184339a4a5), africa_145_splits.json, calibration_half_125.json
configs/       run configuration of every model and seed
results/       per_patient/ (all models, seeds, cohorts), uq/, uq_single/, profiling/, analysis/, segmentation_results.json
scripts/pbs/   PBS job files used for training, evaluation, uncertainty and profiling
```

## Reproducing the tables without a GPU

```bash
python models/table3_stats.py --root <per-patient tree> --out table3           # Table 4
python models/final_analysis.py                                                 # Tables 6-7, Appendix C (edit ROOT inside)
```

## Environment

Python 3.9.13, torch 2.2.0+cu121, numpy 1.22.4, nibabel 5.3.2, scikit-learn 1.6.1, scipy, mamba-ssm 2.2.0, causal-conv1d 1.4.0. Training ran on Tesla V100-SXM2-32GB (NCI Gadi) for E1–E4 and the Mamba U-Net and on Tesla V100S-PCIE-32GB for E0 and all profiling; a Gadi-trained checkpoint re-evaluated in the second environment reproduced its Dice to five decimals.

## BraTS-Africa cohort

145 of the 146 released cases are analysed: our copy lacked the T2 volume of BraTS-SSA-00230 when the standardised copy was built, so that case was excluded before any evaluation. Standardise with `preprocess_africa_faithful.py` and evaluate with `evaluate.py --splits_file splits/africa_145_splits.json --split test`.

## Data

BraTS 2021 (Synapse) and BraTS-Africa (challenge portal) are not redistributed here.

## Citation

Bolleddu D., Yu P., Acharya U. R. A controlled study of state-space models for brain tumor segmentation, with a retraining-free reliability layer evaluated under cross-dataset shift. Biomedical Signal Processing and Control (in revision).
