#!/usr/bin/env python3
"""
preprocess_africa_faithful.py

Standardizes raw BraTS-Africa with the EXACT transform that produced
BRATS2021_standardized (your training data). The normalize_intensity and
resample_volume functions below are copied VERBATIM from process_nifti.py,
which is the script that built the training set. Nothing else is added:
  - no brain-masked normalization (that was preprocessed6 / brats_africa_wy.py)
  - no z-score (that was the .npy folder / brats_africa_dp.py)
  - no reorientation (process_nifti.py writes an identity affine and does
    not reorder the voxel array)

Output matches load_case_standardized in evaluate.py exactly:
    <out>/<pid>/<pid>_flair.nii.gz, _t1, _t1ce, _t2, _seg
so you can evaluate WITHOUT --africa, reusing evaluate.py's standardized path.

It also writes africa_test_splits.json with {"test": [all pids]} so you can run:

  python evaluate.py --model pss_unet \
     --checkpoint ../Results/fair/pss_unet/best.pth \
     --data_dir ../Data/BRATS_Africa_standardized \
     --splits_file ../Data/BRATS_Africa_standardized/africa_test_splits.json \
     --split test \
     --out_dir ../Results/fair/pss_unet/africa

Run the SAME evaluate command for every model (baseline, vnet_se, vnet_ssm,
pss_unet, and the *1 variants) so the Africa comparison is on equal footing.
"""
import os, glob, json, argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# raw Africa modality suffix -> model's expected modality name
# model channel order is [flair, t1, t1ce, t2]; seg handled separately
AFRICA_SUFFIX = {"flair": "t2f", "t1": "t1n", "t1ce": "t1c", "t2": "t2w", "seg": "seg"}
TARGET_SHAPE = (240, 240, 155)


# ---- VERBATIM from process_nifti.py (the training pipeline) ----------------
def normalize_intensity(image_data):
    """Normalizes image intensity to the [0, 1] range."""
    if image_data.max() == image_data.min():
        return image_data
    return (image_data - image_data.min()) / (image_data.max() - image_data.min())


def resample_volume(volume, target_shape, is_mask=False):
    """Resamples a 3D volume to a target shape."""
    original_shape = volume.shape
    zoom_factors = [t / o for t, o in zip(target_shape, original_shape)]
    order = 0 if is_mask else 1
    resampled_volume = zoom(volume, zoom_factors, order=order, mode='nearest')
    return resampled_volume
# ----------------------------------------------------------------------------


def find_mod_file(patient_dir, pid, suffix):
    """Locate a modality file allowing '-suf' or '_suf' naming."""
    for sep in ("-", "_"):
        cands = glob.glob(os.path.join(patient_dir, f"*{sep}{suffix}.nii.gz"))
        if cands:
            return sorted(cands)[0]
    return None


def discover_patients(data_root):
    """Any folder that contains a seg file is a patient folder, at any depth."""
    patients = []
    for dirpath, _, files in os.walk(data_root):
        if any(f.endswith(("-seg.nii.gz", "_seg.nii.gz")) for f in files):
            patients.append(dirpath)
    return sorted(patients)


def process_patient(patient_dir, out_dir, pid):
    os.makedirs(out_dir, exist_ok=True)
    nz_report = {}
    for mod, suf in AFRICA_SUFFIX.items():
        f = find_mod_file(patient_dir, pid, suf)
        if f is None:
            return False, f"missing {mod}"
        data = nib.load(f).get_fdata().astype(np.float32)
        is_mask = (mod == "seg")
        data = resample_volume(data, TARGET_SHAPE, is_mask=is_mask)
        if not is_mask:
            data = normalize_intensity(data)
            nz = data[data != 0]
            nz_report[mod] = float(nz.mean()) if nz.size else 0.0
        # identity affine, exactly like process_nifti.py
        out_name = f"{pid}_{mod}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine=np.eye(4)),
                 os.path.join(out_dir, out_name))
    return True, nz_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../Data/BRATS_Africa",
                    help="raw BraTS-Africa root (nested category/patient or flat)")
    ap.add_argument("--out_root", default="../Data/BRATS_Africa_standardized")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    patients = discover_patients(args.data_root)
    print(f"Found {len(patients)} patient folders under {args.data_root}")

    done, skipped = [], []
    for pdir in patients:
        pid = os.path.basename(pdir.rstrip("/"))
        out_dir = os.path.join(args.out_root, pid)
        try:
            ok, info = process_patient(pdir, out_dir, pid)
        except Exception as e:
            ok, info = False, str(e)
        if ok:
            done.append(pid)
            print(f"[OK] {pid}  nz_means {{"
                  f"flair:{info['flair']:.3f} t1:{info['t1']:.3f} "
                  f"t1ce:{info['t1ce']:.3f} t2:{info['t2']:.3f}}}")
        else:
            skipped.append((pid, info))
            print(f"[SKIP] {pid}: {info}")

    # write a splits file so evaluate.py's standardized --split test path works
    splits_path = os.path.join(args.out_root, "africa_test_splits.json")
    json.dump({"test": done}, open(splits_path, "w"), indent=2)

    print(f"\nStandardized {len(done)} cases, skipped {len(skipped)}.")
    print(f"Wrote splits: {splits_path}")
    print("Training reference (BraTS2021 test case) nonzero-mean was ~0.26;")
    print("the nz_means above should sit in a comparable range. Large deviations")
    print("flag source cases with negative/odd intensities to inspect, not hide.")


if __name__ == "__main__":
    main()
