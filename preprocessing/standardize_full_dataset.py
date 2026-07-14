import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

def standardize_full_dataset(processed_data_dir, target_shape=(240, 240, 155)):
    """Standardize all 241 patients to consistent dimensions."""
    
    print(f"STANDARDIZING FULL DATASET TO {target_shape}")
    print("=" * 50)
    
    # Read all complete patients
    with open(os.path.join(processed_data_dir, "complete_cases.txt"), 'r') as f:
        all_patients = [line.strip() for line in f if not line.startswith('#') and line.strip()]
    
    standardized_dir = os.path.join(processed_data_dir, "standardized_full")
    os.makedirs(standardized_dir, exist_ok=True)
    
    modalities = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    
    for patient_id in tqdm(all_patients, desc="Standardizing patients"):
        patient_source = os.path.join(processed_data_dir, patient_id)
        patient_dest = os.path.join(standardized_dir, patient_id)
        os.makedirs(patient_dest, exist_ok=True)
        
        for mod in modalities:
            source_file = os.path.join(patient_source, f"{patient_id}_{mod}.nii.gz")
            dest_file = os.path.join(patient_dest, f"{patient_id}_{mod}.nii.gz")
            
            if os.path.exists(source_file):
                # Load, resample, and save
                img = nib.load(source_file)
                data = img.get_fdata()
                
                if data.shape != target_shape:
                    # Calculate zoom factors
                    zoom_factors = [target_shape[i] / data.shape[i] for i in range(3)]
                    # Use order=3 (cubic) for high-quality resampling
                    resampled_data = zoom(data, zoom_factors, order=3, mode='nearest')
                else:
                    resampled_data = data
                
                # Save standardized image
                new_img = nib.Nifti1Image(resampled_data, img.affine, img.header)
                nib.save(new_img, dest_file)
    
    print(f"All {len(all_patients)} patients standardized!")
    print(f" Standardized dataset: {standardized_dir}")
    
    return standardized_dir

if __name__ == "__main__":
    processed_data_dir = "../Data/processed_nifti"
    standardize_full_dataset(processed_data_dir)


