import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

# --- Helper Functions for Preprocessing ---

def load_dicom_series(dicom_dir):
    """Loads a DICOM series from a directory and returns a SimpleITK image."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def resample_image(image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    """Resamples a SimpleITK image to a new spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # B-spline is good for smooth interpolation of intensity values
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(image)

def normalize_intensity(image):
    """Normalizes image intensity based on non-zero voxels to the [0, 1] range."""
    image_np = sitk.GetArrayFromImage(image)
    
    # Select non-zero voxels to avoid background bias
    non_zero_voxels = image_np[np.nonzero(image_np)]
    
    if non_zero_voxels.size == 0:
        # Handle cases with no foreground (e.g., all black image)
        return image

    mean = np.mean(non_zero_voxels)
    std = np.std(non_zero_voxels)
    
    # Z-score normalization: (value - mean) / std
    # Clamp values to avoid extreme outliers, common in medical imaging
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    
    normalized_np = np.clip(image_np, lower_bound, upper_bound)
    
    # Scale to [0, 1]
    min_val = np.min(normalized_np)
    max_val = np.max(normalized_np)
    if max_val > min_val:
        normalized_np = (normalized_np - min_val) / (max_val - min_val)
    
    normalized_image = sitk.GetImageFromArray(normalized_np)
    normalized_image.CopyInformation(image)
    return normalized_image

def process_patient_folder(patient_path, output_patient_path):
    """
    Processes all modalities for a single patient, converting DICOM to normalized NIfTI.
    """
    modalities = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    os.makedirs(output_patient_path, exist_ok=True)
    
    for mod in modalities:
        modality_dicom_path = os.path.join(patient_path, mod)
        if not os.path.exists(modality_dicom_path):
            # print(f"Warning: Modality {mod} not found for patient {os.path.basename(patient_path)}")
            continue

        try:
            # 1. Load DICOM series into a 3D volume
            dicom_image = load_dicom_series(modality_dicom_path)
            
            # 2. Resample to a standard voxel size (1x1x1 mm)
            resampled_image = resample_image(dicom_image)
            
            # 3. Normalize intensity
            normalized_image = normalize_intensity(resampled_image)
            
            # 4. Save as NIfTI file
            patient_id = os.path.basename(patient_path)
            output_filename = f"{patient_id}_{mod}.nii.gz"
            output_filepath = os.path.join(output_patient_path, output_filename)
            sitk.WriteImage(normalized_image, output_filepath)
            
        except Exception as e:
            print(f"Error processing {modality_dicom_path}: {e}")

# --- Main Script Execution ---

if __name__ == "__main__":
    # --- Step 1: Define Paths Based on Your Project Structure ---
    # This script is in 'DataPreprocessing', so we go up one level to the project root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path to the raw data
    raw_data_root = os.path.join(project_root, 'Data', 'PKG - RSNA-ASNR-MICCAI-BraTS-2021', 'RSNA-ASNR-MICCAI-BraTS-2021')
    
    # Path where processed data will be saved
    processed_data_output_root = os.path.join(project_root, 'Data', 'processed_nifti')
    
    print(f"Project Root: {project_root}")
    print(f"Reading raw data from: {raw_data_root}")
    print(f"Saving processed data to: {processed_data_output_root}")
    
    os.makedirs(processed_data_output_root, exist_ok=True)

    # --- Step 2: Find all patient folders ---
    patient_folders_to_process = []
    # os.walk will go through all subdirectories
    for dirpath, dirnames, _ in os.walk(raw_data_root):
        # A patient folder is one that contains the modality subfolders
        if all(mod in dirnames for mod in ['FLAIR', 'T1w', 'T1wCE', 'T2w']):
            patient_folders_to_process.append(dirpath)
            # By adding this, we prevent os.walk from going deeper into FLAIR, T1w etc.
            dirnames[:] = [] 
            
    print(f"\nFound {len(patient_folders_to_process)} patient folders to process.")

    # --- Step 3: Process each patient folder ---
    if patient_folders_to_process:
        for patient_path in tqdm(patient_folders_to_process, desc="Processing Patients"):
            patient_id = os.path.basename(patient_path)
            output_patient_path = os.path.join(processed_data_output_root, patient_id)
            process_patient_folder(patient_path, output_patient_path)
    else:
        print("No patient folders found. Please check the 'raw_data_root' path.")

    print("\nData preprocessing finished!")


