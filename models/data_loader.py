#data_loader.py

import os
import torch
import nibabel as nib
import numpy as np
import json
import random
from torch.utils.data import Dataset, DataLoader

# --- Augmentation Class ---
class RandomFlip:
    """Applies a random flip to the image and mask along specified axes."""
    def __init__(self, p=0.5, axes=(0, 1, 2)):
        self.p = p
        self.axes = axes

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        for axis in self.axes:
            if random.random() < self.p:
                image = torch.flip(image, dims=[axis + 1])
                mask = torch.flip(mask, dims=[axis + 1])
        return {'image': image, 'mask': mask}

class BraTSMultimodalDataset(Dataset):
    """
    Dataset for BraTS multimodal MRI data.
    """
    def __init__(self, data_dir, patient_ids, modalities=['flair', 't1', 't1ce', 't2'], transforms=None):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.modalities = modalities
        self.transforms = transforms
        print(f"Dataset initialized for {len(self.patient_ids)} patients.")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_path = os.path.join(self.data_dir, patient_id)
       
        multimodal_data = []
        for modality in self.modalities:
            file_path = os.path.join(patient_path, f"{patient_id}_{modality}.nii.gz")
            img = nib.load(file_path)
            data = img.get_fdata().astype(np.float32)
            multimodal_data.append(data)
       
        image_tensor = torch.from_numpy(np.stack(multimodal_data, axis=0))
       
        mask_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata().astype(np.float32)
        mask_data[mask_data > 0] = 1.0
        mask_tensor = torch.from_numpy(mask_data).unsqueeze(0)

        if image_tensor.shape[-1] == 155:
            crop_start = (155 - 152) // 2
            crop_end = crop_start + 152
            image_tensor = image_tensor[:, :, :, crop_start:crop_end]
            mask_tensor = mask_tensor[:, :, :, crop_start:crop_end]

        sample = {'image': image_tensor, 'mask': mask_tensor}

        if self.transforms:
            sample = self.transforms(sample)
           
        sample['patient_id'] = patient_id
        return sample

def create_data_loaders(data_dir, splits_file, batch_size=1, num_workers=4):
    with open(splits_file, 'r') as f:
        splits = json.load(f)
   
    train_ids = splits['train']
    val_ids = splits['validation']
   
    train_transforms = RandomFlip(p=0.5)

    train_dataset = BraTSMultimodalDataset(data_dir, train_ids, transforms=train_transforms)
    val_dataset = BraTSMultimodalDataset(data_dir, val_ids, transforms=None)
   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
   
    print(f"\nCreated data loaders:")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
   
    return train_loader, val_loader




def get_test_loader(data_dir, splits_file, batch_size=1, num_workers=4):
    with open(splits_file, 'r') as f:
        splits = json.load(f)
   
    # Try 'test' first, then fall back to 'validation'
    if 'test' in splits:
        test_ids = splits['test']
        print(f"Using 'test' split: {len(test_ids)} samples")
    else:
        test_ids = splits['validation']
        print(f"Using 'validation' split as test: {len(test_ids)} samples")
    
    test_dataset = BraTSMultimodalDataset(data_dir, test_ids, transforms=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    return test_loader




