"""
Dataset loader and preprocessing pipeline for Industrial Defect Detection.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

def get_train_transforms(img_size: int = 224):
    """
    Industrial surface domain augmentation:
    - Flips and 90-degree rotations (orientation invariance)
    - ShiftScaleRotate (minor position shifts)
    - CLAHE (vital for revealing faint scratches and surface cracks)
    - GaussNoise & MotionBlur (simulates industrial camera vibration/noise)
    - ImageNet normalization
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.06, 0.06), rotate=(-45, 45), p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(img_size: int = 224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class NEUDefectDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        
        self._discover_classes()
        
    def _discover_classes(self):
        # Auto-detect folder names matching the 6 defect categories
        direct_subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        target_classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]
        
        for c in target_classes:
            for d in direct_subdirs:
                d_clean = d.name.lower().replace(" ", "_").replace("-", "_")
                if c == d_clean or c in d_clean:
                    imgs = list(d.glob("*.bmp")) + list(d.glob("*.jpg")) + list(d.glob("*.png"))
                    if len(imgs) > 0:
                        class_idx = len(self.class_to_idx)
                        clean_name = d.name.lower().replace(" ", "_").replace("-", "_")
                        self.class_to_idx[clean_name] = class_idx
                        self.idx_to_class[class_idx] = clean_name
                        for img_path in imgs:
                            self.samples.append((str(img_path), class_idx))
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image at {img_path}")
            
        # Convert 1-channel grayscale to 3-channel RGB for pretrained backbones
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

def create_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 0, img_size: int = 224):
    """
    Creates stratified Train (70%), Val (15%), and Test (15%) DataLoaders.
    """
    raw_dataset = NEUDefectDataset(data_dir=data_dir)
    if len(raw_dataset) == 0:
        raise RuntimeError(f"No defect images found in {data_dir}. Did you run download_data.py?")
        
    labels = [sample[1] for sample in raw_dataset.samples]
    
    # 70% Train, 30% Temp (Val + Test)
    train_idx, temp_idx = train_test_split(
        range(len(raw_dataset)), test_size=0.30, stratify=labels, random_state=42
    )
    temp_labels = [labels[i] for i in temp_idx]
    
    # 15% Val, 15% Test
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_labels, random_state=42
    )
    
    train_ds = Subset(NEUDefectDataset(data_dir, transform=get_train_transforms(img_size)), train_idx)
    val_ds = Subset(NEUDefectDataset(data_dir, transform=get_val_transforms(img_size)), val_idx)
    test_ds = Subset(NEUDefectDataset(data_dir, transform=get_val_transforms(img_size)), test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, raw_dataset.idx_to_class
