"""
Dataset utilities shared across all models.
Handles loading, transforms, and augmentation.
"""

import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_train_transform():
    """Augmentation transform for training (medically plausible)"""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomResizedCrop(size=224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_test_transform():
    """Standard transform for validation and test (NO augmentation)"""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class MRIDataset(Dataset):
    """
    PyTorch Dataset for the Rare Neurological Diseases MRI dataset
    Supports on-the-fly augmentation via augmentation_factor 
    """
    def __init__(self, root_dir, split='train', transform=None, augmentation_factor=1):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        # Augmentation only for training split
        self.augmentation_factor = augmentation_factor if split == 'train' else 1
        self.samples = []
        self.class_to_idx = {}

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_path = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_path, fname), idx, cls))

        # Expand dataset virtually for augmentation
        if self.augmentation_factor > 1:
            expanded = []
            for item in self.samples:
                for _ in range(self.augmentation_factor):
                    expanded.append(item)
            self.samples = expanded

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, cls_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, cls_name