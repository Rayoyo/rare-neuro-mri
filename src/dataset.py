'''File for data modeling
Manages data augmentation via TorchIO and PyTorch dataset creation for 2D MRI images.
'''

import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchio as tio
import matplotlib.pyplot as plt

# Global default path (can be changed via set_dataset_root)
DEFAULT_DATASET_ROOT = Path("./dataset")

def set_dataset_root(new_path):
    """Sets the global default directory for dataset root."""
    global DEFAULT_DATASET_ROOT
    DEFAULT_DATASET_ROOT = Path(new_path)
    print(f"Global dataset root set to: {DEFAULT_DATASET_ROOT.resolve()}")

# seed
def set_seed(seed=33):
    """
    Set the seed to ensure reproducibility for NumPy, PyTorch, and TorchIO
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# torchio transform
class TorchIOMRITransform:
    """
    Wrapper for applying TorchIO transformations to 2D MRI images (PIL -> Tensor -> TorchIO 4D -> Tensor 3D)
    """
    def __init__(self, img_size=224, is_train=True):
        self.img_size = img_size
        self.is_train = is_train

        # Initial resizing (common to all images)
        self.resize = T.Resize((img_size, img_size))

        # Standard ImageNet normalization (required by ResNet and ConvNeXt)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # TorchIO Augmentation Pipeline (active only during training)
        if self.is_train:
            self.augmentation = tio.Compose([
                # 1. Spatial transformations -> no horizontal flips (dx and sx are not equivalent in brain MRI)
                tio.RandomAffine(
                    degrees=(0, 0, 15),                     # Rotation ONLY around the Z-axis (in-plane ±15°)
                    translation=(10, 10, 0),                # Translation ONLY along X and Y (Z=0)
                    scales=(0.9, 1.1, 0.9, 1.1, 1.0, 1.0),         # Scaling 0.9x-1.1x on X and Y (Z=1.0)
                    p=0.7
                ),
                tio.RandomElasticDeformation(       # 2D B-spline deformation
                    num_control_points=(5, 5, 5),   # Number of control points in each dimension (Z=5 for 2D)
                    max_displacement=(5, 5, 0),     # Shift 0 on Z
                    p=0.3                           # Probability of applying the transformation
                ),

                # 2. Realistic MRI intensities and artifacts
                tio.RandomBiasField(coefficients=0.2, p=0.4),  # Field inhomogeneity
                tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.4),  # Contrast variations
                tio.RandomNoise(std=(0, 0.05), p=0.4),         # Gaussian/Rician noise
                tio.RandomBlur(std=(0.3, 0.8), p=0.3)          # Micro-blur / Artifacts
            ])

    def __call__(self, img_pil):
        # 1. Resize e conversione in Tensor [3, H, W]
        img = self.resize(img_pil)
        tensor = T.ToTensor()(img)

        # 2. Data Augmentation con TorchIO (solo in Train)
        if self.is_train:
            # TorchIO requires 4D [C, H, W, D]. Let's add a dimension D=1
            tensor_4d = tensor.unsqueeze(-1)
            subject = tio.Subject(image=tio.ScalarImage(tensor=tensor_4d))
            
            # Apply transformations
            transformed = self.augmentation(subject)
            
            # Removes dimension D=1 -> Returns to [C, H, W]
            tensor = transformed['image'].data.squeeze(-1)

        # 3. Final Normalization
        return self.normalize(tensor)


class MRIDataset(Dataset):
    """Custom PyTorch dataset for 2D MRI images"""
    def __init__(self, root_dir=None, split='train', transform=None, augmentation_factor=1):
        self.split = split
        self.transform = transform
        self.augmentation_factor = augmentation_factor if split == 'train' else 1
        
        # Imposta la directory radice ed esegue lo scan del dataset
        target_root = root_dir if root_dir is not None else DEFAULT_DATASET_ROOT
        self.set_root_dir(target_root)

    def set_root_dir(self, new_root_dir):
        """
        Allows dynamically setting or updating the dataset root directory from a notebook
        and re-scans the directory structure.
        """
        self.base_root_dir = Path(new_root_dir)
        self.root_dir = self.base_root_dir / self.split
        self.samples = []
        self.class_to_idx = {}

        # Check if the directory exists
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        # Build the dataset by scanning the directory structure
        classes = sorted([
            d for d in os.listdir(self.root_dir)
            if (self.root_dir / d).is_dir()
        ])

        # Create a mapping from class names to indices and collect image paths
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_path = self.root_dir / cls
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((cls_path / fname, idx, cls))

        # Multiply samples to increase epoch size during training
        if self.augmentation_factor > 1:
            self.samples = self.samples * self.augmentation_factor

    # Get the number of samples in the dataset
    def __len__(self):
        return len(self.samples)

    # Get a sample from the dataset
    def __getitem__(self, idx):
        img_path, label, cls_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            
        # Return the image tensor, label, and class name
        return image, label, cls_name

# Build data loaders for training, validation, and testing
def build_dataloaders(dataset_root=None, batch_size=64, img_size=224, aug_factor=2, seed=33):
    """Create datasets for Train, Val, and Test while applying the global seed"""
    set_seed(seed)
    root = dataset_root if dataset_root is not None else DEFAULT_DATASET_ROOT

    train_transform = TorchIOMRITransform(img_size=img_size, is_train=True)
    val_test_transform = TorchIOMRITransform(img_size=img_size, is_train=False)

    train_ds = MRIDataset(root, 'train', train_transform, augmentation_factor=aug_factor)
    val_ds = MRIDataset(root, 'val', val_test_transform, augmentation_factor=1)
    test_ds = MRIDataset(root, 'test', val_test_transform, augmentation_factor=1)

    return train_ds, val_ds, test_ds


def visualize_augmentations(dataset, num_samples=6, seed=33):
    """
    Show a side-by-side comparison between original and augmented images using TorchIO
    """
    set_seed(seed)
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))

    for i in range(num_samples):
        # Select a random index
        idx = random.randint(0, len(dataset) - 1)
        img_path, label, cls_name = dataset.samples[idx]

        # 1. Original Image (PIL)
        raw_img = Image.open(img_path).convert('RGB').resize((dataset.transform.img_size, dataset.transform.img_size))

        # 2. Augmented Image from PyTorch Dataset (Normalized Tensor)
        aug_tensor, _, _ = dataset[idx]

        # Denormalization for visualization with Matplotlib
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        aug_img = aug_tensor.permute(1, 2, 0).cpu().numpy()
        aug_img = std * aug_img + mean
        aug_img = np.clip(aug_img, 0, 1)

        # Plot Original Image
        axes[i, 0].imshow(raw_img)
        axes[i, 0].set_title(f"Original: {cls_name}")
        axes[i, 0].axis('off')

        # Plot Augmented Image
        axes[i, 1].imshow(aug_img)
        axes[i, 1].set_title("TorchIO Augmented")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()