'''File for data modeling
Manages data augmentation via TorchIO, PyTorch dataset creation for 2D MRI images, split verification 
and stratified few-shot subsampling 
'''

import os
import random
import warnings
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as T
import torchio as tio
import matplotlib.pyplot as plt

# Global default path (can be changed via set_dataset_root)
DEFAULT_DATASET_ROOT = Path("./dataset")

# ImageNet statistics: shared by the dataset, the notebooks and the API,
# so that inference preprocessing can never silently drift from training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')
SPLITS = ('train', 'val', 'test')


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


# ---------------------------------------------------------------------------
# Split verification
# ---------------------------------------------------------------------------
def verify_dataset_structure(dataset_root=None, expected_splits=SPLITS, verbose=True):
    """
    Verify that the Kaggle dataset is laid out as <root>/<split>/<class>/<image>.jpg
    and that every split exposes exactly the same class folders

    The Kaggle archive already ships the 70/15/15 split, so no re-splitting script is needed: 
    what IS needed is a guarantee that the class ordering is identical across splits. 
    `MRIDataset` assigns integer labels via sorted(folder_names) independently for each split, 
    so a class missing from one split would silently shift every label after it 
    -> the model would train and evaluate against permuted targets

    Returns
    -------
    classes : list[str]
        The canonical, sorted class list shared by all splits
    counts : dict[str, dict[str, int]]
        Number of images per split and class
    """
    root = Path(dataset_root) if dataset_root is not None else DEFAULT_DATASET_ROOT

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            f"Expected the Kaggle archive extracted as <root>/train|val|test/<class>/*.jpg"
        )

    missing = [s for s in expected_splits if not (root / s).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing split folder(s) {missing} under {root}. "
            f"Found instead: {sorted(p.name for p in root.iterdir())}"
        )

    per_split_classes = {}
    counts = {}
    for split in expected_splits:
        split_path = root / split
        classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
        if not classes:
            raise FileNotFoundError(f"No class folders inside {split_path}")
        per_split_classes[split] = classes
        counts[split] = {
            cls: sum(1 for f in (split_path / cls).iterdir()
                     if f.name.lower().endswith(VALID_EXTENSIONS))
            for cls in classes
        }

    # All splits must agree on the class list, otherwise label indices diverge
    reference = per_split_classes[expected_splits[0]]
    for split, classes in per_split_classes.items():
        if classes != reference:
            raise ValueError(
                "Class folders differ between splits -> label indices would be "
                f"inconsistent.\n  {expected_splits[0]}: {reference}\n  {split}: {classes}"
            )

    empty = [(s, c) for s, d in counts.items() for c, n in d.items() if n == 0]
    if empty:
        raise ValueError(f"Empty class folders detected: {empty}")

    if verbose:
        print(f"Dataset structure verified: {root}")
        print(f"  Canonical class order (label 0..{len(reference)-1}): {reference}")
        for split in expected_splits:
            total = sum(counts[split].values())
            per_class = sorted(set(counts[split].values()))
            print(f"  {split:<5} -> {total:>5} images | per-class: {per_class}")

    return reference, counts


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
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # TorchIO Augmentation Pipeline (active only during training)
        if self.is_train:
            self.augmentation = tio.Compose([
                # 1. Spatial transformations -> no horizontal flips (dx and sx are not equivalent in brain MRI)
                tio.RandomAffine(
                    degrees=(0, 0, 15),                     # Rotation ONLY around the Z-axis (in-plane +-15 deg)
                    translation=(10, 10, 0),                # Translation ONLY along X and Y (Z=0)
                    scales=(0.9, 1.1, 0.9, 1.1, 1.0, 1.0),  # Scaling 0.9x-1.1x on X and Y (Z=1.0)
                    p=0.7
                ),
                tio.RandomElasticDeformation(       # 2D B-spline deformation
                    num_control_points=(5, 5, 5),   # Number of control points in each dimension (Z=5 for 2D)
                    max_displacement=(5, 5, 0),     # Shift 0 on Z
                    p=0.3
                ),

                # 2. Realistic MRI intensities and artifacts
                tio.RandomBiasField(coefficients=0.2, p=0.4),   # Field inhomogeneity
                tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.4),  # Contrast variations
                tio.RandomNoise(std=(0, 0.05), p=0.4),          # Gaussian/Rician noise
                tio.RandomBlur(std=(0.3, 0.8), p=0.3)           # Micro-blur / Artifacts
            ])

    def __call__(self, img_pil):
        # 1. Resize and conversion to Tensor [3, H, W]
        img = self.resize(img_pil)
        tensor = T.ToTensor()(img)

        # 2. Data Augmentation with TorchIO (train only)
        if self.is_train:
            # TorchIO requires 4D [C, H, W, D]. Add a dimension D=1
            tensor_4d = tensor.unsqueeze(-1)
            subject = tio.Subject(image=tio.ScalarImage(tensor=tensor_4d))

            transformed = self.augmentation(subject)

            # Remove dimension D=1 -> back to [C, H, W]
            tensor = transformed['image'].data.squeeze(-1)

        # 3. Final Normalization
        return self.normalize(tensor)


def build_inference_transform(img_size=224):
    """
    Deterministic preprocessing used for val/test, for clean feature extraction,
    and by the API at inference time. Kept as a single source of truth
    """
    return TorchIOMRITransform(img_size=img_size, is_train=False)


class MRIDataset(Dataset):
    """Custom PyTorch dataset for 2D MRI images"""

    def __init__(self, root_dir=None, split='train', transform=None,
                 augmentation_factor=1, classes=None):
        self.split = split
        self.transform = transform
        self.augmentation_factor = augmentation_factor if split == 'train' else 1
        # Canonical class order; if None it is inferred from sorted(folder names)
        self._forced_classes = list(classes) if classes is not None else None

        target_root = root_dir if root_dir is not None else DEFAULT_DATASET_ROOT
        self.set_root_dir(target_root)

    def set_root_dir(self, new_root_dir):
        """
        Allows dynamically setting or updating the dataset root directory from a notebook
        and re-scans the directory structure
        """
        self.base_root_dir = Path(new_root_dir)
        self.root_dir = self.base_root_dir / self.split
        self.samples = []
        self.class_to_idx = {}

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Directory not found: {self.root_dir}\n"
                f"Expected layout: <root>/{self.split}/<class_name>/<image>.jpg"
            )

        found = sorted([d for d in os.listdir(self.root_dir)
                        if (self.root_dir / d).is_dir()])

        if self._forced_classes is not None:
            missing = [c for c in self._forced_classes if c not in found]
            if missing:
                raise ValueError(
                    f"Split '{self.split}' is missing class folder(s) {missing}; "
                    f"label indices would not match the other splits."
                )
            classes = self._forced_classes
        else:
            classes = found

        # Mapping from class names to indices, and image path collection
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_path = self.root_dir / cls
            for fname in sorted(os.listdir(cls_path)):
                if fname.lower().endswith(VALID_EXTENSIONS):
                    self.samples.append((cls_path / fname, idx, cls))

        # Number of distinct images on disk, before any augmentation duplication
        self.n_unique_samples = len(self.samples)

        # Multiply samples to increase epoch size during training
        if self.augmentation_factor > 1:
            self.samples = self.samples * self.augmentation_factor

    @property
    def classes(self):
        """Class names ordered by their integer label."""
        return [c for c, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]

    def get_labels(self):
        """Label array aligned with self.samples (useful for stratified subsampling)"""
        return np.array([label for _, label, _ in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, cls_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, cls_name


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------
def build_dataloaders(dataset_root=None, batch_size=64, img_size=224, aug_factor=2,
                      seed=33, return_train_eval=False, classes=None):
    """
    Create datasets for Train, Val and Test while applying the global seed

    Parameters
    ----------
    return_train_eval : bool
        When True a fourth dataset is returned: 
        the SAME training images but with the deterministic val/test transform and augmentation_factor=1

        This is required for classical Feature Extraction
        Passing the augmented `train_ds` to `extract_features` yields features that are 
        (a) random, because a different TorchIO draw is applied at every call, and 
        (b) duplicated, 
        because aug_factor=2 repeats every image
        It also inflates the few-shot curve, since the real number of distinct images per class becomes half of `n_per_class`

    Returns
    -------
    (train_ds, val_ds, test_ds) or (train_ds, val_ds, test_ds, train_eval_ds)
    """
    set_seed(seed)
    root = dataset_root if dataset_root is not None else DEFAULT_DATASET_ROOT

    train_transform = TorchIOMRITransform(img_size=img_size, is_train=True)
    val_test_transform = build_inference_transform(img_size=img_size)

    train_ds = MRIDataset(root, 'train', train_transform,
                          augmentation_factor=aug_factor, classes=classes)
    val_ds = MRIDataset(root, 'val', val_test_transform,
                        augmentation_factor=1, classes=classes)
    test_ds = MRIDataset(root, 'test', val_test_transform,
                         augmentation_factor=1, classes=classes)

    if not return_train_eval:
        return train_ds, val_ds, test_ds

    train_eval_ds = MRIDataset(root, 'train', val_test_transform,
                               augmentation_factor=1, classes=classes)
    return train_ds, val_ds, test_ds, train_eval_ds


def build_fewshot_train_dataset(dataset_root=None, img_size=224, classes=None):
    """
    Training split with augmentation ON but augmentation_factor=1, so that each physical
    image appears exactly once. Required to draw honest few-shot subsets: 
    with aug_factor=2 a subset of n indices could contain the same image twice
    """
    root = dataset_root if dataset_root is not None else DEFAULT_DATASET_ROOT
    return MRIDataset(root, 'train', TorchIOMRITransform(img_size=img_size, is_train=True),
                      augmentation_factor=1, classes=classes)


def make_fewshot_subset(dataset, n_per_class, seed=33, strict=False):
    """
    Draw a class-balanced subset with exactly `n_per_class` images per class

    If a class holds fewer images than requested, the subset is capped at the smallest
    available class and a warning is emitted, instead of silently dropping that class:
    dropping it would both unbalance the subset and remove a label from training, 
    which quietly changes the task being measured

    Returns
    -------
    torch.utils.data.Subset, int
        The subset and the effective number of examples per class actually used
    """
    rng = np.random.RandomState(seed)
    labels = dataset.get_labels() if hasattr(dataset, 'get_labels') else np.array(
        [dataset[i][1] for i in range(len(dataset))]
    )
    unique_labels = np.unique(labels)

    available = {int(c): int((labels == c).sum()) for c in unique_labels}
    smallest = min(available.values())

    effective = n_per_class
    if smallest < n_per_class:
        if strict:
            raise ValueError(
                f"Requested {n_per_class} examples per class but the smallest class "
                f"only has {smallest}. Class sizes: {available}"
            )
        warnings.warn(
            f"Requested {n_per_class} examples/class but the smallest class has "
            f"{smallest}; capping every class at {smallest} to keep the subset balanced",
            RuntimeWarning
        )
        effective = smallest

    selected = []
    for c in unique_labels:
        cls_indices = np.where(labels == c)[0]
        selected.extend(rng.choice(cls_indices, effective, replace=False))

    rng.shuffle(selected)
    return Subset(dataset, [int(i) for i in selected]), effective


def visualize_augmentations(dataset, num_samples=6, seed=33):
    """
    Show a side-by-side comparison between original and augmented images using TorchIO
    """
    set_seed(seed)
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))

    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        img_path, label, cls_name = dataset.samples[idx]

        # 1. Original Image (PIL)
        raw_img = Image.open(img_path).convert('RGB').resize(
            (dataset.transform.img_size, dataset.transform.img_size))

        # 2. Augmented Image from PyTorch Dataset (Normalized Tensor)
        aug_tensor, _, _ = dataset[idx]

        # Denormalization for visualization with Matplotlib
        mean = np.array(IMAGENET_MEAN)
        std = np.array(IMAGENET_STD)
        aug_img = aug_tensor.permute(1, 2, 0).cpu().numpy()
        aug_img = std * aug_img + mean
        aug_img = np.clip(aug_img, 0, 1)

        axes[i, 0].imshow(raw_img)
        axes[i, 0].set_title(f"Original: {cls_name}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(aug_img)
        axes[i, 1].set_title("TorchIO Augmented")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
