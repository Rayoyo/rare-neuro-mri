'''Feature extraction utilities for extracting features from MRI images using a pre-trained ResNet-50 model'''

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_resnet50_extractor(device):
    """Loads the pre-trained ResNet-50 and removes the final classifier (returns 2048-dimensional vectors)"""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Identity()  # Remove the final classifier
    model = model.to(device)
    model.eval()
    return model


def extract_features(model, dataset, device, batch_size=64, desc="Extracting Features"):
    """Extract features from a given dataset using the specified model"""
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2 if torch.cuda.is_available() else 0, 
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    all_features, all_labels = [], []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc=desc):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.vstack(all_features), np.array(all_labels)


def save_features(output_dir, features_dict):
    """Save the .npy files in the specified directory"""
    output_dir = Path(output_dir)         # path conversion to avoid issues with string paths
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, array in features_dict.items():
        # add .npy extension if not present
        filename = name if name.endswith('.npy') else f"{name}.npy"
        np.save(output_dir / filename, array)
        
    print(f"\n✅ All features have been saved in: {output_dir.resolve()}")