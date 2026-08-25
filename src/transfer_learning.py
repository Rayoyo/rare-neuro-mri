'''File for ConvNeXt-Tiny model definition, fine-tuning configuration, and feature extraction'''

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_convnext_tiny_model(num_classes=5, freeze_backbone=False, unfreeze_last_stage=True, device='cuda'):
    """
    Initializes ConvNeXt-Tiny pre-trained on ImageNet and replaces the classification head with a 
    new linear layer for the specified number of classes. Optionally freezes the backbone for light fine-tuning.
    """
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)

    # Optional freezing of the backbone for light fine-tuning
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
            
        # Freeze also the LayerNorm inside the classifier head (index 0)
        for param in model.classifier[0].parameters():
            param.requires_grad = False

        # If requested, unfreeze the last stage (Stage 3/4 of ConvNeXt)
        if unfreeze_last_stage:
            for param in model.features[7].parameters():
                param.requires_grad = True

    # Replacement of the final classification layer (768 in_features -> num_classes)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model.to(device)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Execute one epoch of training"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate_epoch(model, dataloader, criterion, device):
    """Validate the model during validation or test phase"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_labels), np.array(all_preds)


def extract_features_from_finetuned(model, dataset, device, batch_size=32, num_workers=2):
    """
    Extract features (768-D) from the fine-tuned ConvNeXt model by temporarily removing 
    the final linear layer of the classifier. Safe state restoration guaranteed via try/finally
    """
    model = model.to(device)
    model.eval()

    # Save original linear layer
    original_fc = model.classifier[2]

    try:
        # Temporarily replace final Linear with Identity
        model.classifier[2] = nn.Identity()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        all_features, all_labels = [], []

        with torch.no_grad():
            for images, labels, _ in tqdm(loader, desc="Extracting Fine-Tuned Features"):
                images = images.to(device)
                features = model(images)
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())

        return np.vstack(all_features), np.array(all_labels)

    finally:
        # Guarantee restoration of the original classifier even if interrupted
        model.classifier[2] = original_fc