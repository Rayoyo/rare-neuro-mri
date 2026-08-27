'''ConvNeXt-Tiny model definition, fine-tuning configuration, feature extraction
and genuine few-shot fine-tuning.'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import make_fewshot_subset, set_seed


def get_convnext_tiny_model(num_classes=5, freeze_backbone=True, unfreeze_last_stage=True, device='cuda'):
    """
    Initializes ConvNeXt-Tiny pre-trained on ImageNet and replaces the classification head
    with a new linear layer for the specified number of classes

    `freeze_backbone` defaults to True: the project describes a *lightweight* fine-tuning
    (last stage + head only), and a False default would silently train all 28M parameters
    on 1400 images
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

        # If requested, unfreeze the last stage (Stage 4 of ConvNeXt)
        if unfreeze_last_stage:
            for param in model.features[7].parameters():
                param.requires_grad = True

    # Replacement of the final classification layer (768 in_features -> num_classes)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model.to(device)


def count_trainable_parameters(model):
    """Returns (trainable, total) parameter counts - useful to prove the freezing worked"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


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

    As with ResNet, `dataset` must use the deterministic inference transform
    """
    if getattr(dataset, 'augmentation_factor', 1) > 1 or getattr(
            getattr(dataset, 'transform', None), 'is_train', False):
        raise ValueError(
            "extract_features_from_finetuned requires a deterministic dataset "
            "(build_eval_dataset / return_train_eval=True); otherwise the extracted "
            "features are random and duplicated"
        )

    model = model.to(device)
    model.eval()

    original_fc = model.classifier[2]

    try:
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
        model.classifier[2] = original_fc


# ---------------------------------------------------------------------------
# Genuine few-shot fine-tuning
# ---------------------------------------------------------------------------
def run_true_fewshot_finetuning(
    train_dataset, test_dataset, class_names, device,
    fewshot_sizes=(10, 25, 50, 100, 200), n_runs=3, epochs=8,
    batch_size=32, lr=1e-4, weight_decay=1e-2, num_workers=2,
    freeze_backbone=True, unfreeze_last_stage=True,
    full_acc=None, full_f1=None, n_full_per_class=None, seed=33, verbose=True
):
    """
    Genuine few-shot robustness curve for ConvNeXt-Tiny.

    Why this replaces the previous approach: extracting features from a model already
    fine-tuned on the FULL training set and then shrinking only the SVM's training data
    leaks the full-data supervision into every "few-shot" point. That is exactly why the
    old curve read ~0.996 accuracy at 10 examples/class - it was never measuring a
    low-data regime

    Here, for every subset size the model is re-initialised from ImageNet weights and
    fine-tuned from scratch on that subset only. The test set is never touched during
    training

    Parameters
    ----------
    train_dataset : MRIDataset
        Training split with augmentation ON and augmentation_factor=1
        (see `build_fewshot_train_dataset`), so each physical image appears once
    test_dataset : MRIDataset
        Deterministic test split

    Returns
    -------
    pandas.DataFrame with columns
        n_per_class, acc_mean, acc_std, f1_mean, f1_std, n_runs
    """
    class_names = list(class_names)
    criterion = nn.CrossEntropyLoss()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    results = []

    for n_per_class in fewshot_sizes:
        run_accs, run_f1s = [], []
        effective_n = n_per_class

        for run in range(n_runs):
            run_seed = seed + run
            set_seed(run_seed)

            subset, effective_n = make_fewshot_subset(
                train_dataset, n_per_class, seed=run_seed
            )
            loader = DataLoader(subset, batch_size=min(batch_size, len(subset)),
                                shuffle=True, num_workers=num_workers, drop_last=False)

            # Fresh ImageNet initialisation: no leakage from the full-data run
            model = get_convnext_tiny_model(
                num_classes=len(class_names),
                freeze_backbone=freeze_backbone,
                unfreeze_last_stage=unfreeze_last_stage,
                device=device
            )
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            for _ in range(epochs):
                train_one_epoch(model, loader, criterion, optimizer, device)
                scheduler.step()

            _, _, y_true, y_pred = validate_epoch(model, test_loader, criterion, device)
            run_accs.append(accuracy_score(y_true, y_pred))
            run_f1s.append(f1_score(y_true, y_pred, average='macro'))

            # Free GPU memory before the next re-initialisation
            del model, optimizer, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append({
            'n_per_class': int(effective_n),
            'acc_mean': float(np.mean(run_accs)),
            'acc_std': float(np.std(run_accs)),
            'f1_mean': float(np.mean(run_f1s)),
            'f1_std': float(np.std(run_f1s)),
            'n_runs': int(n_runs),
        })

        if verbose:
            r = results[-1]
            print(f"  n_per_class={r['n_per_class']:>4} | "
                  f"Acc {r['acc_mean']:.4f} +/- {r['acc_std']:.4f} | "
                  f"Macro-F1 {r['f1_mean']:.4f} +/- {r['f1_std']:.4f}")

    df = pd.DataFrame(results)

    # Append the full-dataset reference point, already measured by the main training run
    if full_acc is not None and full_f1 is not None:
        if n_full_per_class is None:
            n_full_per_class = len(train_dataset) // max(len(class_names), 1)
        df = pd.concat([df, pd.DataFrame([{
            'n_per_class': int(n_full_per_class),
            'acc_mean': float(full_acc),
            'acc_std': 0.0,
            'f1_mean': float(full_f1),
            'f1_std': 0.0,
            'n_runs': 1,
        }])], ignore_index=True)

    return df.sort_values('n_per_class').reset_index(drop=True)
