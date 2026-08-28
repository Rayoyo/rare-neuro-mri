'''ConvNeXt-Tiny: lightweight fine-tuning and few-shot robustness.

Strategy 2 of the comparison: the ImageNet backbone is frozen except for the last stage,
the classification head is replaced, and the network is trained end-to-end on the MRI
images. No separate classifier is involved.
'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from dataset import make_fewshot_subset, set_seed


def get_convnext_tiny_model(num_classes=5, freeze_backbone=True, unfreeze_last_stage=True,
                            device='cuda'):
    """
    ImageNet ConvNeXt-Tiny with a new classification head.

    `freeze_backbone=True` by default, so the notebook does lightweight fine-tuning
    (last stage + head) rather than silently training all 28M parameters on 1400 images.
    """
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier[0].parameters():      # LayerNorm in the head
            p.requires_grad = False
        if unfreeze_last_stage:
            for p in model.features[7].parameters():    # ConvNeXt stage 4
                p.requires_grad = True

    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model.to(device)


def count_trainable_parameters(model):
    """(trainable, total) parameter counts — shows how much the freezing actually froze."""
    return (sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters()))


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """One training epoch. Returns (loss, accuracy)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate_epoch(model, dataloader, criterion, device):
    """Evaluation pass. Returns (loss, accuracy, true_labels, predictions)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_labels), np.array(all_preds)


def run_true_fewshot_finetuning(
    train_dataset, test_dataset, class_names, device,
    fewshot_sizes=(10, 25, 50, 100, 200), n_runs=3, epochs=8,
    batch_size=32, lr=1e-4, weight_decay=1e-2, num_workers=2,
    freeze_backbone=True, unfreeze_last_stage=True,
    full_acc=None, full_f1=None, n_full_per_class=None, seed=33
):
    """
    Few-shot robustness curve, measured the only way that is valid for a fine-tuned network:
    for every subset size the model is **re-initialised from ImageNet** and re-trained on
    that subset alone, then evaluated on the untouched test set.

    Reducing only a classifier fitted on features of the fully-trained network would leak
    the full-data supervision into every point of the curve.

    Returns a DataFrame with n_per_class, acc_mean, acc_std, f1_mean, f1_std, n_runs.
    """
    class_names = list(class_names)
    criterion = nn.CrossEntropyLoss()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    results = []

    for n_per_class in fewshot_sizes:
        accs, f1s = [], []

        for run in range(n_runs):
            set_seed(seed + run)
            subset, effective_n = make_fewshot_subset(train_dataset, n_per_class, seed=seed + run)
            loader = DataLoader(subset, batch_size=min(batch_size, len(subset)),
                                shuffle=True, num_workers=num_workers)

            model = get_convnext_tiny_model(len(class_names), freeze_backbone,
                                            unfreeze_last_stage, device)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            for _ in range(epochs):
                train_one_epoch(model, loader, criterion, optimizer, device)
                scheduler.step()

            _, _, y_true, y_pred = validate_epoch(model, test_loader, criterion, device)
            accs.append(accuracy_score(y_true, y_pred))
            f1s.append(f1_score(y_true, y_pred, average='macro'))

            del model, optimizer, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append({'n_per_class': int(effective_n),
                        'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
                        'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
                        'n_runs': int(n_runs)})
        r = results[-1]
        print(f"  n_per_class={r['n_per_class']:>4} | Acc {r['acc_mean']:.4f} +/- {r['acc_std']:.4f}"
              f" | Macro-F1 {r['f1_mean']:.4f} +/- {r['f1_std']:.4f}")

    df = pd.DataFrame(results)

    # right-hand end of the curve: the full-data model already measured in the notebook
    if full_acc is not None and full_f1 is not None:
        df = pd.concat([df, pd.DataFrame([{
            'n_per_class': int(n_full_per_class or len(train_dataset) // len(class_names)),
            'acc_mean': float(full_acc), 'acc_std': 0.0,
            'f1_mean': float(full_f1), 'f1_std': 0.0, 'n_runs': 1}])], ignore_index=True)

    return df.sort_values('n_per_class').reset_index(drop=True)
