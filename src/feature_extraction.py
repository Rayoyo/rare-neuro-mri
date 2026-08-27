'''Feature extraction utilities (pre-trained ResNet-50) and the full SVM training
pipeline used to produce the deployable resnet50_svm artifacts.'''

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
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
    """
    Extract features from a given dataset using the specified model.

    IMPORTANT: `dataset` must use the deterministic inference transform
    (`build_eval_dataset` / `return_train_eval=True`)
    Feeding the augmented training dataset produces random, duplicated features and silently corrupts every downstream
    result, from the SVM fit to the few-shot curve
    """
    if getattr(dataset, 'augmentation_factor', 1) > 1:
        raise ValueError(
            "extract_features received a dataset with augmentation_factor="
            f"{dataset.augmentation_factor}: features would be duplicated. "
            "Use build_eval_dataset(...) or build_dataloaders(..., return_train_eval=True)"
        )
    transform = getattr(dataset, 'transform', None)
    if getattr(transform, 'is_train', False):
        raise ValueError(
            "extract_features received a dataset with training-time augmentation enabled: "
            "features would be random. Use the deterministic inference transform instead"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )

    all_features, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc=desc):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.vstack(all_features), np.array(all_labels)


def tune_svm_C(X_train, y_train, X_val, y_val, C_values=(0.01, 0.1, 1, 10, 100), seed=33, verbose=True):
    """
    Grid search over the SVM regularization parameter, selected on the VALIDATION set

    Ties are broken in favour of the smallest C (strict `>`), i.e. the widest margin
    among equally-performing models
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_C, best_val_f1, history = None, -1.0, []

    if verbose:
        print("Hyperparameter tuning (Linear SVM)...")

    for C in C_values:
        svm = SVC(kernel='linear', C=C, random_state=seed)
        svm.fit(X_train_s, y_train)
        val_preds = svm.predict(X_val_s)

        val_f1 = f1_score(y_val, val_preds, average='macro')
        val_acc = accuracy_score(y_val, val_preds)
        history.append({'C': float(C), 'val_accuracy': float(val_acc), 'val_macro_f1': float(val_f1)})

        if verbose:
            print(f"  C={C:>6.2f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_C = C

    if verbose:
        print(f"\nBest C: {best_C} (Val Macro-F1: {best_val_f1:.4f})")

    return best_C, best_val_f1, history


def train_svm_classifier(
    X_train, y_train, X_val, y_val, X_test, y_test, class_names,
    C_values=(0.01, 0.1, 1, 10, 100), seed=33, refit_on_train_val=True, verbose=True
):
    """
    Full SVM training pipeline on the complete dataset: this is the step that produces
    the deployable `resnet50_svm_artifacts.pkl`

    1. Tune C on the validation set
    2. Refit on Train+Val with `probability=True`
    3. Evaluate ONCE on the held-out test set

    `probability=True` is mandatory here, not cosmetic: without it `SVC` exposes no
    `predict_proba`, and the API cannot return the probability distribution over the five
    classes. It costs an internal 5-fold Platt calibration at fit time

    `class_names` is stored in the artifacts because the SVM only ever sees integer
    labels. Losing the mapping produces an API that predicts correctly but reports
    permuted disease names: a silent, hard-to-notice failure

    Returns
    -------
    (artifacts, results) : (dict, dict)
    """
    class_names = list(class_names)

    best_C, best_val_f1, tuning_history = tune_svm_C(
        X_train, y_train, X_val, y_val, C_values=C_values, seed=seed, verbose=verbose
    )

    if refit_on_train_val:
        X_fit = np.vstack([X_train, X_val])
        y_fit = np.concatenate([y_train, y_val])
    else:
        X_fit, y_fit = X_train, y_train

    # A NEW scaler is fitted because the training distribution changed;
    # the test set is only transformed, never fitted
    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)
    X_test_s = scaler.transform(X_test)

    if verbose:
        print(f"\nFitting final SVM on {X_fit.shape[0]} samples "
              f"({'train+val' if refit_on_train_val else 'train'}), probability=True...")

    svm = SVC(kernel='linear', C=best_C, probability=True, random_state=seed)
    svm.fit(X_fit_s, y_fit)

    test_preds = svm.predict(X_test_s)
    test_proba = svm.predict_proba(X_test_s)

    test_acc = accuracy_score(y_test, test_preds)
    test_macro_f1 = f1_score(y_test, test_preds, average='macro')

    if verbose:
        print("=" * 60)
        print("RESNET-50 + SVM - FINAL TEST RESULTS")
        print("=" * 60)
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Test Macro-F1:  {test_macro_f1:.4f}")

    artifacts = {
        'svm': svm,
        'scaler': scaler,
        'class_names': class_names,
        'class_to_idx': {c: i for i, c in enumerate(class_names)},
        'best_C': float(best_C),
        'feature_extractor': 'resnet50_imagenet1k_v2',
        'feature_dim': int(X_fit.shape[1]),
        'img_size': 224,
        'calibrated_probabilities': True,
        'fitted_on': 'train+val' if refit_on_train_val else 'train',
        'n_fit_samples': int(X_fit.shape[0]),
        'tuning_history': tuning_history,
        'val_macro_f1': float(best_val_f1),
    }

    results = {
        'test_preds': test_preds,
        'test_proba': test_proba,
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_macro_f1),
        'best_C': float(best_C),
        'scaler': scaler,
        'svm': svm,
    }

    return artifacts, results


def save_features(output_dir, features_dict):
    """Save the .npy files in the specified directory"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, array in features_dict.items():
        filename = name if name.endswith('.npy') else f"{name}.npy"
        np.save(output_dir / filename, array)

    print(f"\nAll features have been saved in: {output_dir.resolve()}")
