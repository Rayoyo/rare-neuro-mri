'''ResNet-50 feature extraction and the linear-SVM classifier trained on those features.

Strategy 1 of the comparison: the backbone is frozen and never sees the labels; only the
SVM is trained.
'''

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
    """ImageNet ResNet-50 with the classifier head removed: images -> 2048-D vectors."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    return model.to(device).eval()


def extract_features(model, dataset, device, batch_size=64, desc="Extracting Features"):
    """
    Run the backbone over a dataset and return (features, labels).

    `dataset` must be the deterministic one (no augmentation): feature extraction needs one
    fixed vector per image, not a fresh random draw at every call.
    """
    assert not getattr(getattr(dataset, 'transform', None), 'is_train', False) \
        and getattr(dataset, 'augmentation_factor', 1) == 1, \
        "extract_features needs a non-augmented dataset (build_dataloaders(..., return_train_eval=True))"

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2 if torch.cuda.is_available() else 0,
                        pin_memory=torch.cuda.is_available())

    features, labels = [], []
    with torch.no_grad():
        for images, y, _ in tqdm(loader, desc=desc):
            features.append(model(images.to(device)).cpu().numpy())
            labels.extend(y.numpy())

    return np.vstack(features), np.array(labels)


def train_svm_classifier(X_train, y_train, X_val, y_val, X_test, y_test, class_names,
                         C_values=(0.01, 0.1, 1, 10, 100), seed=33):
    """
    Tune C on validation, refit on train+val, evaluate once on test.

    `probability=True` is required so the SVM exposes predict_proba: without it the API
    cannot return the probability distribution over the five classes.

    Returns (artifacts, test_preds, test_proba).
    """
    class_names = list(class_names)

    # --- tuning on validation ---
    sc = StandardScaler()
    Xtr, Xva = sc.fit_transform(X_train), sc.transform(X_val)

    best_C, best_f1 = None, -1.0
    print("Tuning C on the validation set:")
    for C in C_values:
        preds = SVC(kernel='linear', C=C, random_state=seed).fit(Xtr, y_train).predict(Xva)
        f1 = f1_score(y_val, preds, average='macro')
        print(f"  C={C:>6.2f} | Val Acc: {accuracy_score(y_val, preds):.4f} | Val Macro-F1: {f1:.4f}")
        if f1 > best_f1:                      # strict >: ties keep the smallest C
            best_f1, best_C = f1, C
    print(f"Best C: {best_C}")

    # --- final model on train+val ---
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])

    scaler = StandardScaler()                 # refit: the training distribution changed
    X_fit_s = scaler.fit_transform(X_fit)

    svm = SVC(kernel='linear', C=best_C, probability=True, random_state=seed)
    svm.fit(X_fit_s, y_fit)
    print(f"Final SVM fitted on {len(y_fit)} samples (train+val).")

    X_test_s = scaler.transform(X_test)
    test_preds = svm.predict(X_test_s)
    test_proba = svm.predict_proba(X_test_s)

    # class_names travels with the model: the SVM only ever sees integer labels
    artifacts = {
        'svm': svm,
        'scaler': scaler,
        'class_names': class_names,
        'best_C': float(best_C),
        'img_size': 224,
        'feature_extractor': 'resnet50_imagenet1k_v2',
        'calibrated_probabilities': True,
    }
    return artifacts, test_preds, test_proba


def save_features(output_dir, features_dict):
    """Save the feature matrices as .npy so the SVM can be re-tuned without the GPU."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in features_dict.items():
        np.save(output_dir / (name if name.endswith('.npy') else f"{name}.npy"), array)
    print(f"Features saved in: {output_dir}")
