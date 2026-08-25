'''File for metrics evaluation, visualization, few-shot analysis, and model artifact saving'''

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    f1_score, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title="Confusion Matrix"):
    """Generate and save the confusion matrix heatmap using Seaborn"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_per_class_metrics(y_true, y_pred, class_names, save_path=None, title="Per-Class Metrics"):
    """Generate and save the bar chart showing Precision, Recall, and F1-Score per class"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='coral')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='mediumseagreen')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' ', '\n') for c in class_names], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def run_fewshot_analysis(X_train, y_train, X_test, y_test, class_names, full_acc, full_f1, 
                         fewshot_sizes=[10, 25, 50, 100, 200], n_runs=5, C=1.0, seed=33):
    """Generate and save the few-shot robustness analysis by varying the training set size"""
    np.random.seed(seed)

    # security cast to numpy arrays in case they are not
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    results = []

    for n_per_class in fewshot_sizes:
        run_accs, run_f1s = [], []
        for run in range(n_runs):
            selected = []
            for cls_idx in range(len(class_names)):
                cls_indices = np.where(y_train == cls_idx)[0]
                if len(cls_indices) >= n_per_class:
                    sel = np.random.choice(cls_indices, n_per_class, replace=False)
                    selected.extend(sel)
            
            X_sub = X_train[selected]
            y_sub = y_train[selected]

            sc = StandardScaler()
            X_sub_s = sc.fit_transform(X_sub)
            X_test_s = sc.transform(X_test)

            svm = SVC(kernel='linear', C=C, random_state=seed + run)
            svm.fit(X_sub_s, y_sub)
            preds = svm.predict(X_test_s)

            run_accs.append(accuracy_score(y_test, preds))
            run_f1s.append(f1_score(y_test, preds, average='macro'))

        results.append({
            'n_per_class': n_per_class,
            'acc_mean': float(np.mean(run_accs)),
            'acc_std': float(np.std(run_accs)),
            'f1_mean': float(np.mean(run_f1s)),
            'f1_std': float(np.std(run_f1s))
        })

    fewshot_df = pd.DataFrame(results)

    # Adds the row with the complete dataset
    full_row = pd.DataFrame([{
        'n_per_class': len(X_train) // len(class_names),
        'acc_mean': float(full_acc),
        'acc_std': 0.0,
        'f1_mean': float(full_f1),
        'f1_std': 0.0
    }])
    fewshot_df = pd.concat([fewshot_df, full_row], ignore_index=True)
    return fewshot_df


def plot_fewshot_results(fewshot_df, model_name="Model", save_path=None):
    """Generate and save the plots for Accuracy and Macro-F1 as a function of the number of samples per class"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].errorbar(fewshot_df['n_per_class'], fewshot_df['acc_mean'],
                     yerr=fewshot_df['acc_std'], marker='o', capsize=5,
                     color='steelblue', linewidth=2, markersize=8)
    axes[0].set_xlabel('Examples per Class')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Accuracy vs Training Set Size', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].errorbar(fewshot_df['n_per_class'], fewshot_df['f1_mean'],
                     yerr=fewshot_df['f1_std'], marker='s', capsize=5,
                     color='coral', linewidth=2, markersize=8)
    axes[1].set_xlabel('Examples per Class')
    axes[1].set_ylabel('Test Macro-F1')
    axes[1].set_title('Macro-F1 vs Training Set Size', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.suptitle(f'{model_name}: Few-Shot Robustness', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def evaluate_and_save_all(
    y_true, y_pred, class_names, utils_dir, model_dir, model_name="resnet50_svm",
    X_train_feats=None, y_train_labels=None, X_test_feats=None, C=1.0,
    artifacts_to_save=None, torch_model_to_save=None, seed=33
):
    """
    Orchestrator function:
    1. Calculates and displays the classification report
    2. Saves and displays the Confusion Matrix and per-class plot on `UTILS_DIR`
    3. Executes and saves the Few-Shot Analysis on `UTILS_DIR`
    4. Saves the JSON file with results and the .pkl file with Artifacts on `MODEL_DIR`
    """
    utils_dir = Path(utils_dir)
    model_dir = Path(model_dir)
    utils_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print("=" * 60)
    print(f" EVALUATION RESULTS: {model_name.upper()}")
    print("=" * 60)
    print(f"Test Accuracy:     {acc:.4f}")
    print(f"Test Macro-F1:     {macro_f1:.4f}")
    print(f"Test Weighted-F1:  {weighted_f1:.4f}\n")
    
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, digits=4)
    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # confusion matrix and per-class metrics plots
    plot_confusion_matrix(
        y_true, y_pred, class_names, 
        save_path=utils_dir / f"confmat_{model_name}.png", 
        title=f"Confusion Matrix — {model_name}"
    )

    # metrics per class plot
    plot_per_class_metrics(
        y_true, y_pred, class_names, 
        save_path=utils_dir / f"perclass_metrics_{model_name}.png", 
        title=f"Per-Class Metrics — {model_name}"
    )

    # Few-Shot Analysis (if feature vectors are provided)
    fewshot_data = None
    if X_train_feats is not None and X_test_feats is not None and y_train_labels is not None:
        print("\nRunning few-shot experiments...")
        fewshot_df = run_fewshot_analysis(
            X_train_feats, y_train_labels, X_test_feats, y_true, 
            class_names, acc, macro_f1, C=C, seed=seed
        )
        plot_fewshot_results(
            fewshot_df, model_name=model_name, 
            save_path=utils_dir / f"fewshot_{model_name}.png"
        )

        # Convert few-shot results to a list of dictionaries for JSON serialization
        fewshot_data = [
            {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in row.items()}
            for row in fewshot_df.to_dict('records')
        ]

        print(fewshot_df.to_string(index=False))

    # saving JSON Files with Metrics
    metrics_summary = {
        'model_name': model_name,
        'test_accuracy': float(acc),
        'test_macro_f1': float(macro_f1),
        'test_weighted_f1': float(weighted_f1),
        'report': report_dict,
        'fewshot_results': fewshot_data
    }
    with open(model_dir / f"{model_name}_results.json", 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    # saving Pickle Artifacts (SVM, Scaler, Class Names, ...)
    if artifacts_to_save:
        artifacts_to_save['test_accuracy'] = float(acc)
        artifacts_to_save['test_macro_f1'] = float(macro_f1)
        artifacts_to_save['fewshot_results'] = fewshot_data
        
        with open(model_dir / f"{model_name}_artifacts.pkl", 'wb') as f:
            pickle.dump(artifacts_to_save, f)

    # saving PyTorch Weights (ResNet backbone or entire ConvNeXt model)
    if torch_model_to_save is not None:
        import torch
        torch.save(torch_model_to_save.state_dict(), model_dir / f"{model_name}_weights.pth")

    print(f"\n Output saved:")
    print(f"  - Graphs saved in: {utils_dir}")
    print(f"  - Artifacts and Weights saved in: {model_dir}")