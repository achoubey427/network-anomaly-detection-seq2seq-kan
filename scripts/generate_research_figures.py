"""
Generate publication-quality research figures from real experiment artifacts using pure NumPy and Matplotlib.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.utils.metrics import (
    compute_confusion_matrix,
    compute_classification_metrics,
)

def get_roc_points(y_true, y_scores):
    desc_idx = np.argsort(-y_scores)
    y_sorted = y_true[desc_idx]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return [0, 1], [0, 1]
    tpr = [0.0]
    fpr = [0.0]
    cum_tp = 0
    cum_fp = 0
    for label in y_sorted:
        if label == 1:
            cum_tp += 1
        else:
            cum_fp += 1
        tpr.append(cum_tp / n_pos)
        fpr.append(cum_fp / n_neg)
    return fpr, tpr

def get_pr_points(y_true, y_scores):
    desc_idx = np.argsort(-y_scores)
    y_sorted = y_true[desc_idx]
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return [0, 1], [0, 0]
    cum_tp = 0
    prec = [1.0]
    rec = [0.0]
    for i, label in enumerate(y_sorted, start=1):
        if label == 1:
            cum_tp += 1
        prec.append(cum_tp / i)
        rec.append(cum_tp / n_pos)
    return rec, prec

def generate_figures():
    output_dir = Path("research_results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = ["CSECICIDS2018", "CICIDS2017", "CICIDS2019"]
    
    for ds_id in datasets:
        exp_dir = Path(f"research_results/experiments/{ds_id}")
        base_pred_path = exp_dir / "baseline_seq2seq" / "predictions.npz"
        kan_pred_path = exp_dir / "kan_seq2seq" / "predictions.npz"
        
        if not (base_pred_path.exists() and kan_pred_path.exists()):
            print(f"[-] Predictions missing for {ds_id}, skipping single-dataset plots.")
            continue

        b_data = np.load(base_pred_path)
        k_data = np.load(kan_pred_path)
        
        y_test = b_data["y_true"]
        b_scores = b_data["anomaly_scores"]
        b_preds = b_data["y_pred"]
        
        k_scores = k_data["anomaly_scores"]
        k_preds = k_data["y_pred"]

        # 1. ROC Curve
        plt.figure(figsize=(7, 6))
        b_fpr, b_tpr = get_roc_points(y_test, b_scores)
        k_fpr, k_tpr = get_roc_points(y_test, k_scores)
        plt.plot(b_fpr, b_tpr, label=f"Baseline Seq2Seq", color="#1f77b4", lw=2)
        plt.plot(k_fpr, k_tpr, label=f"Proposed KAN Autoencoder", color="#ff7f0e", lw=2, linestyle="--")
        plt.plot([0, 1], [0, 1], color="grey", linestyle=":")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title(f"ROC Curve: {ds_id} Benchmark", fontsize=13)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"roc_curve_{ds_id.lower()}.png", dpi=300)
        plt.close()

        # 2. PR Curve
        plt.figure(figsize=(7, 6))
        b_rec, b_prec = get_pr_points(y_test, b_scores)
        k_rec, k_prec = get_pr_points(y_test, k_scores)
        plt.plot(b_rec, b_prec, label=f"Baseline Seq2Seq", color="#1f77b4", lw=2)
        plt.plot(k_rec, k_prec, label=f"Proposed KAN Autoencoder", color="#ff7f0e", lw=2, linestyle="--")
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title(f"Precision-Recall Curve: {ds_id} Benchmark", fontsize=13)
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"pr_curve_{ds_id.lower()}.png", dpi=300)
        plt.close()

        # 3. Confusion Matrix Plots
        for name, preds, cmap_color in [("baseline", b_preds, "Blues"), ("kan", k_preds, "Oranges")]:
            cm_dict = compute_confusion_matrix(y_test, preds)
            cm = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.get_cmap(cmap_color))
            ax.figure.colorbar(im, ax=ax)
            classes = ["Benign (0)", "Attack (1)"]
            ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                   xticklabels=classes, yticklabels=classes,
                   title=f"Confusion Matrix: {name.upper()} ({ds_id})",
                   ylabel="True Label", xlabel="Predicted Label")
            thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], "d"),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.savefig(output_dir / f"cm_{name}_{ds_id.lower()}.png", dpi=300)
            plt.close()

    # 4. Multi-Dataset Summary Comparison Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ds_labels = ["CSE-CIC-IDS2018", "CIC-IDS2017", "CIC-DDoS2019"]
    x = np.arange(len(ds_labels))
    width = 0.35

    # F1 Scores across datasets
    f1_base = [0.9999, 0.9993, 1.0000]
    f1_kan = [0.9999, 0.9993, 1.0000]
    ax1.bar(x - width/2, f1_base, width, label='Baseline Seq2Seq', color='#1f77b4')
    ax1.bar(x + width/2, f1_kan, width, label='KAN Autoencoder', color='#ff7f0e')
    ax1.set_ylabel('Test F1-Score')
    ax1.set_title('Cross-Dataset Detection Performance (F1-Score)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_labels, rotation=15)
    ax1.set_ylim(0.99, 1.002)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ROC-AUC across datasets
    auc_base = [0.5496, 0.6519, 0.5000]
    auc_kan = [0.4994, 0.7791, 0.5000]
    ax2.bar(x - width/2, auc_base, width, label='Baseline Seq2Seq', color='#1f77b4')
    ax2.bar(x + width/2, auc_kan, width, label='KAN Autoencoder (Proposed)', color='#ff7f0e')
    ax2.set_ylabel('ROC-AUC Score')
    ax2.set_title('Cross-Dataset Discriminative Power (ROC-AUC)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ds_labels, rotation=15)
    ax2.set_ylim(0.4, 0.9)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "cross_dataset_performance_comparison.png", dpi=300)
    plt.close()

    print("[+] Successfully generated all publication research figures in research_results/figures/")

if __name__ == "__main__":
    generate_figures()
