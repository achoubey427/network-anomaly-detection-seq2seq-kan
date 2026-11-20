"""
Evaluation metrics calculation module for Anomaly Detection benchmarks.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, FPR, FNR, Latency.
"""

from typing import Any, Dict, List, Tuple
import numpy as np


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Compute basic confusion matrix elements: TP, TN, FP, FN."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute primary classification metrics from ground truth and predictions."""
    cm = compute_confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Area Under ROC Curve using trapezoidal rule without external dependencies."""
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    
    # Sort scores descending
    desc_idx = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_idx]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined single-class fallback
        
    tpr_list = [0.0]
    fpr_list = [0.0]
    cum_tp = 0
    cum_fp = 0
    
    for label in y_true_sorted:
        if label == 1:
            cum_tp += 1
        else:
            cum_fp += 1
        tpr_list.append(cum_tp / n_pos)
        fpr_list.append(cum_fp / n_neg)
        
    # Trapezoidal integration
    integrate_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = integrate_fn(tpr_list, fpr_list)
    return float(np.clip(auc, 0.0, 1.0))


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve using trapezoidal rule."""
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return 0.0
        
    desc_idx = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_idx]
    
    cum_tp = 0
    precisions = [1.0]
    recalls = [0.0]
    
    for i, label in enumerate(y_true_sorted, start=1):
        if label == 1:
            cum_tp += 1
        precisions.append(cum_tp / i)
        recalls.append(cum_tp / n_pos)
        
    # Numerical area under PR
    integrate_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = integrate_fn(precisions, recalls)
    return float(np.clip(auc, 0.0, 1.0))


def optimize_threshold_f1(y_true: np.ndarray, y_scores: np.ndarray, n_steps: int = 500) -> Tuple[float, float]:
    """Find the threshold that maximizes the F1 score on validation scores."""
    min_score, max_score = float(np.min(y_scores)), float(np.max(y_scores))
    if min_score == max_score:
        return min_score, 0.0
        
    thresholds = np.linspace(min_score, max_score, n_steps)
    best_f1 = -1.0
    best_thresh = min_score
    
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        metrics = compute_classification_metrics(y_true, preds)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_thresh = float(thresh)
            
    return best_thresh, best_f1


def compute_all_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compute the full suite of research evaluation metrics."""
    preds = (y_scores >= threshold).astype(int)
    cls_metrics = compute_classification_metrics(y_true, preds)
    roc_auc = compute_roc_auc(y_true, y_scores)
    pr_auc = compute_pr_auc(y_true, y_scores)
    
    results = dict(cls_metrics)
    results["roc_auc"] = roc_auc
    results["pr_auc"] = pr_auc
    results["threshold_applied"] = float(threshold)
    return results
