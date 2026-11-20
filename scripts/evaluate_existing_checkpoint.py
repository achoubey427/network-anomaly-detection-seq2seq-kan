"""
Evaluate Existing Checkpoint on Held-Out Test Set.
Evaluates saved model checkpoint without retraining, optimizing threshold on validation only
and generating complete test metrics, predictions.npz, and threshold.json.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.utils.config_loader import load_research_config
from src.data_adapters.dataset_manager import load_processed_dataset
from src.models import build_model
from src.utils.metrics import optimize_threshold_f1, compute_all_metrics, compute_classification_metrics
from src.models.summary import summarize_model


def evaluate_checkpoint(dataset_id: str, model_name: str, config_path: str = "configs/research.yaml", note: str = "time_bounded_checkpoint"):
    config = load_research_config(config_path)
    exp_dir = Path(f"research_results/experiments/{dataset_id}/{model_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = exp_dir / f"{dataset_id}_best_model.pt"

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_file} not found.")

    print(f"[+] Loading dataset '{dataset_id}' from processed data...")
    data = load_processed_dataset(dataset_id)
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    input_dim = X_val.shape[-1]
    seq_len = X_val.shape[1]

    print(f"[+] Loading checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", 0)
    print(f"[+] Checkpoint is from Epoch {epoch} (Val Loss: {ckpt.get('val_loss', 'N/A')})")

    model = build_model(model_name, input_dim=input_dim, config=config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    summary = summarize_model(model, input_shape=(1, seq_len, input_dim))

    # 1. Validation Anomaly Scoring & Threshold Selection
    print("[+] Scoring validation set and searching optimal decision threshold (theta*)...")
    S_val = model.compute_anomaly_score(X_val, aggregation="mean")
    n_search_steps = config.get("thresholding", {}).get("search_steps", 500)
    theta_star, val_f1 = optimize_threshold_f1(y_val, S_val, n_steps=n_search_steps)

    if val_f1 == 0.0 and len(np.unique(y_val)) == 1:
        k = config.get("thresholding", {}).get("statistical_fallback_k", 3.0)
        theta_star = float(np.mean(S_val) + k * np.std(S_val))
        print(f"[!] Single-class validation fallback applied: theta* = {theta_star:.6f}")

    val_preds = (S_val >= theta_star).astype(int)
    val_metrics = compute_classification_metrics(y_val, val_preds)

    threshold_data = {
        "optimal_threshold": float(theta_star),
        "validation_f1": float(val_metrics["f1_score"]),
        "validation_precision": float(val_metrics["precision"]),
        "validation_recall": float(val_metrics["recall"]),
        "validation_accuracy": float(val_metrics["accuracy"]),
        "validation_samples": len(y_val),
        "search_method": "f1_optimal",
        "search_steps": n_search_steps,
    }
    with open(exp_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2)

    # 2. Held-Out Test Evaluation
    print("[+] Evaluating held-out test partition with frozen threshold...")
    latencies_ms = []
    with torch.no_grad():
        for i in range(min(5, len(X_test))):
            warm_seq = torch.as_tensor(X_test[i : i + 1], dtype=torch.float32)
            _ = model(warm_seq, teacher_forcing_ratio=0.0)

        benchmark_n = min(100, len(X_test))
        for i in range(benchmark_n):
            sample_seq = torch.as_tensor(X_test[i : i + 1], dtype=torch.float32)
            t0 = time.perf_counter()
            _ = model(sample_seq, teacher_forcing_ratio=0.0)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        S_test = model.compute_anomaly_score(X_test, aggregation="mean")

    test_metrics = compute_all_metrics(y_test, S_test, threshold=theta_star)
    y_test_pred = (S_test >= theta_star).astype(int)

    mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    throughput = (1000.0 / mean_latency) if mean_latency > 0 else 0.0

    metrics_payload = {
        "experiment_id": f"{dataset_id}_{model_name}",
        "dataset": dataset_id,
        "model_architecture": model_name,
        "device": "cpu",
        "threshold": float(theta_star),
        "convergence_epoch": int(epoch),
        "total_parameters": summary["total_parameters"],
        "trainable_parameters": summary["trainable_parameters"],
        "mean_latency_ms": mean_latency,
        "throughput_seq_per_sec": throughput,
        "test_metrics": test_metrics,
        "training_mode": note,
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    np.savez_compressed(
        exp_dir / "predictions.npz",
        y_true=y_test,
        anomaly_scores=S_test,
        y_pred=y_test_pred,
        threshold=np.array([theta_star]),
    )

    meta = {
        "dataset_id": dataset_id,
        "model_name": model_name,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "training_protocol": note,
        "epoch": int(epoch),
        "test_samples": len(y_test),
        "normal_samples": int(np.sum(y_test == 0)),
        "anomaly_samples": int(np.sum(y_test == 1)),
    }
    with open(exp_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n================================================================================")
    print(f" [RESULT] {dataset_id} | {model_name} ({note})")
    print(f"  - Checkpoint Epoch  : {epoch}")
    print(f"  - Threshold (theta*): {theta_star:.6f}")
    print(f"  - Test Accuracy     : {test_metrics['accuracy']:.4f}")
    print(f"  - Test Precision    : {test_metrics['precision']:.4f}")
    print(f"  - Test Recall       : {test_metrics['recall']:.4f}")
    print(f"  - Test F1-Score     : {test_metrics['f1_score']:.4f}")
    print(f"  - Test ROC-AUC      : {test_metrics['roc_auc']:.4f}")
    print(f"  - Test PR-AUC       : {test_metrics['pr_auc']:.4f}")
    print(f"  - Mean Latency      : {mean_latency:.3f} ms/seq")
    print(f"================================================================================\n")
    return metrics_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--note", default="time_bounded_checkpoint_epoch_3")
    args = parser.parse_args()
    evaluate_checkpoint(args.dataset, args.model, note=args.note)
