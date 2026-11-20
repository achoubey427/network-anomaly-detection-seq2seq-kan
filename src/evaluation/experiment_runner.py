"""
Controlled Experiment Runner Engine for 4x2 Benchmark Matrix.
Executes standardized training, validation threshold optimization, frozen test set evaluation,
metric computation, latency benchmarking, and structured artifact persistence.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.utils.config_loader import load_research_config
from src.utils.metrics import compute_all_metrics, optimize_threshold_f1, compute_classification_metrics
from src.data_adapters.dataset_manager import validate_dataset, load_processed_dataset, prepare_dataset
from src.models import build_model
from src.models.trainer import train_autoencoder, get_device, load_checkpoint
from src.models.summary import summarize_model


class ExperimentRunner:
    """Manages and executes controlled research experiments with strict partition isolation."""

    def __init__(self, config_path: str = "configs/research.yaml"):
        self.config_path = config_path
        self.config = load_research_config(config_path)
        self.seed = self.config.get("reproducibility", {}).get("seed", 42)
        self.device = get_device()

    def run_experiment(
        self,
        dataset_id: str,
        model_name: str,
        custom_data: Optional[Dict[str, Any]] = None,
        is_smoke_test: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute an end-to-end controlled experiment.
        Guarantees zero data leakage: validation threshold is frozen before single test evaluation.
        """
        start_time = time.time()
        exp_dir = Path(f"research_results/experiments/{dataset_id}/{model_name}")
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Dataset Availability & Verification
        if custom_data is None:
            status, reason = validate_dataset(dataset_id, self.config)
            if status == "MISSING":
                print(f"[-] Experiment Aborted: Dataset '{dataset_id}' is MISSING.")
                print(f"[-] Reason: {reason}")
                print(f"[-] No synthetic/fake research metrics will be generated.")
                result_metadata = {
                    "experiment_id": f"{dataset_id}_{model_name}",
                    "dataset_id": dataset_id,
                    "model_name": model_name,
                    "status": "DATASET_MISSING",
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(exp_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(result_metadata, f, indent=2)
                return result_metadata
            elif status == "INCOMPATIBLE":
                print(f"[-] Experiment Aborted: Dataset '{dataset_id}' is INCOMPATIBLE.")
                result_metadata = {
                    "experiment_id": f"{dataset_id}_{model_name}",
                    "dataset_id": dataset_id,
                    "model_name": model_name,
                    "status": "INCOMPATIBLE",
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(exp_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(result_metadata, f, indent=2)
                return result_metadata

            # Load or prepare processed dataset
            proc_dir = Path(f"data/processed/{dataset_id}")
            if not (proc_dir / "train_data.npz").exists():
                print(f"[+] Processed data not found. Preparing dataset '{dataset_id}'...")
                data = prepare_dataset(dataset_id, self.config, mode="research")
            else:
                print(f"[+] Loading processed partitions from {proc_dir}...")
                data = load_processed_dataset(dataset_id)
        else:
            data = custom_data

        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]

        input_dim = X_train.shape[-1]
        seq_len = X_train.shape[1]

        print(f"================================================================================")
        print(f" EXPERIMENT: {dataset_id} | Model: {model_name} (Device: {self.device})")
        print(f" Partition Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"================================================================================")

        # 2. Build Model Architecture
        model = build_model(model_name, input_dim=input_dim, config=self.config)
        summary = summarize_model(model, input_shape=(1, seq_len, input_dim))

        # 3. Train Autoencoder strictly on Benign Data (y_train == 0)
        trained_model, history = train_autoencoder(
            model=model,
            X_train=X_train,
            X_val=X_val,
            config=self.config,
            dataset_name=dataset_id,
            checkpoint_dir=str(exp_dir),
            is_smoke_test=is_smoke_test,
        )

        # 4. Validation Anomaly Scoring & Optimal Threshold Search (Frozen Selection)
        print("[+] Scoring validation set and searching optimal decision threshold (theta*)...")
        S_val = trained_model.compute_anomaly_score(X_val, aggregation="mean")

        # Threshold optimization strictly on validation labels
        n_search_steps = self.config.get("thresholding", {}).get("search_steps", 500)
        theta_star, val_f1 = optimize_threshold_f1(y_val, S_val, n_steps=n_search_steps)

        # If validation has only 1 class, apply statistical fallback
        if val_f1 == 0.0 and len(np.unique(y_val)) == 1:
            k = self.config.get("thresholding", {}).get("statistical_fallback_k", 3.0)
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

        # 5. Held-out Test Evaluation with Frozen Threshold theta* (Executed ONCE)
        print("[+] Evaluating held-out test partition with frozen threshold...")
        
        # Benchmark single-sequence inference latency
        latencies_ms = []
        trained_model.eval()
        with torch.no_grad():
            # Warm up
            for i in range(min(5, len(X_test))):
                warm_seq = torch.as_tensor(X_test[i : i + 1], dtype=torch.float32).to(self.device)
                _ = trained_model(warm_seq, teacher_forcing_ratio=0.0)

            # Measure individual sequence latency
            benchmark_n = min(100, len(X_test))
            for i in range(benchmark_n):
                sample_seq = torch.as_tensor(X_test[i : i + 1], dtype=torch.float32).to(self.device)
                t0 = time.perf_counter()
                _ = trained_model(sample_seq, teacher_forcing_ratio=0.0)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            S_test = trained_model.compute_anomaly_score(X_test, aggregation="mean")

        # Compute full suite of test metrics
        test_metrics = compute_all_metrics(y_test, S_test, threshold=theta_star)
        test_preds = (S_test >= theta_star).astype(int)

        # Add operational latency metrics
        test_metrics["mean_latency_ms"] = float(np.mean(latencies_ms))
        test_metrics["median_latency_ms"] = float(np.median(latencies_ms))
        test_metrics["p95_latency_ms"] = float(np.percentile(latencies_ms, 95))
        test_metrics["p99_latency_ms"] = float(np.percentile(latencies_ms, 99))
        test_metrics["throughput_seq_per_sec"] = float(1000.0 / np.mean(latencies_ms)) if np.mean(latencies_ms) > 0 else 0.0

        total_runtime = time.time() - start_time

        # 6. Save Structured Experiment Artifacts
        metadata = {
            "experiment_id": f"{dataset_id}_{model_name}",
            "dataset_id": dataset_id,
            "model_name": model_name,
            "status": "COMPLETED",
            "timestamp": datetime.now().isoformat(),
            "execution_duration_sec": total_runtime,
            "device": str(self.device),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "random_seed": self.seed,
            "total_parameters": summary["total_parameters"],
            "trainable_parameters": summary["trainable_parameters"],
            "input_dimension": input_dim,
            "sequence_length": seq_len,
            "is_smoke_test": is_smoke_test,
        }

        # Save config.json
        with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        # Save training_history.json
        with open(exp_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Save threshold.json
        with open(exp_dir / "threshold.json", "w", encoding="utf-8") as f:
            json.dump(threshold_data, f, indent=2)

        # Save metrics.json
        with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        # Save experiment_metadata.json
        with open(exp_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Save predictions.npz
        np.savez_compressed(
            exp_dir / "predictions.npz",
            y_true=y_test,
            y_pred=test_preds,
            anomaly_scores=S_test,
            threshold=theta_star,
        )

        print(f"[+] Experiment Finished Successfully: {dataset_id} | {model_name}")
        print(f"    - Threshold (theta*) : {theta_star:.6f}")
        print(f"    - Test Accuracy      : {test_metrics['accuracy']:.4f}")
        print(f"    - Test Precision     : {test_metrics['precision']:.4f}")
        print(f"    - Test Recall        : {test_metrics['recall']:.4f}")
        print(f"    - Test F1-Score      : {test_metrics['f1_score']:.4f}")
        print(f"    - Test ROC-AUC       : {test_metrics['roc_auc']:.4f}")
        print(f"    - Test PR-AUC        : {test_metrics['pr_auc']:.4f}")
        print(f"    - Mean Latency       : {test_metrics['mean_latency_ms']:.3f} ms/seq")
        print(f"    - Artifacts Path     : {exp_dir}")

        return {
            "metadata": metadata,
            "metrics": test_metrics,
            "threshold": threshold_data,
            "history": history,
            "artifacts_dir": str(exp_dir),
        }
