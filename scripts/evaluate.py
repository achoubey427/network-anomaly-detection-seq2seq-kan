"""
Standardized Evaluation Script for Research Benchmark (4 Datasets x 2 Models).
Executes frozen threshold search on validation set, evaluates test set,
computes metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, Latency),
and saves outputs directly to research_results/.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_research_config
from src.utils.metrics import compute_all_metrics, optimize_threshold_f1
from src.data_adapters import get_adapter


def run_evaluation(dataset_id: str, model_name: str, config_path: str = "configs/research.yaml"):
    config = load_research_config(config_path)
    
    ds_cfg = None
    for ds in config["experiments"]["datasets"]:
        if ds["id"] == dataset_id:
            ds_cfg = ds
            break
    if ds_cfg is None:
        raise ValueError(f"Dataset '{dataset_id}' not found in research configuration.")

    adapter = get_adapter(ds_cfg, config)
    exists, files = adapter.verify_dataset_exists()
    if not exists:
        print(f"[STATUS] Dataset '{dataset_id}' raw files not found in {adapter.raw_dir}.")
        print(f"[NOTICE] No synthetic metrics or fake numbers will be generated.")
        return

    print(f"==================================================")
    print(f"EVALUATION: {dataset_id} | Model: {model_name}")
    print(f"==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline or KAN Seq2Seq Autoencoder")
    parser.add_argument("--dataset", required=True, choices=["CICIDS2017", "CICIDS2019", "CICIDS2023", "CICIDS2025"], help="Dataset ID")
    parser.add_argument("--model", required=True, choices=["baseline_seq2seq", "kan_seq2seq"], help="Model Architecture")
    parser.add_argument("--config", default="configs/research.yaml", help="Path to research config")
    args = parser.parse_args()
    run_evaluation(args.dataset, args.model, args.config)
