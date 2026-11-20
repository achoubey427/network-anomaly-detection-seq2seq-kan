"""
Standardized Training Script for Research Benchmark (4 Datasets x 2 Models).
Loads dataset via adapter or processed tensors, initializes model architecture,
and runs unsupervised benign-only training with validation early stopping.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_research_config
from src.data_adapters.dataset_manager import validate_dataset, load_processed_dataset, prepare_dataset
from src.models import build_model
from src.models.trainer import train_autoencoder, get_device
from src.models.summary import save_model_summary


def run_training(dataset_id: str, model_name: str, config_path: str = "configs/research.yaml"):
    config = load_research_config(config_path)

    # 1. Dataset Verification
    status, reason = validate_dataset(dataset_id, config)
    if status == "MISSING":
        print(f"[STATUS] Dataset '{dataset_id}' raw files not found in data/raw/{dataset_id}.")
        print(f"[ACTION] Place raw CSV files in data/raw/{dataset_id}/ to execute training.")
        print("[NOTICE] Aborting run cleanly without fabrication.")
        return False
    elif status == "INCOMPATIBLE":
        print(f"[STATUS] Dataset '{dataset_id}' is incompatible: {reason}")
        return False

    print(f"==================================================")
    print(f"TRAINING START: {dataset_id} | Model: {model_name}")
    print(f"Device: {get_device()} | Seed: {config.get('reproducibility', {}).get('seed', 42)}")
    print(f"==================================================")

    # 2. Load Processed Data (or prepare on-the-fly)
    proc_dir = Path("data/processed") / dataset_id
    if not (proc_dir / "train_data.npz").exists():
        print(f"[+] Processed data not found. Running preparation pipeline...")
        data = prepare_dataset(dataset_id, config, mode="research")
    else:
        print(f"[+] Loading prepared dataset from {proc_dir}...")
        data = load_processed_dataset(dataset_id)

    X_train = data["X_train"]
    X_val = data["X_val"]
    input_dim = X_train.shape[-1]
    seq_len = X_train.shape[1]

    # 3. Model Build & Summary
    model = build_model(model_name, input_dim=input_dim, config=config)
    save_model_summary(model, f"research_results/individual/model_architecture_{model_name}_{dataset_id}.json")

    # 4. Train Autoencoder
    trained_model, history = train_autoencoder(
        model=model,
        X_train=X_train,
        X_val=X_val,
        config=config,
        dataset_name=dataset_id,
        is_smoke_test=False,
    )

    print(f"[+] Training completed! Best epoch: {history['best_epoch']} | Best Val Loss: {history['best_val_loss']:.6f}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline or KAN Seq2Seq Autoencoder")
    parser.add_argument("--dataset", required=True, choices=["CICIDS2017", "CICIDS2019", "CICIDS2023", "CICIDS2025"], help="Dataset ID")
    parser.add_argument("--model", default="baseline_seq2seq", choices=["baseline_seq2seq", "kan_seq2seq"], help="Model Architecture")
    parser.add_argument("--config", default="configs/research.yaml", help="Path to research config")
    args = parser.parse_args()
    run_training(args.dataset, args.model, args.config)
