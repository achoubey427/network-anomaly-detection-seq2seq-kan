"""
Model Trainer Module for Unsupervised Sequence Autoencoders.
Handles Adam optimization, annealed teacher forcing, validation monitoring,
early stopping, deterministic checkpoint saving, history logging, and loss plotting.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_device() -> Any:
    """Detect and return optimal computing device (CUDA or CPU)."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu") if TORCH_AVAILABLE else "cpu"


class EarlyStopping:
    """Early stopping monitor based on validation reconstruction loss."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # Improvement: save checkpoint
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class StreamingSequenceDataset(torch.utils.data.Dataset):
    """Zero-overhead dataset that slices batches on demand from numpy/memmap or tensor buffers."""
    def __init__(self, data_array):
        self.data_array = data_array

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        item = self.data_array[idx]
        if isinstance(item, torch.Tensor):
            return (item,)
        return (torch.tensor(item, dtype=torch.float32),)


def train_autoencoder(
    model: Any,
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: Dict[str, Any],
    dataset_name: str = "smoke_test",
    checkpoint_dir: Optional[str] = None,
    is_smoke_test: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Execute unsupervised autoencoder training loop.
    X_train must contain exclusively normal/benign sequences.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for model training.")

    device = get_device()
    model.to(device)

    model_dir_name = "kan_seq2seq" if "KAN" in model.__class__.__name__ else "baseline_seq2seq"
    model_prefix = "kan" if "KAN" in model.__class__.__name__ else "baseline"
    if checkpoint_dir is None:
        checkpoint_dir = f"models/checkpoints/{model_dir_name}"

    # Configuration extraction
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 64)
    epochs = train_cfg.get("epochs", 50)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 1e-5)
    initial_tf = train_cfg.get("teacher_forcing", {}).get("initial_ratio", 0.5)
    tf_decay = train_cfg.get("teacher_forcing", {}).get("decay_rate", 0.95)
    min_tf = train_cfg.get("teacher_forcing", {}).get("min_ratio", 0.0)
    patience = train_cfg.get("early_stopping", {}).get("patience", 7)
    seed = config.get("reproducibility", {}).get("seed", 42)

    # Convert to PyTorch DataLoaders with on-demand streaming
    val_batch_size = max(batch_size, 256)
    train_loader = DataLoader(StreamingSequenceDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(StreamingSequenceDataset(X_val), batch_size=val_batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)

    history = {
        "dataset": dataset_name,
        "model_type": model.__class__.__name__,
        "model_architecture": model_dir_name,
        "device": str(device),
        "random_seed": seed,
        "is_smoke_test": is_smoke_test,
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "teacher_forcing_ratios": [],
        "epoch_times_sec": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
    }

    current_tf = initial_tf
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    best_checkpoint_file = ckpt_path / f"{dataset_name}_best_model.pt"

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # 1. Training Phase
        model.train()
        running_train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x, teacher_forcing_ratio=current_tf)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_train_loss / len(X_train)

        # 2. Validation Phase (Zero teacher forcing)
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                recon = model(batch_x, teacher_forcing_ratio=0.0)
                val_loss = criterion(recon, batch_x)
                running_val_loss += val_loss.item() * batch_x.size(0)

        epoch_val_loss = running_val_loss / len(X_val)
        epoch_time = time.time() - epoch_start
        import gc
        gc.collect()

        # Record metrics
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])
        history["teacher_forcing_ratios"].append(current_tf)
        history["epoch_times_sec"].append(epoch_time)

        # Decay teacher forcing
        current_tf = max(min_tf, current_tf * tf_decay)

        # Checkpoint Saving on improvement
        is_best = early_stopping(epoch_val_loss)
        if is_best:
            history["best_epoch"] = epoch
            history["best_val_loss"] = epoch_val_loss
            checkpoint = {
                "epoch": epoch,
                "model_class": model.__class__.__name__,
                "model_architecture": model_dir_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": epoch_val_loss,
                "train_loss": epoch_train_loss,
                "config": config,
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "latent_dim": model.latent_dim,
                "num_layers": model.num_layers,
                "sequence_length": X_train.shape[1],
                "random_seed": seed,
                "timestamp": datetime.now().isoformat(),
                "is_smoke_test": is_smoke_test,
            }
            torch.save(checkpoint, best_checkpoint_file)

        if early_stopping.early_stop:
            print(f"[Early Stopping] Triggered at epoch {epoch}. Best epoch: {history['best_epoch']}")
            break

    total_duration = time.time() - start_total_time
    history["total_duration_sec"] = total_duration

    # Save training history JSON
    log_dir = Path("research_results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    history_file = log_dir / f"{model_prefix}_{dataset_name}_training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Generate Loss Curve Plot
    fig_dir = Path("research_results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_file = fig_dir / f"{model_prefix}_{dataset_name}_training_loss.png"

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Train Loss (MSE)", color="#2563eb", lw=2)
    plt.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="Val Loss (MSE)", color="#dc2626", lw=2)
    title_suffix = " (DEVELOPMENT SMOKE TEST)" if is_smoke_test else ""
    plt.title(f"{model.__class__.__name__} Training Curve - {dataset_name}{title_suffix}", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Reconstruction MSE Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_file, dpi=200)
    plt.close()

    return model, history


def load_checkpoint(checkpoint_path: str, model: Any) -> Tuple[Any, Dict[str, Any]]:
    """Load model weights and metadata from checkpoint."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required to load checkpoint.")
    checkpoint = torch.load(checkpoint_path, map_location=get_device())
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
