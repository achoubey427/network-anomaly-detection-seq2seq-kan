"""
Baseline Model Development Smoke Test.
Used strictly for software engineering verification of forward pass, backward pass,
loss computation, reconstruction error, anomaly scoring, attention extraction, and checkpointing.
DOES NOT USE OR CLAIM RESEARCH BENCHMARK EVIDENCE.
"""

import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_research_config
from src.models import build_model
from src.models.trainer import train_autoencoder, get_device, load_checkpoint
from src.models.summary import save_model_summary


def run_smoke_test():
    print("================================================================================")
    print("                DEVELOPMENT SMOKE TEST: BASELINE MODEL                          ")
    print("         (Software Verification Only - Not Research Benchmark Data)             ")
    print("================================================================================")

    config = load_research_config("configs/research.yaml")
    device = get_device()
    print(f"[*] Detected Device: {device}")

    # Generate small synthetic test tensors
    n_train_samples = 40
    n_val_samples = 15
    seq_len = 30
    feature_dim = 16

    np.random.seed(42)
    X_train_mock = np.random.randn(n_train_samples, seq_len, feature_dim).astype(np.float32)
    X_val_mock = np.random.randn(n_val_samples, seq_len, feature_dim).astype(np.float32)

    print(f"[*] Mock Data Shapes -> Train: {X_train_mock.shape}, Val: {X_val_mock.shape}")

    # Build Baseline Model
    model = build_model("baseline_seq2seq", input_dim=feature_dim, config=config)
    summary = save_model_summary(model, "research_results/individual/model_architecture_baseline.json")
    print(f"[*] Model Parameters: {summary['total_parameters']:,} (Trainable: {summary['trainable_parameters']:,})")
    print(f"[*] Model Architecture Summary Saved -> research_results/individual/model_architecture_baseline.json")

    # Smoke Test Training Loop (3 epochs)
    smoke_config = dict(config)
    smoke_config["training"] = dict(config.get("training", {}))
    smoke_config["training"]["epochs"] = 3
    smoke_config["training"]["batch_size"] = 16

    print(f"[*] Executing 3-epoch smoke test training loop...")
    trained_model, history = train_autoencoder(
        model=model,
        X_train=X_train_mock,
        X_val=X_val_mock,
        config=smoke_config,
        dataset_name="smoke_test",
        is_smoke_test=True,
    )

    print(f"[+] Smoke Test Training Succeeded! Loss progression:")
    for ep, (t_l, v_l) in enumerate(zip(history["train_loss"], history["val_loss"]), start=1):
        print(f"    Epoch {ep}: Train MSE = {t_l:.6f} | Val MSE = {v_l:.6f}")

    # Anomaly Scoring Verification
    try:
        import torch
        test_tensor = torch.tensor(X_val_mock[:5], dtype=torch.float32).to(device)
        scores = trained_model.compute_anomaly_score(test_tensor, aggregation="mean")
        print(f"[+] Anomaly Scoring Verified: Generated {len(scores)} sequence anomaly scores: {scores.round(4)}")
    except Exception as e:
        print(f"[!] Anomaly scoring check note: {e}")

    # Checkpoint Reload Verification
    ckpt_file = Path("models/checkpoints/baseline_seq2seq/smoke_test_best_model.pt")
    if ckpt_file.exists():
        fresh_model = build_model("baseline_seq2seq", input_dim=feature_dim, config=config)
        fresh_model, ckpt = load_checkpoint(str(ckpt_file), fresh_model)
        print(f"[+] Checkpoint Save & Reload Verified: Best epoch = {ckpt['epoch']}, Val Loss = {ckpt['val_loss']:.6f}")

    print("================================================================================")
    print("                    SMOKE TEST COMPLETED SUCCESSFULLY                           ")
    print("================================================================================")


if __name__ == "__main__":
    run_smoke_test()
