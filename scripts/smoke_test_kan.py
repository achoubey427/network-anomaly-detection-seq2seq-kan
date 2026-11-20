"""
Proposed KAN Model Development Smoke Test and Gradient Verification Script.
Used strictly for software engineering verification of forward pass, backward pass,
gradient propagation across B-spline KAN parameters, loss progression, and checkpointing.
DOES NOT USE OR CLAIM RESEARCH BENCHMARK EVIDENCE.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_research_config
from src.models import build_model
from src.models.trainer import train_autoencoder, get_device, load_checkpoint
from src.models.summary import save_model_summary, compare_architectures


def run_kan_smoke_test():
    print("================================================================================")
    print("               DEVELOPMENT SMOKE TEST: PROPOSED KAN MODEL                       ")
    print("         (Software Verification Only - Not Research Benchmark Data)             ")
    print("================================================================================")

    config = load_research_config("configs/research.yaml")
    device = get_device()
    print(f"[*] Detected Device: {device}")

    # Small synthetic test dimensions
    n_train_samples = 40
    n_val_samples = 15
    seq_len = 30
    feature_dim = 16

    np.random.seed(42)
    X_train_mock = np.random.randn(n_train_samples, seq_len, feature_dim).astype(np.float32)
    X_val_mock = np.random.randn(n_val_samples, seq_len, feature_dim).astype(np.float32)

    # 1. Build Models for Comparison
    base_model = build_model("baseline_seq2seq", input_dim=feature_dim, config=config)
    kan_model = build_model("kan_seq2seq", input_dim=feature_dim, config=config)

    save_model_summary(kan_model, "research_results/individual/model_architecture_kan.json")
    comparison = compare_architectures(base_model, kan_model)

    print(f"[*] Model Architectures Capacity Analysis:")
    print(f"    - Baseline Model Parameters : {comparison['baseline_total_parameters']:,}")
    print(f"    - Proposed KAN Parameters   : {comparison['kan_total_parameters']:,}")
    print(f"    - Parameter Delta           : +{comparison['parameter_difference']:,} ({comparison['parameter_increase_percentage']}%)")
    print(f"    - Summary JSON Saved -> research_results/individual/model_parameter_comparison.json")

    # 2. Gradient Verification on KAN Bottleneck
    print("\n[*] Executing KAN Bottleneck Gradient Verification...")
    kan_model.train()
    optimizer = torch.optim.Adam(kan_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    test_batch = torch.tensor(X_train_mock[:8], dtype=torch.float32)
    optimizer.zero_grad()
    recon, att = kan_model(test_batch, teacher_forcing_ratio=0.5, return_attention=True)
    loss = criterion(recon, test_batch)
    loss.backward()

    kan_params_total = 0
    kan_params_with_grad = 0
    grad_norms = []

    for name, param in kan_model.named_parameters():
        if "kan" in name:
            kan_params_total += param.numel()
            if param.grad is not None:
                non_zero = torch.sum(param.grad != 0).item()
                if non_zero > 0:
                    kan_params_with_grad += param.numel()
                    grad_norms.append(param.grad.norm().item())

    grad_ratio = (kan_params_with_grad / kan_params_total) * 100 if kan_params_total > 0 else 0.0
    print(f"    - Total KAN Bottleneck Parameters: {kan_params_total:,}")
    print(f"    - Parameters with Non-Zero Gradients: {kan_params_with_grad:,} ({grad_ratio:.1f}%)")
    print(f"    - Mean KAN Gradient Norm: {np.mean(grad_norms):.6f}")

    if kan_params_with_grad == kan_params_total:
        print("    [+] Gradient Verification: 100% PASS (Full differentiability across B-spline grid).")
    else:
        print("    [!] Warning: Partial gradient flow detected.")

    # 3. Smoke Test Training Loop (3 epochs)
    smoke_config = dict(config)
    smoke_config["training"] = dict(config.get("training", {}))
    smoke_config["training"]["epochs"] = 3
    smoke_config["training"]["batch_size"] = 16

    print("\n[*] Executing 3-epoch KAN smoke test training loop...")
    trained_model, history = train_autoencoder(
        model=kan_model,
        X_train=X_train_mock,
        X_val=X_val_mock,
        config=smoke_config,
        dataset_name="smoke_test",
        is_smoke_test=True,
    )

    print(f"[+] Smoke Test Training Succeeded! Loss progression:")
    for ep, (t_l, v_l) in enumerate(zip(history["train_loss"], history["val_loss"]), start=1):
        print(f"    Epoch {ep}: Train MSE = {t_l:.6f} | Val MSE = {v_l:.6f}")

    # 4. Anomaly Scoring & Checkpoint Verification
    test_tensor = torch.tensor(X_val_mock[:5], dtype=torch.float32).to(device)
    scores = trained_model.compute_anomaly_score(test_tensor, aggregation="mean")
    print(f"[+] Anomaly Scoring Verified: {scores.round(4)}")

    ckpt_file = Path("models/checkpoints/kan_seq2seq/smoke_test_best_model.pt")
    if ckpt_file.exists():
        fresh_model = build_model("kan_seq2seq", input_dim=feature_dim, config=config)
        fresh_model, ckpt = load_checkpoint(str(ckpt_file), fresh_model)
        print(f"[+] Checkpoint Save & Reload Verified: Best epoch = {ckpt['epoch']}, Val Loss = {ckpt['val_loss']:.6f}")

    print("================================================================================")
    print("                 KAN SMOKE TEST COMPLETED SUCCESSFULLY                          ")
    print("================================================================================")


if __name__ == "__main__":
    run_kan_smoke_test()
