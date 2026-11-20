"""
Models Package for ICS Anomaly Detection Research Benchmark.
Provides Baseline Seq2Seq LSTM-Attention and Proposed KAN-enhanced Seq2Seq Autoencoders.
"""

from typing import Any, Dict
from src.models.baseline_seq2seq import BaselineSeq2SeqAutoencoder
from src.models.kan_seq2seq import KANSeq2SeqAutoencoder


def build_model(model_name: str, input_dim: int, config: Dict[str, Any]):
    """
    Factory function to build Baseline or KAN Seq2Seq model based on configuration.
    Both models share identical input, hidden, latent, and layer parameters.
    """
    model_cfg = config.get("model", {})
    kan_cfg = config.get("kan", {})

    hidden_dim = model_cfg.get("hidden_dim", 64)
    latent_dim = model_cfg.get("latent_dim", 32)
    num_layers = model_cfg.get("num_layers", 2)
    dropout = model_cfg.get("dropout", 0.2)
    bidirectional = model_cfg.get("bidirectional_encoder", False)

    if model_name in ["kan_seq2seq", "kan", "proposed_kan"]:
        return KANSeq2SeqAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            kan_grid_size=kan_cfg.get("grid_size", 5),
            kan_spline_order=kan_cfg.get("spline_order", 3),
            kan_grid_range=tuple(kan_cfg.get("grid_range", [-1.0, 1.0])),
        )
    elif model_name in ["baseline_seq2seq", "baseline", "Seq2SeqAutoencoder"]:
        return BaselineSeq2SeqAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. Available: 'baseline_seq2seq', 'kan_seq2seq'"
        )
