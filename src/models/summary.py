"""
Model Inspection, Parameter Counting, and Architecture Comparison Utility.
Generates structured JSON summaries for individual models and comparative analyses.
"""

import json
from pathlib import Path
from typing import Any, Dict
import torch


def summarize_model(model: Any, input_shape: tuple = (1, 30, 16)) -> Dict[str, Any]:
    """Inspect model architecture, count trainable parameters, and produce summary dictionary."""
    if not hasattr(model, "parameters"):
        return {"error": "Invalid model object"}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layers_info = []
    for name, module in model.named_children():
        mod_params = sum(p.numel() for p in module.parameters())
        layers_info.append({
            "layer_name": name,
            "layer_type": module.__class__.__name__,
            "parameters": mod_params
        })

    summary = {
        "model_class": model.__class__.__name__,
        "input_dimension": getattr(model, "input_dim", input_shape[-1]),
        "hidden_dimension": getattr(model, "hidden_dim", None),
        "latent_dimension": getattr(model, "latent_dim", None),
        "num_layers": getattr(model, "num_layers", None),
        "bidirectional": getattr(model, "bidirectional", False),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "estimated_model_size_mb": round((total_params * 4) / (1024 * 1024), 4),
        "layers": layers_info
    }
    return summary


def save_model_summary(model: Any, output_path: str):
    """Serialize model summary to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_model(model)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def compare_architectures(baseline_model: Any, kan_model: Any) -> Dict[str, Any]:
    """Compare parameter counts and structural capacity between Baseline and KAN models."""
    base_summary = summarize_model(baseline_model)
    kan_summary = summarize_model(kan_model)

    base_params = base_summary["total_parameters"]
    kan_params = kan_summary["total_parameters"]
    diff = kan_params - base_params
    pct_increase = round((diff / base_params) * 100, 2) if base_params > 0 else 0.0

    comparison = {
        "baseline_model": base_summary["model_class"],
        "proposed_kan_model": kan_summary["model_class"],
        "input_dimension": base_summary["input_dimension"],
        "hidden_dimension": base_summary["hidden_dimension"],
        "latent_dimension": base_summary["latent_dimension"],
        "baseline_total_parameters": base_params,
        "kan_total_parameters": kan_params,
        "parameter_difference": diff,
        "parameter_increase_percentage": pct_increase,
        "baseline_layers": base_summary["layers"],
        "kan_layers": kan_summary["layers"],
    }

    out_file = Path("research_results/individual/model_parameter_comparison.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    return comparison
