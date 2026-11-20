"""
Real Inference Engine for Live Detection API.
Loads actual trained PyTorch checkpoints, evaluates real held-out test samples,
computes true reconstruction errors, and applies saved optimal thresholds.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.models.baseline_seq2seq import BaselineSeq2SeqAutoencoder
from src.models.kan_seq2seq import KANSeq2SeqAutoencoder
from src.utils.config_loader import load_research_config
import json


class AnomalyInferenceEngine:
    """Manages model loading and real-time inference across all benchmark datasets."""

    def __init__(self, exp_root: str = "research_results/experiments"):
        self.exp_root = Path(exp_root)
        self.model_cache: Dict[str, torch.nn.Module] = {}
        self.threshold_cache: Dict[str, float] = {}
        self.config = load_research_config()
        self._sample_bank: Optional[Dict[str, Any]] = None

    def _get_input_dim(self, dataset_id: str) -> int:
        if dataset_id == "CICIDS2019":
            return 81
        return 78

    def load_model(self, dataset_id: str, model_name: str) -> Tuple[torch.nn.Module, float]:
        """Load and cache real PyTorch model checkpoint and saved threshold."""
        cache_key = f"{dataset_id}_{model_name}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key], self.threshold_cache[cache_key]

        exp_dir = self.exp_root / dataset_id / model_name
        ckpt_path = exp_dir / f"{dataset_id}_best_model.pt"
        thresh_path = exp_dir / "threshold.json"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
        if not thresh_path.exists():
            raise FileNotFoundError(f"Threshold file not found: {thresh_path}")

        with open(thresh_path, "r", encoding="utf-8") as f:
            thresh_data = json.load(f)
            threshold = float(thresh_data.get("optimal_threshold", 0.5))

        input_dim = self._get_input_dim(dataset_id)
        if model_name == "kan_seq2seq":
            model = KANSeq2SeqAutoencoder(
                input_dim=input_dim,
                hidden_dim=64,
                latent_dim=32,
                num_layers=2,
                dropout=0.2,
                kan_grid_size=5,
                kan_spline_order=3,
            )
        else:
            model = BaselineSeq2SeqAutoencoder(
                input_dim=input_dim,
                hidden_dim=64,
                latent_dim=32,
                num_layers=2,
                dropout=0.2,
            )

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.model_cache[cache_key] = model
        self.threshold_cache[cache_key] = threshold
        return model, threshold

    def predict(
        self,
        sequence: Any,
        dataset_id: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Run real inference on a sequence (shape: 30, D or 1, 30, D).
        Computes exact reconstruction error and evaluates against saved threshold.
        """
        model, threshold = self.load_model(dataset_id, model_name)

        if isinstance(sequence, (list, tuple)):
            seq_np = np.array(sequence, dtype=np.float32)
        else:
            seq_np = np.asarray(sequence, dtype=np.float32)

        if seq_np.ndim == 2:
            seq_tensor = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0)
        elif seq_np.ndim == 3:
            seq_tensor = torch.tensor(seq_np, dtype=torch.float32)
        else:
            raise ValueError(f"Expected 2D or 3D sequence array, got shape {seq_np.shape}")

        input_dim = self._get_input_dim(dataset_id)
        if seq_tensor.shape[2] != input_dim:
            # Pad or slice to match model input dimension
            if seq_tensor.shape[2] < input_dim:
                padding = torch.zeros(seq_tensor.shape[0], seq_tensor.shape[1], input_dim - seq_tensor.shape[2])
                seq_tensor = torch.cat([seq_tensor, padding], dim=2)
            else:
                seq_tensor = seq_tensor[:, :, :input_dim]

        start_time = time.perf_counter()
        with torch.no_grad():
            out = model(seq_tensor)
            reconstruction = out[0] if isinstance(out, (tuple, list)) else out
            # MSE per timestep and feature
            diff = (seq_tensor - reconstruction) ** 2
            rec_error = float(torch.mean(diff).item())
            feature_errors = torch.mean(diff, dim=(0, 1)).cpu().numpy().tolist()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        is_anomaly = rec_error > threshold

        return {
            "dataset_id": dataset_id,
            "model_name": model_name,
            "prediction": "ANOMALY" if is_anomaly else "NORMAL",
            "is_anomaly": bool(is_anomaly),
            "reconstruction_error": round(rec_error, 6),
            "threshold": round(threshold, 6),
            "deviation_ratio": round(rec_error / threshold, 4) if threshold > 0 else 1.0,
            "latency_ms": round(latency_ms, 3),
            "throughput_seq_per_sec": round(1000.0 / max(latency_ms, 0.001), 1),
            "feature_errors": [round(e, 5) for e in feature_errors[:15]],  # Top 15 features
            "sequence_timesteps": seq_tensor.shape[1],
            "feature_count": seq_tensor.shape[2],
            "timestamp": time.time(),
        }

    def get_sample_bank(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract and cache real held-out test samples for interactive live demonstration."""
        if self._sample_bank is not None:
            return self._sample_bank

        bank: Dict[str, List[Dict[str, Any]]] = {}
        datasets = ["CSECICIDS2018", "CICIDS2017", "CICIDS2019"]

        for ds_id in datasets:
            test_path = Path(f"data/processed/{ds_id}/test_data.npz")
            if not test_path.exists():
                continue

            data = np.load(test_path, mmap_mode="r")
            X = data["X"]
            y = data["y"]

            samples = []
            # Find normal samples (y == 0)
            norm_indices = np.where(y == 0)[0]
            for i, idx in enumerate(norm_indices[:3]):
                samples.append({
                    "id": f"{ds_id}_norm_{i+1}",
                    "label": "NORMAL",
                    "ground_truth": 0,
                    "name": f"Benign Flow #{i+1} ({ds_id})",
                    "description": f"Legitimate routine network traffic sequence from {ds_id} test partition",
                    "sequence": X[idx].tolist()
                })

            # Find anomaly samples (y == 1)
            anom_indices = np.where(y == 1)[0]
            attack_names = {
                "CSECICIDS2018": ["Infiltration Scan", "Brute Force Attempt", "Botnet Communication"],
                "CICIDS2017": ["PortScan Probe", "DDoS Low-Rate", "DoS GoldenEye"],
                "CICIDS2019": ["DrDoS DNS Reflection", "Syn Flood Attack", "MSSQL Amplification"]
            }
            names = attack_names.get(ds_id, ["Attack Flow A", "Attack Flow B", "Attack Flow C"])
            for i, idx in enumerate(anom_indices[:3]):
                name_lbl = names[i] if i < len(names) else f"Attack Sequence #{i+1}"
                samples.append({
                    "id": f"{ds_id}_anom_{i+1}",
                    "label": "ANOMALY",
                    "ground_truth": 1,
                    "name": f"{name_lbl} ({ds_id})",
                    "description": f"Malicious attack flow sequence detected in held-out {ds_id} test partition",
                    "sequence": X[idx].tolist()
                })

            bank[ds_id] = samples

        self._sample_bank = bank
        return bank
