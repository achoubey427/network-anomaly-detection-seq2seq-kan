"""
Base Dataset Adapter Class.
Defines abstract interface for dataset ingestion, column normalization,
missing/infinite value cleaning, label binarization, feature scaling,
and sliding-window sequence generation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import os
import csv
from pathlib import Path


class BaseDatasetAdapter(ABC):
    """Abstract base class for all dataset adapters in the benchmark matrix."""

    def __init__(self, dataset_config: Dict[str, Any], global_config: Dict[str, Any]):
        self.dataset_config = dataset_config
        self.global_config = global_config
        self.dataset_id = dataset_config.get("id", "UNKNOWN")
        self.raw_dir = Path(dataset_config.get("raw_dir", f"data/raw/{self.dataset_id}"))
        self.processed_dir = Path(dataset_config.get("processed_dir", f"data/processed/{self.dataset_id}"))
        self.label_column = dataset_config.get("label_column", "Label")
        self.normal_labels = set(str(lbl).strip().lower() for lbl in dataset_config.get("normal_labels", ["benign", "0"]))
        
        # Scaling statistics
        self.feature_medians: Optional[np.ndarray] = None
        self.feature_iqr: Optional[np.ndarray] = None
        self.feature_min: Optional[np.ndarray] = None
        self.feature_max: Optional[np.ndarray] = None
        self.numerical_cols: List[str] = dataset_config.get("features", {}).get("numerical", [])
        self.categorical_cols: List[str] = dataset_config.get("features", {}).get("categorical", [])

    def verify_dataset_exists(self) -> Tuple[bool, List[Path]]:
        """Check if raw dataset files exist on disk."""
        if not self.raw_dir.exists():
            return False, []
        files = list(self.raw_dir.glob("*.csv")) + list(self.raw_dir.glob("*.parquet")) + list(self.raw_dir.glob("*.txt"))
        return len(files) > 0, files

    @abstractmethod
    def detect_schema(self, sample_file: Path) -> Dict[str, Any]:
        """Inspect a data file and detect available columns and types."""
        pass

    def clean_numerical_array(self, arr: np.ndarray) -> np.ndarray:
        """Handle NaN, Inf, -Inf and clip extreme outliers."""
        arr = arr.astype(np.float32)
        # Replace Inf with NaN temporarily
        arr[np.isinf(arr)] = np.nan
        
        # Column-wise median imputation
        col_medians = np.nanmedian(arr, axis=0)
        # Fallback for columns with all NaN
        col_medians[np.isnan(col_medians)] = 0.0
        
        # Impute
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_medians, inds[1])
        return arr

    def binarize_labels(self, labels: List[str]) -> np.ndarray:
        """Convert multi-class / string labels to binary: 0 = Benign/Normal, 1 = Anomaly/Attack."""
        binary_labels = []
        for lbl in labels:
            cleaned = str(lbl).strip().lower()
            if cleaned in self.normal_labels:
                binary_labels.append(0)
            else:
                binary_labels.append(1)
        return np.array(binary_labels, dtype=np.int32)

    def fit_scaler(self, x_normal_train: np.ndarray):
        """Fit scaling parameters exclusively on normal training data."""
        self.feature_medians = np.median(x_normal_train, axis=0)
        q75, q25 = np.percentile(x_normal_train, [75, 25], axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0  # Prevent divide by zero
        self.feature_iqr = iqr

    def transform_features(self, x: np.ndarray) -> np.ndarray:
        """Scale features using fitted statistics."""
        if self.feature_medians is None or self.feature_iqr is None:
            raise ValueError("Scaler must be fitted before transforming.")
        scaled = (x - self.feature_medians) / self.feature_iqr
        # Outlier clipping
        clip_range = self.global_config.get("preprocessing", {}).get("outlier_clip", [-5.0, 5.0])
        return np.clip(scaled, clip_range[0], clip_range[1])

    def create_sequences(self, features: np.ndarray, labels: np.ndarray, window_size: int = 30, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sliding-window sequences with direct numpy preallocation.
        Sequence label is 1 if any sample in the window is anomalous, else 0.
        """
        n_samples = len(features)
        if n_samples < window_size:
            return np.empty((0, window_size, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int32)
            
        n_seq = (n_samples - window_size) // stride + 1
        n_feats = features.shape[1]
        sequences = np.empty((n_seq, window_size, n_feats), dtype=np.float32)
        seq_labels = np.empty(n_seq, dtype=np.int32)
        
        for idx, i in enumerate(range(0, n_samples - window_size + 1, stride)):
            sequences[idx] = features[i: i + window_size]
            seq_labels[idx] = 1 if np.any(labels[i: i + window_size] == 1) else 0
            
        return sequences, seq_labels
