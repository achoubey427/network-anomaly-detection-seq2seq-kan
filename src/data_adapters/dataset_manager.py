"""
Unified Dataset Preparation, Ingestion, and Inspection API.
Supports chunked reading, memory-efficient float32 casting, leakage protection,
scaler serialization, sequence generation, and reproducible NPZ export.
"""

import csv
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.data_adapters import get_adapter
from src.data_adapters.canonical_features import map_raw_columns, sanitize_column_name
from src.data_adapters.leakage_detector import audit_leakage_all


def inspect_dataset(dataset_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspect raw dataset files and compute profiling metadata.
    Uses chunked reading to handle large CSV files safely without memory exhaustion.
    """
    ds_cfg = next((d for d in config["experiments"]["datasets"] if d["id"] == dataset_id), None)
    if ds_cfg is None:
        raise ValueError(f"Dataset '{dataset_id}' not found in configuration.")

    adapter = get_adapter(ds_cfg, config)
    exists, files = adapter.verify_dataset_exists()

    if not exists:
        return {
            "dataset_id": dataset_id,
            "status": "MISSING",
            "found": False,
            "message": f"Raw files not found in {adapter.raw_dir}. No experiment will be executed.",
            "files": [],
            "row_count": 0,
            "column_count": 0,
        }

    total_rows = 0
    total_missing = 0
    total_inf = 0
    class_counts: Dict[str, int] = {}
    detected_columns: List[str] = []
    file_summaries: List[Dict[str, Any]] = []

    # Read and inspect files
    for file_path in files:
        file_size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
        file_rows = 0
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            headers = [h.strip() for h in next(reader, [])]
            if not detected_columns:
                detected_columns = headers

            label_idx = -1
            for idx, col in enumerate(headers):
                if col.lower() in ["label", "class", "attack", "is_attack"]:
                    label_idx = idx
                    break

            for row in reader:
                if not row or len(row) != len(headers):
                    continue
                file_rows += 1
                total_rows += 1

                # Label distribution
                if label_idx != -1 and label_idx < len(row):
                    raw_lbl = row[label_idx].strip()
                    class_counts[raw_lbl] = class_counts.get(raw_lbl, 0) + 1

                # Check missing / inf tokens
                for cell in row:
                    c_clean = cell.strip().lower()
                    if c_clean in ["", "nan", "null", "none"]:
                        total_missing += 1
                    elif c_clean in ["inf", "-inf", "infinity", "-infinity"]:
                        total_inf += 1

        file_summaries.append({
            "filename": file_path.name,
            "size_mb": file_size_mb,
            "rows": file_rows
        })

    # Class percentages
    class_distribution = {}
    for lbl, count in class_counts.items():
        class_distribution[lbl] = {
            "count": count,
            "percentage": round((count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        }

    # Identify candidate ID, constant, and flow columns
    retained_raw, canonical_names, discarded = map_raw_columns(detected_columns, dataset_id)

    return {
        "dataset_id": dataset_id,
        "status": "READY" if total_rows > 0 else "MISSING",
        "found": True,
        "raw_dir": str(adapter.raw_dir),
        "files_count": len(files),
        "files": file_summaries,
        "row_count": total_rows,
        "column_count": len(detected_columns),
        "columns": detected_columns,
        "retained_feature_count": len(retained_raw),
        "retained_features": canonical_names,
        "discarded_columns": discarded,
        "missing_values_count": total_missing,
        "infinite_values_count": total_inf,
        "unique_classes": len(class_counts),
        "class_distribution": class_distribution,
    }


def validate_dataset(dataset_id: str, config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Validate readiness of a dataset. Returns (status, reason).
    Status is one of: READY, READY_WITH_WARNINGS, INCOMPATIBLE, MISSING.
    """
    ds_cfg = next((d for d in config["experiments"]["datasets"] if d["id"] == dataset_id), None)
    if ds_cfg is None:
        return "INCOMPATIBLE", f"Dataset '{dataset_id}' not defined in config."

    adapter = get_adapter(ds_cfg, config)
    exists, files = adapter.verify_dataset_exists()
    if not exists:
        return "MISSING", f"Dataset not found in {adapter.raw_dir} — no experiment will be executed."

    # Validate header compatibility
    sample_file = files[0]
    schema = adapter.detect_schema(sample_file)
    if schema["total_columns"] < 5:
        return "INCOMPATIBLE", f"File {sample_file.name} has only {schema['total_columns']} columns. Incompatible format."

    if not schema.get("detected_label_column"):
        return "READY_WITH_WARNINGS", "Label column not automatically detected; fallback configuration applied."

    return "READY", f"Dataset verified: {len(files)} files found, compatible schema."


def prepare_dataset(
    dataset_id: str,
    config: Dict[str, Any],
    mode: str = "research",
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full reproducible dataset preprocessing pipeline:
    1. Read and clean raw data
    2. Normalize labels (preserve raw labels, create binary labels)
    3. Align canonical features, strip non-learning / ID columns
    4. Deterministic 70/15/15 train/val/test partition
    5. Fit scaler strictly on normal training data (y=0)
    6. Generate sliding-window sequence tensors
    7. Execute data leakage audit
    """
    ds_cfg = next((d for d in config["experiments"]["datasets"] if d["id"] == dataset_id), None)
    if ds_cfg is None:
        raise ValueError(f"Dataset '{dataset_id}' not found in configuration.")

    adapter = get_adapter(ds_cfg, config)
    exists, files = adapter.verify_dataset_exists()
    if not exists:
        raise FileNotFoundError(f"Dataset '{dataset_id}' files missing in {adapter.raw_dir}.")

    # Inspect headers and feature mapping first
    headers = []
    label_idx = -1
    with open(files[0], "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader, [])]
        for idx, col in enumerate(headers):
            if col.lower() in ["label", "class", "attack", "is_attack"]:
                label_idx = idx
                break

    feature_headers = [h for i, h in enumerate(headers) if i != label_idx] if label_idx != -1 else headers
    retained_raw, canonical_names, discarded = map_raw_columns(feature_headers, dataset_id)
    # Map retained feature headers back to raw column indices in the CSV row
    retained_raw_indices = []
    for c in retained_raw:
        if c in headers:
            retained_raw_indices.append(headers.index(c))

    # Check if max_samples_per_file or sample_size is configured to protect memory
    max_samples_per_file = config.get("preprocessing", {}).get("max_samples_per_file", None)
    if sample_size is not None and sample_size > 0:
        max_samples_per_file = max(1, sample_size // len(files))

    # Fast row counting to preallocate numpy array with balanced representation across all files
    file_row_targets = []
    total_valid_rows = 0
    for file_path in files:
        file_valid = 0
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row and len(row) == len(headers):
                    file_valid += 1
                    if max_samples_per_file and file_valid >= max_samples_per_file:
                        break
        file_row_targets.append(file_valid)
        total_valid_rows += file_valid

    total_raw_rows = total_valid_rows
    n_features = len(retained_raw_indices)
    X_raw = np.empty((total_valid_rows, n_features), dtype=np.float32)
    raw_labels = [""] * total_valid_rows

    # Populate preallocated array directly across all files
    curr_row = 0
    for file_idx, file_path in enumerate(files):
        target_for_this_file = file_row_targets[file_idx]
        file_curr = 0
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row or len(row) != len(headers):
                    continue
                if file_curr >= target_for_this_file or curr_row >= total_valid_rows:
                    break

                raw_labels[curr_row] = row[label_idx].strip() if label_idx != -1 else "BENIGN"
                for j, raw_idx in enumerate(retained_raw_indices):
                    val_str = row[raw_idx].strip()
                    try:
                        X_raw[curr_row, j] = float(val_str)
                    except ValueError:
                        X_raw[curr_row, j] = np.nan
                curr_row += 1
                file_curr += 1

    X_cleaned = adapter.clean_numerical_array(X_raw)
    y_binary = adapter.binarize_labels(raw_labels)

    # Free unneeded initial arrays
    del X_raw
    import gc
    gc.collect()

    # Deterministic Split: 70% Train, 15% Val, 15% Test
    seed = config.get("reproducibility", {}).get("seed", 42)
    rng = np.random.RandomState(seed)
    n_samples = len(X_cleaned)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)

    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # For anomaly detection, filter training set strictly to BENIGN samples
    train_normal_mask = (y_binary[train_idx] == 0)
    train_idx_normal = train_idx[train_normal_mask]

    X_train_raw = X_cleaned[train_idx_normal]
    y_train_raw = y_binary[train_idx_normal]

    X_val_raw = X_cleaned[val_idx]
    y_val_raw = y_binary[val_idx]

    X_test_raw = X_cleaned[test_idx]
    y_test_raw = y_binary[test_idx]

    del X_cleaned
    gc.collect()

    # Scaler fitting strictly on normal training data
    adapter.fit_scaler(X_train_raw)
    
    # Target output directory
    out_dir = Path(adapter.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sequence Generation (W=30)
    window_size = config.get("preprocessing", {}).get("window_size", 30)
    stride_train = config.get("preprocessing", {}).get("stride_train", 5)
    stride_eval = config.get("preprocessing", {}).get("stride_eval", 1)

    # 1. Train partition
    X_train_scaled = adapter.transform_features(X_train_raw)
    del X_train_raw
    gc.collect()
    X_train_seq, y_train_seq = adapter.create_sequences(X_train_scaled, y_train_raw, window_size, stride_train)
    del X_train_scaled
    gc.collect()
    np.savez_compressed(out_dir / "train_data.npz", X=X_train_seq, y=y_train_seq)
    train_seq_count = len(X_train_seq)
    del X_train_seq, y_train_seq
    gc.collect()

    # 2. Validation partition
    X_val_scaled = adapter.transform_features(X_val_raw)
    del X_val_raw
    gc.collect()
    X_val_seq, y_val_seq = adapter.create_sequences(X_val_scaled, y_val_raw, window_size, stride_eval)
    del X_val_scaled
    gc.collect()
    np.savez_compressed(out_dir / "val_data.npz", X=X_val_seq, y=y_val_seq)
    val_seq_count = len(X_val_seq)
    val_normal_count = int(np.sum(y_val_seq == 0))
    val_anomaly_count = int(np.sum(y_val_seq == 1))
    del X_val_seq, y_val_seq
    gc.collect()

    # 3. Test partition
    X_test_scaled = adapter.transform_features(X_test_raw)
    del X_test_raw
    gc.collect()
    X_test_seq, y_test_seq = adapter.create_sequences(X_test_scaled, y_test_raw, window_size, stride_eval)
    del X_test_scaled
    gc.collect()
    np.savez_compressed(out_dir / "test_data.npz", X=X_test_seq, y=y_test_seq)
    test_seq_count = len(X_test_seq)
    test_normal_count = int(np.sum(y_test_seq == 0))
    test_anomaly_count = int(np.sum(y_test_seq == 1))
    del X_test_seq, y_test_seq
    gc.collect()

    # Run Data Leakage Audit
    audit_report = audit_leakage_all(
        feature_names=canonical_names,
        train_y=y_train_raw,
        train_indices=train_idx_normal,
        val_indices=val_idx,
        test_indices=test_idx
    )

    metadata = {
        "dataset_id": dataset_id,
        "preprocessing_timestamp": datetime.now().isoformat(),
        "random_seed": seed,
        "mode": mode,
        "window_size": window_size,
        "stride_train": stride_train,
        "stride_eval": stride_eval,
        "raw_rows_total": total_raw_rows,
        "feature_count": len(canonical_names),
        "canonical_features": canonical_names,
        "discarded_columns": discarded,
        "train_rows_normal": len(train_idx_normal),
        "val_rows_total": len(val_idx),
        "test_rows_total": len(test_idx),
        "train_sequences": train_seq_count,
        "val_sequences": val_seq_count,
        "test_sequences": test_seq_count,
        "val_normal_sequences": val_normal_count,
        "val_anomaly_sequences": val_anomaly_count,
        "test_normal_sequences": test_normal_count,
        "test_anomaly_sequences": test_anomaly_count,
        "leakage_audit": audit_report,
    }

    np.savez_compressed(out_dir / "scaler.npz", medians=adapter.feature_medians, iqr=adapter.feature_iqr)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return load_processed_dataset(dataset_id, processed_base_dir=str(out_dir.parent))


def save_processed_dataset(dataset_id: str, data: Dict[str, Any], output_base_dir: str = "data/processed"):
    """Persist processed sequence tensors, scaler statistics, and metadata to disk."""
    out_dir = Path(output_base_dir) / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if "X_train" in data and "y_train" in data:
        np.savez_compressed(out_dir / "train_data.npz", X=data["X_train"], y=data["y_train"])
    if "X_val" in data and "y_val" in data:
        np.savez_compressed(out_dir / "val_data.npz", X=data["X_val"], y=data["y_val"])
    if "X_test" in data and "y_test" in data:
        np.savez_compressed(out_dir / "test_data.npz", X=data["X_test"], y=data["y_test"])
    if "scaler" in data:
        np.savez_compressed(out_dir / "scaler.npz", medians=data["scaler"]["medians"], iqr=data["scaler"]["iqr"])
    if "metadata" in data:
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(data["metadata"], f, indent=2)


def load_processed_dataset(dataset_id: str, processed_base_dir: str = "data/processed") -> Dict[str, Any]:
    """Load processed sequence tensors, scaler parameters, and metadata."""
    proc_dir = Path(processed_base_dir) / dataset_id
    if not proc_dir.exists():
        raise FileNotFoundError(f"Processed directory '{proc_dir}' does not exist.")

    train_data = np.load(proc_dir / "train_data.npz")
    val_data = np.load(proc_dir / "val_data.npz")
    test_data = np.load(proc_dir / "test_data.npz")
    scaler_data = np.load(proc_dir / "scaler.npz")

    with open(proc_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "X_train": train_data["X"],
        "y_train": train_data["y"],
        "X_val": val_data["X"],
        "y_val": val_data["y"],
        "X_test": test_data["X"],
        "y_test": test_data["y"],
        "scaler": {
            "medians": scaler_data["medians"],
            "iqr": scaler_data["iqr"]
        },
        "metadata": metadata
    }
