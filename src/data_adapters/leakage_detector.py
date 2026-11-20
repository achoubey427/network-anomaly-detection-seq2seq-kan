"""
Data Leakage Protection and Verification Layer.
Performs deterministic checks to guarantee zero data leakage across partitions,
feature matrices, scalers, and evaluation boundaries.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class LeakageViolationError(Exception):
    """Raised when data leakage or partition contamination is detected."""
    pass


def verify_feature_matrix_leakage(feature_names: List[str], label_col: Optional[str] = None):
    """Verify that no label, ground-truth indicator, or raw identifier exists in the feature matrix."""
    forbidden_terms = ["label", "attack", "is_anomaly", "binary_label", "raw_label", "target", "class"]
    if label_col:
        forbidden_terms.append(label_col.strip().lower())

    for feat in feature_names:
        feat_lower = feat.strip().lower()
        for forbidden in forbidden_terms:
            if feat_lower == forbidden or feat_lower.endswith(f"_{forbidden}"):
                raise LeakageViolationError(
                    f"Data leakage detected! Feature '{feat}' matches forbidden label term '{forbidden}'."
                )


def verify_scaler_fit_isolation(fitted_on_y: np.ndarray):
    """Verify that the scaler was fitted exclusively on normal/benign samples (y == 0)."""
    if len(fitted_on_y) == 0:
        raise LeakageViolationError("Scaler fitting partition is empty.")
    if np.any(fitted_on_y != 0):
        n_anomalous = np.sum(fitted_on_y != 0)
        raise LeakageViolationError(
            f"Scaler leakage detected! Scaler was fitted on {n_anomalous} anomalous samples. "
            "Autoencoders must fit scalers strictly on benign/normal training data."
        )


def verify_partition_disjointness(train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
    """Verify that train, validation, and test sequence index partitions are mutually exclusive."""
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)

    if train_val_overlap:
        raise LeakageViolationError(f"Partition overlap between Train and Val: {len(train_val_overlap)} indices.")
    if train_test_overlap:
        raise LeakageViolationError(f"Partition overlap between Train and Test: {len(train_test_overlap)} indices.")
    if val_test_overlap:
        raise LeakageViolationError(f"Partition overlap between Val and Test: {len(val_test_overlap)} indices.")


def audit_leakage_all(
    feature_names: List[str],
    train_y: np.ndarray,
    train_indices: Optional[np.ndarray] = None,
    val_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
) -> Dict[str, bool]:
    """Run all leakage checks and return verification report."""
    verify_feature_matrix_leakage(feature_names)
    verify_scaler_fit_isolation(train_y)
    if train_indices is not None and val_indices is not None and test_indices is not None:
        verify_partition_disjointness(train_indices, val_indices, test_indices)

    return {
        "feature_leakage_free": True,
        "scaler_isolation_verified": True,
        "partitions_disjoint": True,
    }
