"""
Phase 2 Preprocessing & Ingestion Unit Tests.
Tests dataset discovery, canonical feature alignment, label normalization,
missing/inf handling, data leakage protection, partition isolation,
sequence generation, and metadata persistence.
"""

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

from src.utils.config_loader import load_research_config
from src.data_adapters.canonical_features import map_raw_columns, sanitize_column_name, DISCARD_COLUMNS
from src.data_adapters.leakage_detector import (
    verify_feature_matrix_leakage,
    verify_scaler_fit_isolation,
    verify_partition_disjointness,
    LeakageViolationError,
)
from src.data_adapters.dataset_manager import (
    inspect_dataset,
    validate_dataset,
    prepare_dataset,
    save_processed_dataset,
    load_processed_dataset,
)


class TestPhase2Preprocessing(unittest.TestCase):
    def setUp(self):
        self.config = load_research_config("configs/research.yaml")
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_canonical_feature_mapping(self):
        raw_cols_2017 = [" Flow Duration", " Total Fwd Packets", " Destination Port", " Flow ID", " Timestamp", " Label"]
        retained_raw, canonical_names, discarded = map_raw_columns(raw_cols_2017, "CICIDS2017")
        
        self.assertIn("flow_duration", canonical_names)
        self.assertIn("total_fwd_packets", canonical_names)
        self.assertIn(" Flow ID", discarded)
        self.assertIn(" Timestamp", discarded)
        self.assertIn(" Label", discarded)

    def test_leakage_detector_feature_violation(self):
        # Should raise LeakageViolationError if label or target is in feature matrix
        with self.assertRaises(LeakageViolationError):
            verify_feature_matrix_leakage(["flow_duration", "packet_length", "is_anomaly"])

        with self.assertRaises(LeakageViolationError):
            verify_feature_matrix_leakage(["flow_duration", "label"])

        # Clean list should pass
        try:
            verify_feature_matrix_leakage(["flow_duration", "total_fwd_packets", "flow_iat_mean"])
        except LeakageViolationError:
            self.fail("Clean feature list raised LeakageViolationError unexpectedly.")

    def test_scaler_isolation_leakage(self):
        # Should fail if scaler is fitted on array containing y == 1
        dirty_y = np.array([0, 0, 1, 0])
        with self.assertRaises(LeakageViolationError):
            verify_scaler_fit_isolation(dirty_y)

        # Pure normal training partition should pass
        clean_y = np.zeros(100, dtype=int)
        try:
            verify_scaler_fit_isolation(clean_y)
        except LeakageViolationError:
            self.fail("Clean normal training labels raised LeakageViolationError unexpectedly.")

    def test_partition_disjointness(self):
        train_idx = np.array([0, 1, 2, 3])
        val_idx = np.array([4, 5])
        test_idx = np.array([6, 7])
        verify_partition_disjointness(train_idx, val_idx, test_idx)

        # Overlapping partitions should raise error
        overlapping_val = np.array([3, 4, 5])
        with self.assertRaises(LeakageViolationError):
            verify_partition_disjointness(train_idx, overlapping_val, test_idx)

    def test_missing_and_incompatible_dataset_handling(self):
        # Missing dataset verification
        status, reason = validate_dataset("CICIDS2023", self.config)
        self.assertEqual(status, "MISSING")
        self.assertIn("Dataset not found", reason)

        # Incompatible dataset verification
        bad_status, bad_reason = validate_dataset("NON_EXISTENT_DATASET", self.config)
        self.assertEqual(bad_status, "INCOMPATIBLE")

    def test_end_to_end_pipeline_with_mock_data(self):
        # Create a mock raw dataset folder
        mock_raw_dir = Path(self.test_dir) / "data/raw/CICIDS2017"
        mock_raw_dir.mkdir(parents=True, exist_ok=True)
        mock_file = mock_raw_dir / "sample_traffic.csv"

        headers = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Flow ID", "Timestamp", "Label"]
        rows = [
            [100.0, 5.0, 4.0, "192.168.1.1-10.0.0.1", "2026-10-04 12:00:00", "BENIGN"],
            [120.0, 6.0, 5.0, "192.168.1.1-10.0.0.1", "2026-10-04 12:00:01", "BENIGN"],
            [150.0, np.inf, 4.0, "192.168.1.1-10.0.0.1", "2026-10-04 12:00:02", "BENIGN"],
            [200.0, 10.0, 8.0, "192.168.1.1-10.0.0.1", "2026-10-04 12:00:03", "DDoS"],
        ] * 30  # 120 rows

        with open(mock_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        # Update test config with mock raw dir and isolated test processed dir
        import copy
        custom_config = copy.deepcopy(self.config)
        custom_config["experiments"]["datasets"][0]["raw_dir"] = str(mock_raw_dir)
        custom_config["experiments"]["datasets"][0]["processed_dir"] = str(Path(self.test_dir) / "data/processed/CICIDS2017")

        # Test Inspection
        summary = inspect_dataset("CICIDS2017", custom_config)
        self.assertTrue(summary["found"])
        self.assertEqual(summary["row_count"], 120)
        self.assertIn("BENIGN", summary["class_distribution"])
        self.assertIn("DDoS", summary["class_distribution"])

        # Test Preparation
        processed = prepare_dataset("CICIDS2017", custom_config, mode="research")
        self.assertIn("X_train", processed)
        self.assertIn("X_val", processed)
        self.assertIn("X_test", processed)
        self.assertEqual(processed["X_train"].ndim, 3) # (batch, window_size, features)
        self.assertEqual(processed["X_train"].shape[1], 30) # Window size 30

        # Verify training partition has only benign sequences
        self.assertTrue(np.all(processed["y_train"] == 0))

        # Test Saving and Loading
        proc_out_dir = Path(self.test_dir) / "data/processed"
        save_processed_dataset("CICIDS2017", processed, output_base_dir=str(proc_out_dir))

        loaded = load_processed_dataset("CICIDS2017", processed_base_dir=str(proc_out_dir))
        self.assertEqual(loaded["X_train"].shape, processed["X_train"].shape)
        self.assertEqual(loaded["metadata"]["dataset_id"], "CICIDS2017")


if __name__ == "__main__":
    unittest.main()
