"""
Unit tests for Dataset Adapters.
"""

import unittest
import numpy as np
from src.utils.config_loader import load_research_config
from src.data_adapters import get_adapter


class TestAdapters(unittest.TestCase):
    def setUp(self):
        self.config = load_research_config("configs/research.yaml")

    def test_instantiate_all_adapters(self):
        for ds_cfg in self.config["experiments"]["datasets"]:
            adapter = get_adapter(ds_cfg, self.config)
            self.assertIsNotNone(adapter)
            self.assertEqual(adapter.dataset_id, ds_cfg["id"])

    def test_adapter_cleaning_and_binarization(self):
        adapter = get_adapter(self.config["experiments"]["datasets"][0], self.config)
        # Test label binarization
        raw_labels = ["BENIGN", "Benign", "DoS Hulk", "PortScan", "0", "DDoS"]
        binary = adapter.binarize_labels(raw_labels)
        np.testing.assert_array_equal(binary, np.array([0, 0, 1, 1, 0, 1]))

        # Test infinite/missing values cleaning
        raw_arr = np.array([[1.0, np.inf], [np.nan, 4.0], [3.0, 5.0]])
        cleaned = adapter.clean_numerical_array(raw_arr)
        self.assertFalse(np.any(np.isnan(cleaned)))
        self.assertFalse(np.any(np.isinf(cleaned)))

    def test_sequence_generation(self):
        adapter = get_adapter(self.config["experiments"]["datasets"][0], self.config)
        features = np.random.randn(100, 10).astype(np.float32)
        labels = np.zeros(100, dtype=int)
        labels[35] = 1 # One anomaly at index 35

        seqs, seq_lbls = adapter.create_sequences(features, labels, window_size=30, stride=1)
        self.assertEqual(len(seqs), 71)
        # Windows covering index 35 should have label 1
        self.assertEqual(seq_lbls[0], 0)
        self.assertEqual(seq_lbls[10], 1) # Window 10 covers 10..39 including 35


if __name__ == "__main__":
    unittest.main()
