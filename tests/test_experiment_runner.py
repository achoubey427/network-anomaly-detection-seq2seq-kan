"""
Phase 5 Unit Tests for Controlled Experiment Runner and Evaluation Framework.
Tests experiment configuration, model selection, missing-dataset handling, deterministic seeds,
validation threshold optimization, test-set isolation, metric calculations, artifact persistence,
status lifecycles, and comparative table synthesis.
"""

from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import torch

from src.utils.config_loader import load_research_config
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.comparator import ExperimentComparator


class TestExperimentRunner(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = "configs/research.yaml"
        self.runner = ExperimentRunner(config_path=self.config_path)
        # Small training epochs for unit tests
        self.runner.config["training"]["epochs"] = 2
        self.runner.config["training"]["batch_size"] = 8

        # Create mock data partitions
        seq_len = 30
        feat_dim = 16
        np.random.seed(42)

        self.mock_data = {
            "X_train": np.random.randn(20, seq_len, feat_dim).astype(np.float32),
            "y_train": np.zeros(20, dtype=int), # Benign only
            "X_val": np.random.randn(10, seq_len, feat_dim).astype(np.float32),
            "y_val": np.array([0] * 7 + [1] * 3, dtype=int),
            "X_test": np.random.randn(10, seq_len, feat_dim).astype(np.float32),
            "y_test": np.array([0] * 7 + [1] * 3, dtype=int),
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_missing_dataset_handling(self):
        # When dataset files are missing, runner should return DATASET_MISSING without crashing or fabricating data
        result = self.runner.run_experiment(dataset_id="CICIDS2023", model_name="baseline_seq2seq")
        self.assertEqual(result["status"], "DATASET_MISSING")
        self.assertNotIn("metrics", result) # Must not produce fake metrics

    def test_deterministic_seed(self):
        self.assertEqual(self.runner.seed, 42)

    def test_end_to_end_baseline_experiment_execution(self):
        result = self.runner.run_experiment(
            dataset_id="unit_test_ds",
            model_name="baseline_seq2seq",
            custom_data=self.mock_data,
            is_smoke_test=True,
        )
        self.assertEqual(result["metadata"]["status"], "COMPLETED")
        self.assertIn("metrics", result)
        self.assertIn("threshold", result)
        self.assertIn("accuracy", result["metrics"])
        self.assertIn("f1_score", result["metrics"])
        self.assertIn("roc_auc", result["metrics"])
        self.assertIn("pr_auc", result["metrics"])
        self.assertIn("mean_latency_ms", result["metrics"])

        # Verify artifacts written to disk
        exp_dir = Path(result["artifacts_dir"])
        self.assertTrue((exp_dir / "config.json").exists())
        self.assertTrue((exp_dir / "training_history.json").exists())
        self.assertTrue((exp_dir / "metrics.json").exists())
        self.assertTrue((exp_dir / "threshold.json").exists())
        self.assertTrue((exp_dir / "predictions.npz").exists())
        self.assertTrue((exp_dir / "experiment_metadata.json").exists())

    def test_end_to_end_kan_experiment_execution(self):
        result = self.runner.run_experiment(
            dataset_id="unit_test_ds",
            model_name="kan_seq2seq",
            custom_data=self.mock_data,
            is_smoke_test=True,
        )
        self.assertEqual(result["metadata"]["status"], "COMPLETED")
        self.assertIn("metrics", result)
        self.assertIn("optimal_threshold", result["threshold"])
        self.assertGreater(result["metadata"]["total_parameters"], 0)

    def test_threshold_derived_from_validation_only(self):
        result = self.runner.run_experiment(
            dataset_id="unit_test_ds",
            model_name="baseline_seq2seq",
            custom_data=self.mock_data,
            is_smoke_test=True,
        )
        thresh_info = result["threshold"]
        # Threshold metadata must record validation search metrics
        self.assertEqual(thresh_info["search_method"], "f1_optimal")
        self.assertIn("validation_f1", thresh_info)
        self.assertEqual(thresh_info["validation_samples"], len(self.mock_data["y_val"]))

    def test_predictions_npz_contents(self):
        result = self.runner.run_experiment(
            dataset_id="unit_test_ds",
            model_name="baseline_seq2seq",
            custom_data=self.mock_data,
            is_smoke_test=True,
        )
        pred_file = Path(result["artifacts_dir"]) / "predictions.npz"
        loaded = np.load(pred_file)
        self.assertIn("y_true", loaded)
        self.assertIn("y_pred", loaded)
        self.assertIn("anomaly_scores", loaded)
        self.assertEqual(len(loaded["y_true"]), len(self.mock_data["y_test"]))
        self.assertEqual(len(loaded["y_pred"]), len(self.mock_data["y_test"]))

    def test_comparator_generation(self):
        # Run baseline and kan experiments
        self.runner.run_experiment("unit_test_comp", "baseline_seq2seq", custom_data=self.mock_data, is_smoke_test=True)
        self.runner.run_experiment("unit_test_comp", "kan_seq2seq", custom_data=self.mock_data, is_smoke_test=True)

        comparator = ExperimentComparator()
        report = comparator.generate_comparison_table(include_smoke_tests=True)
        self.assertGreaterEqual(report["completed_experiments_count"], 2)

        # Check export files exist
        self.assertTrue(Path("research_results/comparisons/overall_comparison_table.json").exists())
        self.assertTrue(Path("research_results/comparisons/overall_comparison_table.csv").exists())
        self.assertTrue(Path("research_results/tables/comparison_table.tex").exists())


if __name__ == "__main__":
    unittest.main()
