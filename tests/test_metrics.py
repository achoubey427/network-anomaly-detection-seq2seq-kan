"""
Unit tests for evaluation metrics calculations.
"""

import unittest
import numpy as np
from src.utils.metrics import (
    compute_confusion_matrix,
    compute_classification_metrics,
    compute_roc_auc,
    compute_pr_auc,
    optimize_threshold_f1,
    compute_all_metrics
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        self.y_scores = np.array([0.1, 0.2, 0.15, 0.6, 0.4, 0.8, 0.85, 0.9])

    def test_confusion_matrix(self):
        cm = compute_confusion_matrix(self.y_true, self.y_pred)
        self.assertEqual(cm["tp"], 3)
        self.assertEqual(cm["tn"], 3)
        self.assertEqual(cm["fp"], 1)
        self.assertEqual(cm["fn"], 1)

    def test_classification_metrics(self):
        metrics = compute_classification_metrics(self.y_true, self.y_pred)
        self.assertAlmostEqual(metrics["accuracy"], 6 / 8)
        self.assertAlmostEqual(metrics["precision"], 3 / 4)
        self.assertAlmostEqual(metrics["recall"], 3 / 4)
        self.assertAlmostEqual(metrics["f1_score"], 0.75)

    def test_roc_auc(self):
        auc = compute_roc_auc(self.y_true, self.y_scores)
        self.assertGreaterEqual(auc, 0.5)
        self.assertLessEqual(auc, 1.0)

    def test_pr_auc(self):
        pr_auc = compute_pr_auc(self.y_true, self.y_scores)
        self.assertGreaterEqual(pr_auc, 0.0)
        self.assertLessEqual(pr_auc, 1.0)

    def test_optimize_threshold_f1(self):
        best_thresh, best_f1 = optimize_threshold_f1(self.y_true, self.y_scores, n_steps=100)
        self.assertGreater(best_f1, 0.0)
        self.assertGreaterEqual(best_thresh, float(np.min(self.y_scores)))


if __name__ == "__main__":
    unittest.main()
