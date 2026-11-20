"""
Development Smoke Test for Experiment Runner and Comparator Engine.
Verifies end-to-end execution of both Baseline and KAN models on small in-memory tensors,
validation threshold search, frozen test evaluation, and comparative table synthesis.
DOES NOT USE OR CLAIM RESEARCH BENCHMARK EVIDENCE.
"""

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_research_config
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.comparator import ExperimentComparator


def run_runner_smoke_test():
    print("================================================================================")
    print("           DEVELOPMENT SMOKE TEST: EXPERIMENT RUNNER & COMPARATOR               ")
    print("        (Software Verification Only - Not Research Benchmark Evidence)          ")
    print("================================================================================")

    config = load_research_config("configs/research.yaml")
    
    # Create small mock partitions
    seq_len = 30
    feature_dim = 16
    np.random.seed(42)

    # Train: 40 benign sequences (y=0)
    X_train = np.random.randn(40, seq_len, feature_dim).astype(np.float32)
    y_train = np.zeros(40, dtype=int)

    # Val: 20 mixed sequences (15 normal, 5 anomalous)
    X_val = np.random.randn(20, seq_len, feature_dim).astype(np.float32)
    y_val = np.array([0] * 15 + [1] * 5, dtype=int)
    X_val[15:] += 2.0  # Inject synthetic anomaly offset in val anomalies

    # Test: 20 mixed sequences (15 normal, 5 anomalous)
    X_test = np.random.randn(20, seq_len, feature_dim).astype(np.float32)
    y_test = np.array([0] * 15 + [1] * 5, dtype=int)
    X_test[15:] += 2.0  # Inject synthetic anomaly offset in test anomalies

    mock_data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    runner = ExperimentRunner(config_path="configs/research.yaml")
    # Small epochs for smoke test
    runner.config["training"]["epochs"] = 3
    runner.config["training"]["batch_size"] = 16

    # 1. Run Baseline Smoke Test
    print("\n[1/2] Executing Baseline Model Smoke Test...")
    res_baseline = runner.run_experiment(
        dataset_id="smoke_test",
        model_name="baseline_seq2seq",
        custom_data=mock_data,
        is_smoke_test=True,
    )

    # 2. Run KAN Smoke Test
    print("\n[2/2] Executing Proposed KAN Model Smoke Test...")
    res_kan = runner.run_experiment(
        dataset_id="smoke_test",
        model_name="kan_seq2seq",
        custom_data=mock_data,
        is_smoke_test=True,
    )

    # 3. Test Comparator
    print("\n[*] Generating comparative analysis from smoke test runs...")
    comparator = ExperimentComparator()
    comp_report = comparator.generate_comparison_table(include_smoke_tests=True)

    print(f"[+] Comparator synthesized {comp_report['completed_experiments_count']} runs:")
    for row in comp_report["rows"]:
        print(f"    - {row['model_name']:<20}: Params = {row['total_parameters']:,} | F1 = {row['f1_score']:.4f} | ROC-AUC = {row['roc_auc']:.4f}")

    print("\n================================================================================")
    print("                 SMOKE TEST COMPLETED SUCCESSFULLY                              ")
    print("================================================================================")


if __name__ == "__main__":
    run_runner_smoke_test()
