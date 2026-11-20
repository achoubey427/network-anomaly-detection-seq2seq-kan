"""
Command-Line Interface for Controlled Experiment Execution.
Supports single-experiment runs, full 4x2 matrix execution (--all), and automated comparison generation.
Terminates gracefully if raw datasets are missing without generating fake data or fabricated metrics.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_research_config
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.comparator import ExperimentComparator


def main():
    parser = argparse.ArgumentParser(description="Execute controlled research experiments")
    parser.add_argument("--dataset", choices=["CICIDS2017", "CICIDS2019", "CICIDS2023", "CSECICIDS2018", "CSE-CIC-IDS2018"], help="Target Dataset ID")
    parser.add_argument("--model", choices=["baseline_seq2seq", "kan_seq2seq"], help="Model Architecture")
    parser.add_argument("--all", action="store_true", help="Execute all 8 experiments in the 4x2 research matrix")
    parser.add_argument("--config", default="configs/research.yaml", help="Path to research config")
    parser.add_argument("--compare", action="store_true", help="Generate comparison table after execution")
    args = parser.parse_args()

    if not args.all and (not args.dataset or not args.model):
        parser.error("Specify both --dataset and --model, or use --all to run the complete 4x2 matrix.")

    config = load_research_config(args.config)
    runner = ExperimentRunner(config_path=args.config)
    comparator = ExperimentComparator()

    if args.all:
        datasets = [d["id"] for d in config.get("experiments", {}).get("datasets", [])]
        models = [m["name"] for m in config.get("experiments", {}).get("models", [])]
        print("================================================================================")
        print("                  EXECUTING 4x2 RESEARCH EXPERIMENT MATRIX                      ")
        print(f" Datasets ({len(datasets)}) : {', '.join(datasets)}")
        print(f" Models ({len(models)})   : {', '.join(models)}")
        print("================================================================================")

        results = []
        for ds_id in datasets:
            for model_name in models:
                res = runner.run_experiment(dataset_id=ds_id, model_name=model_name)
                results.append(res)

        print("\n================================================================================")
        print("                          BENCHMARK RUN SUMMARY                                 ")
        print("================================================================================")
        for r in results:
            exp_id = r.get("experiment_id", "UNKNOWN")
            status = r.get("status", "UNKNOWN")
            print(f"  {exp_id:<30} : {status}")
        print("================================================================================")

        # Generate comparison table for completed real experiments
        comparator.generate_comparison_table(include_smoke_tests=False)

    else:
        res = runner.run_experiment(dataset_id=args.dataset, model_name=args.model)
        if args.compare:
            comparator.generate_comparison_table(include_smoke_tests=False)


if __name__ == "__main__":
    main()
