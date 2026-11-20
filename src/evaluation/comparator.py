"""
Controlled Comparison and Results Synthesis Engine.
Aggregates completed experiment metrics, performs statistical comparisons between
Baseline and KAN models, and generates CSV, JSON, Markdown, and LaTeX comparison tables.
"""

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class ExperimentComparator:
    """Aggregates and formats comparative benchmark results between Baseline and KAN."""

    def __init__(self, experiments_root: str = "research_results/experiments"):
        self.exp_root = Path(experiments_root)

    def load_completed_experiments(self) -> List[Dict[str, Any]]:
        """Discover and load all completed experiments."""
        if not self.exp_root.exists():
            return []

        completed = []
        for meta_file in self.exp_root.glob("**/experiment_metadata.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                if meta.get("status") != "COMPLETED":
                    continue

                exp_dir = meta_file.parent
                metrics_file = exp_dir / "metrics.json"
                threshold_file = exp_dir / "threshold.json"
                history_file = exp_dir / "training_history.json"

                if not (metrics_file.exists() and threshold_file.exists()):
                    continue

                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                with open(threshold_file, "r", encoding="utf-8") as f:
                    threshold = json.load(f)

                history = {}
                if history_file.exists():
                    with open(history_file, "r", encoding="utf-8") as f:
                        history = json.load(f)

                test_m = metrics.get("test_metrics", {})
                completed.append({
                    "dataset_id": meta.get("dataset_id"),
                    "model_name": meta.get("model_name"),
                    "total_parameters": meta.get("total_parameters"),
                    "device": meta.get("device"),
                    "duration_sec": meta.get("execution_duration_sec"),
                    "is_smoke_test": meta.get("is_smoke_test", False),
                    "timestamp": meta.get("timestamp"),
                    "threshold": threshold.get("optimal_threshold"),
                    "val_f1": threshold.get("validation_f1"),
                    "accuracy": metrics.get("accuracy", test_m.get("accuracy")),
                    "precision": metrics.get("precision", test_m.get("precision")),
                    "recall": metrics.get("recall", test_m.get("recall")),
                    "f1_score": metrics.get("f1_score", test_m.get("f1_score")),
                    "specificity": metrics.get("specificity", test_m.get("specificity")),
                    "fpr": metrics.get("fpr", test_m.get("fpr")),
                    "fnr": metrics.get("fnr", test_m.get("fnr")),
                    "roc_auc": metrics.get("roc_auc", test_m.get("roc_auc")),
                    "pr_auc": metrics.get("pr_auc", test_m.get("pr_auc")),
                    "mean_latency_ms": metrics.get("mean_latency_ms"),
                    "best_val_loss": history.get("best_val_loss"),
                    "final_train_loss": history.get("train_loss", [-1])[-1] if history.get("train_loss") else None,
                })
            except Exception as e:
                print(f"[!] Error loading experiment from {meta_file}: {e}")

        return completed

    def generate_comparison_table(self, include_smoke_tests: bool = False) -> Dict[str, Any]:
        """Generate structured comparison table and export CSV, JSON, and LaTeX."""
        all_runs = self.load_completed_experiments()
        if not include_smoke_tests:
            runs = [r for r in all_runs if not r.get("is_smoke_test")]
        else:
            runs = all_runs

        comp_dir = Path("research_results/comparisons")
        tbl_dir = Path("research_results/tables")
        comp_dir.mkdir(parents=True, exist_ok=True)
        tbl_dir.mkdir(parents=True, exist_ok=True)

        if not runs:
            # Cleanly report zero completed real experiments
            empty_report = {
                "generated_at": datetime.now().isoformat(),
                "completed_experiments_count": 0,
                "message": "No completed research experiments found on actual datasets.",
                "rows": [],
            }
            with open(comp_dir / "overall_comparison_table.json", "w", encoding="utf-8") as f:
                json.dump(empty_report, f, indent=2)
            return empty_report

        # 1. JSON Export
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "completed_experiments_count": len(runs),
            "rows": runs,
        }
        with open(comp_dir / "overall_comparison_table.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 2. CSV Export
        csv_file = comp_dir / "overall_comparison_table.csv"
        headers = [
            "dataset_id", "model_name", "total_parameters", "accuracy", "precision",
            "recall", "f1_score", "roc_auc", "pr_auc", "fpr", "threshold", "mean_latency_ms", "duration_sec"
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(runs)

        # 3. LaTeX Table Export
        tex_file = tbl_dir / "comparison_table.tex"
        with open(tex_file, "w", encoding="utf-8") as f:
            f.write("% Auto-generated Controlled Research Comparison Table\n")
            f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
            f.write("\\begin{tabular}{llrrrrrrrr}\n\\hline\n")
            f.write("Dataset & Model & Params & Accuracy & Precision & Recall & F1 & ROC-AUC & PR-AUC & Latency (ms) \\\\\n\\hline\n")
            for r in runs:
                f.write(
                    f"{r['dataset_id']} & {r['model_name']} & {r['total_parameters']:,} & "
                    f"{r['accuracy']:.4f} & {r['precision']:.4f} & {r['recall']:.4f} & "
                    f"{r['f1_score']:.4f} & {r['roc_auc']:.4f} & {r['pr_auc']:.4f} & {r['mean_latency_ms']:.2f} \\\\\n"
                )
            f.write("\\hline\n\\end{tabular}\n")
            f.write("\\caption{Controlled Comparison: Baseline Seq2Seq vs. Proposed KAN Autoencoder across Intrusion Datasets}\n")
            f.write("\\label{tab:research_comparison}\n\\end{table*}\n")

        return report_data
