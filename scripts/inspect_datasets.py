"""
Dataset Discovery & Verification Script.
Inspects all 4 benchmark datasets in data/raw/ and generates comprehensive
machine-readable (JSON) and human-readable (Markdown) audit reports in research_results/dataset_audit/.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_research_config
from src.data_adapters.dataset_manager import inspect_dataset, validate_dataset


def generate_markdown_summary(summary: dict) -> str:
    """Generate human-readable markdown audit report."""
    md = []
    ds_id = summary.get("dataset_id", "UNKNOWN")
    status = summary.get("status", "UNKNOWN")

    md.append(f"# Dataset Audit Report: {ds_id}")
    md.append(f"**Verification Status:** `{status}`  ")
    md.append(f"**Files Found:** {summary.get('files_count', 0)}  ")
    md.append(f"**Total Raw Rows:** {summary.get('row_count', 0):,}  ")
    md.append(f"**Total Columns Detected:** {summary.get('column_count', 0)}  ")
    md.append(f"**Retained Canonical Features:** {summary.get('retained_feature_count', 0)}  \n")

    if not summary.get("found"):
        md.append(f"> [!WARNING]\n> {summary.get('message', 'Dataset not found.')}\n")
        return "\n".join(md)

    md.append("## File Details\n")
    md.append("| Filename | Size (MB) | Rows |")
    md.append("|---|---|---|")
    for f in summary.get("files", []):
        md.append(f"| `{f['filename']}` | {f['size_mb']} | {f['rows']:,} |")

    md.append("\n## Data Integrity & Quality\n")
    md.append(f"- **Missing Values Detected:** {summary.get('missing_values_count', 0):,}")
    md.append(f"- **Infinite Values Detected:** {summary.get('infinite_values_count', 0):,}")
    md.append(f"- **Unique Label Classes:** {summary.get('unique_classes', 0)}")

    md.append("\n## Class Distribution\n")
    md.append("| Class / Attack Label | Count | Percentage |")
    md.append("|---|---|---|")
    for lbl, stats in summary.get("class_distribution", {}).items():
        md.append(f"| `{lbl}` | {stats['count']:,} | {stats['percentage']}% |")

    md.append("\n## Feature Alignment & Discarded Columns\n")
    md.append("### Retained Canonical Features:")
    md.append("`" + "`, `".join(summary.get("retained_features", [])) + "`\n")
    md.append("### Discarded Non-Learning / ID / Leakage Columns:")
    md.append("`" + "`, `".join(summary.get("discarded_columns", [])) + "`\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Audit and inspect research datasets in data/raw/")
    parser.add_argument("--dataset", default=None, help="Specific dataset ID to inspect (e.g. CICIDS2017)")
    parser.add_argument("--config", default="configs/research.yaml", help="Path to research config")
    args = parser.parse_args()

    config = load_research_config(args.config)
    datasets = config.get("experiments", {}).get("datasets", [])
    
    audit_base = Path("research_results/dataset_audit")
    audit_base.mkdir(parents=True, exist_ok=True)

    target_ids = [args.dataset] if args.dataset else [d["id"] for d in datasets]

    print("================================================================================")
    print("                    DATASET AUDIT & VERIFICATION REPORT                         ")
    print("================================================================================")
    print(f"{'Dataset ID':<15} | {'Found':<6} | {'Status':<18} | {'Rows':<10} | {'Features':<10}")
    print("-" * 80)

    overall_summaries = []

    for ds_id in target_ids:
        summary = inspect_dataset(ds_id, config)
        overall_summaries.append(summary)

        status_code, reason = validate_dataset(ds_id, config)
        summary["validation_status"] = status_code
        summary["validation_reason"] = reason

        found_str = "YES" if summary.get("found") else "NO"
        rows_str = f"{summary.get('row_count', 0):,}"
        feat_str = str(summary.get("retained_feature_count", 0))

        print(f"{ds_id:<15} | {found_str:<6} | {status_code:<18} | {rows_str:<10} | {feat_str:<10}")

        # Save individual audit
        ds_audit_dir = audit_base / ds_id
        ds_audit_dir.mkdir(parents=True, exist_ok=True)

        with open(ds_audit_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        md_content = generate_markdown_summary(summary)
        with open(ds_audit_dir / "dataset_summary.md", "w", encoding="utf-8") as f:
            f.write(md_content)

    # Save aggregated audit summary
    with open(audit_base / "all_datasets_audit.json", "w", encoding="utf-8") as f:
        json.dump(overall_summaries, f, indent=2)

    print("================================================================================")
    print(f"Audit outputs saved to: {audit_base.resolve()}")


if __name__ == "__main__":
    main()
