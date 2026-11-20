"""
Dataset Preparation CLI.
Executes cleaning, label normalization, canonical feature alignment, train/val/test split,
scaler isolation, sequence generation, leakage audit, and NPZ persistence.
Supports --dataset <NAME>, --all, and --mode <research|development>.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_research_config
from src.data_adapters.dataset_manager import (
    validate_dataset,
    prepare_dataset,
    save_processed_dataset
)


def process_single_dataset(dataset_id: str, config: dict, mode: str, sample_size: int = None):
    print(f"\n>>> Checking dataset: {dataset_id}")
    status, reason = validate_dataset(dataset_id, config)
    
    if status == "MISSING":
        print(f"[-] Status: {status} ({reason})")
        print(f"[-] Skipped without fabrication.")
        return False
    elif status == "INCOMPATIBLE":
        print(f"[-] Status: {status} ({reason})")
        print(f"[-] Incompatible format. Skipped.")
        return False

    print(f"[+] Status: {status} ({reason})")
    print(f"[+] Running preparation pipeline (mode={mode})...")
    
    try:
        processed_data = prepare_dataset(dataset_id, config, mode=mode, sample_size=sample_size)
        save_processed_dataset(dataset_id, processed_data)
        meta = processed_data["metadata"]
        
        print(f"[+] Successfully processed {dataset_id}:")
        print(f"    - Raw Rows: {meta['raw_rows_total']:,}")
        print(f"    - Canonical Features: {meta['feature_count']}")
        print(f"    - Normal Train Sequences: {meta['train_sequences']:,}")
        print(f"    - Val Sequences: {meta['val_sequences']:,} (Normal: {meta['val_normal_sequences']}, Anomaly: {meta['val_anomaly_sequences']})")
        print(f"    - Test Sequences: {meta['test_sequences']:,} (Normal: {meta['test_normal_sequences']}, Anomaly: {meta['test_anomaly_sequences']})")
        print(f"    - Output: data/processed/{dataset_id}/")
        return True
    except Exception as e:
        print(f"[!] Error preparing dataset {dataset_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepare and preprocess research datasets")
    parser.add_argument("--dataset", default=None, choices=["CICIDS2017", "CICIDS2019", "CICIDS2023", "CICIDS2025", "CSECICIDS2018", "CSE-CIC-IDS2018"], help="Dataset ID")
    parser.add_argument("--all", action="store_true", help="Process all configured datasets")
    parser.add_argument("--mode", default="research", choices=["research", "development"], help="Processing mode")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample limit in development mode")
    parser.add_argument("--config", default="configs/research.yaml", help="Path to research config")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Specify either --dataset <NAME> or --all")

    config = load_research_config(args.config)
    datasets = config.get("experiments", {}).get("datasets", [])
    target_ids = [d["id"] for d in datasets] if args.all else [args.dataset]

    print("================================================================================")
    print("                     RESEARCH DATASET PREPARATION PIPELINE                      ")
    print(f" Mode: {args.mode.upper()} | Config: {args.config}")
    print("================================================================================")

    results = {}
    for ds_id in target_ids:
        success = process_single_dataset(ds_id, config, mode=args.mode, sample_size=args.sample_size)
        results[ds_id] = "PROCESSED" if success else "SKIPPED/MISSING"

    print("\n================================================================================")
    print("                              FINAL SUMMARY                                     ")
    print("================================================================================")
    for ds_id, res in results.items():
        print(f"  {ds_id:<15}: {res}")
    print("================================================================================")


if __name__ == "__main__":
    main()
