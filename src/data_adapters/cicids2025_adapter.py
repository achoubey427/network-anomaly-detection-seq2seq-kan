"""
Dataset Adapter for CICIDS2025 / CIC-DGG-2025 (Faculty 2025 Target Dataset).
Features dynamic schema detection and automatic numerical/categorical column classification.
Does NOT fabricate data if no files exist.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import csv

from src.data_adapters.base_adapter import BaseDatasetAdapter


class CICIDS2025Adapter(BaseDatasetAdapter):
    """Adapter for CICIDS2025 / CIC-DGG-2025 benchmark dataset."""

    def detect_schema(self, sample_file: Path) -> Dict[str, Any]:
        """Dynamically inspect column types and detect candidate label columns."""
        with open(sample_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            headers = [h.strip() for h in next(reader, [])]
            first_row = [v.strip() for v in next(reader, [])]

        numerical_candidates = []
        categorical_candidates = []
        label_col = None

        for col, val in zip(headers, first_row):
            if col.lower() in ["label", "class", "attack", "is_attack", "anomaly"]:
                label_col = col
                continue
            try:
                float(val)
                numerical_candidates.append(col)
            except ValueError:
                categorical_candidates.append(col)

        return {
            "total_columns": len(headers),
            "columns": headers,
            "detected_label_column": label_col or "Label",
            "detected_numerical": numerical_candidates,
            "detected_categorical": categorical_candidates
        }
