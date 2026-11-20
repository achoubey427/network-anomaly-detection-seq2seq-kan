"""
Dataset Adapter for CSE-CIC-IDS2018 / CICDDoS2019.
Handles modern DDoS, reflection attack signatures, and timestamp formatting.
"""

from pathlib import Path
from typing import Any, Dict, List
import csv

from src.data_adapters.base_adapter import BaseDatasetAdapter


class CICIDS2019Adapter(BaseDatasetAdapter):
    """Adapter for CICIDS2019 / CSE-CIC-IDS2018 benchmark dataset."""

    def detect_schema(self, sample_file: Path) -> Dict[str, Any]:
        """Detect columns and handle variations in column names."""
        with open(sample_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            headers = [h.strip() for h in next(reader, [])]
            
        label_col = None
        for col in headers:
            if col.lower() in ["label", "class", "attack"]:
                label_col = col
                break
                
        return {
            "total_columns": len(headers),
            "columns": headers,
            "detected_label_column": label_col or self.label_column
        }
