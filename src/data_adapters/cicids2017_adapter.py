"""
Dataset Adapter for ISCX CICIDS2017.
Handles whitespace-padded headers, flow-based network attributes, and multi-class intrusion labels.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import csv

from src.data_adapters.base_adapter import BaseDatasetAdapter


class CICIDS2017Adapter(BaseDatasetAdapter):
    """Adapter for CICIDS2017 benchmark dataset."""

    def detect_schema(self, sample_file: Path) -> Dict[str, Any]:
        """Detect headers, strip whitespaces, and identify potential label column."""
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
