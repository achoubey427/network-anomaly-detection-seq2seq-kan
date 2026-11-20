"""
Dataset Adapter for CICIoT2023 / Edge IoT SCADA Benchmark.
Handles topological IoT network attributes, high-dimensional numerical counters, and multi-class device threats.
"""

from pathlib import Path
from typing import Any, Dict, List
import csv

from src.data_adapters.base_adapter import BaseDatasetAdapter


class CICIDS2023Adapter(BaseDatasetAdapter):
    """Adapter for CICIDS2023 / CICIoT2023 benchmark dataset."""

    def detect_schema(self, sample_file: Path) -> Dict[str, Any]:
        """Detect columns and handle lowercase label headers."""
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
