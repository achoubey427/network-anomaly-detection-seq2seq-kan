"""
Dataset Adapter for CSE-CIC-IDS2018 (Communications Security Establishment & CIC).
Handles 80+ CICFlowMeter network-flow attributes, multi-day enterprise topologies,
and diverse intrusion scenarios (Brute Force, DoS, DDoS, Web Attacks, Infiltration, Botnet).
"""

from pathlib import Path
from typing import Any, Dict, List
import csv

from src.data_adapters.base_adapter import BaseDatasetAdapter


class CSECICIDS2018Adapter(BaseDatasetAdapter):
    """Adapter for CSE-CIC-IDS2018 benchmark dataset."""

    def detect_schema(self, sample_file: Path) -> Dict[str, Any]:
        """Detect headers, strip whitespaces, and identify label column."""
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
