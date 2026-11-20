"""
Dataset Adapters Package.
Provides access to dataset adapters for all 4 benchmark datasets:
1. CIC-IDS2017
2. CIC-DDoS2019
3. CICIoT2023
4. CSE-CIC-IDS2018
"""

from typing import Any, Dict
from src.data_adapters.base_adapter import BaseDatasetAdapter
from src.data_adapters.cicids2017_adapter import CICIDS2017Adapter
from src.data_adapters.cicids2019_adapter import CICIDS2019Adapter
from src.data_adapters.cicids2023_adapter import CICIDS2023Adapter
from src.data_adapters.csecicids2018_adapter import CSECICIDS2018Adapter


ADAPTER_REGISTRY = {
    "cicids2017_adapter": CICIDS2017Adapter,
    "cicids2019_adapter": CICIDS2019Adapter,
    "cicids2023_adapter": CICIDS2023Adapter,
    "csecicids2018_adapter": CSECICIDS2018Adapter,
    # Alias / backward compatibility
    "cicids2018_adapter": CSECICIDS2018Adapter,
}


def get_adapter(dataset_config: Dict[str, Any], global_config: Dict[str, Any]) -> BaseDatasetAdapter:
    """Factory function to instantiate the correct dataset adapter."""
    adapter_name = dataset_config.get("adapter", "cicids2017_adapter")
    if adapter_name not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter '{adapter_name}'. Registered adapters: {list(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[adapter_name](dataset_config, global_config)
