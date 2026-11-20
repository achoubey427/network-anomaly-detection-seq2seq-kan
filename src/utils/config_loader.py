"""
Configuration loader and validator for ICS Anomaly Detection.
Supports both default real-time pipeline config and research benchmark config.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def load_research_config(config_path: str = "configs/research.yaml") -> Dict[str, Any]:
    """Load and validate the research benchmark configuration."""
    config = load_yaml(config_path)
    
    # Validation checks
    required_sections = ["experiments", "preprocessing", "model", "kan", "training", "thresholding", "evaluation", "output"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in research config: '{section}'")
            
    # Verify dataset entries
    datasets = config.get("experiments", {}).get("datasets", [])
    if len(datasets) != 4:
        raise ValueError(f"Expected 4 datasets in research config, found {len(datasets)}")
        
    # Verify model entries
    models = config.get("experiments", {}).get("models", [])
    if len(models) != 2:
        raise ValueError(f"Expected 2 models in research config, found {len(models)}")
        
    return config


def load_default_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load and validate the default real-time pipeline configuration."""
    config = load_yaml(config_path)
    required_sections = ["pipeline", "features", "model", "scoring", "server"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in default config: '{section}'")
    return config
