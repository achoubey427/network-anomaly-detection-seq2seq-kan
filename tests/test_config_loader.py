"""
Unit tests for configuration loader and validator.
"""

import unittest
from pathlib import Path
from src.utils.config_loader import load_default_config, load_research_config


class TestConfigLoader(unittest.TestCase):
    def test_load_research_config(self):
        config = load_research_config("configs/research.yaml")
        self.assertIn("experiments", config)
        self.assertEqual(len(config["experiments"]["datasets"]), 4)
        self.assertEqual(len(config["experiments"]["models"]), 2)
        self.assertEqual(config["reproducibility"]["seed"], 42)

    def test_load_default_config(self):
        config = load_default_config("configs/default.yaml")
        self.assertIn("pipeline", config)
        self.assertIn("features", config)
        self.assertEqual(config["pipeline"]["packet_window_size"], 30)


if __name__ == "__main__":
    unittest.main()
