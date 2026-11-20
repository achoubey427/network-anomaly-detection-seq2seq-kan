"""
SQLite storage manager for alerts, packet telemetry, and experiment tracking.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time


class StorageManager:
    """Manages SQLite database schema and persistence for ICS anomaly detection."""

    def __init__(self, db_path: str = "data/alerts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    flow_key TEXT,
                    anomaly_score REAL,
                    threshold REAL,
                    severity TEXT,
                    explanation TEXT,
                    payload_json TEXT
                )
            """)
            # Traffic log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traffic_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    flow_key TEXT,
                    function_code INTEGER,
                    register_address REAL,
                    register_value REAL,
                    is_anomaly INTEGER
                )
            """)
            conn.commit()

    def insert_alert(self, alert: Dict[str, Any]) -> int:
        """Insert a detected anomaly alert."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (timestamp, flow_key, anomaly_score, threshold, severity, explanation, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.get("timestamp", time.time()),
                alert.get("flow_key", "unknown"),
                float(alert.get("anomaly_score", 0.0)),
                float(alert.get("threshold", 0.0)),
                alert.get("severity", "HIGH"),
                alert.get("explanation", "Reconstruction error threshold exceeded"),
                json.dumps(alert.get("payload", {}))
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve latest alerts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
