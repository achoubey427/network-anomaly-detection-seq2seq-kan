"""
Unified Production HTTP Server & Live Inference Engine.
Title: Network Anomaly Detection: Seq2Seq LSTM + Attention + KAN
Uses Python standard library ThreadingHTTPServer for zero-dependency, robust serving
of real-time PyTorch inference and the compiled React production frontend.
"""

import csv
import io
import json
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.parse
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.api.inference_engine import AnomalyInferenceEngine
from src.evaluation.comparator import ExperimentComparator
from src.utils.config_loader import load_research_config

# Global inference engine instance
inference_engine = AnomalyInferenceEngine()
comparator = ExperimentComparator()
config = load_research_config()


class AnomalyDetectionHTTPHandler(SimpleHTTPRequestHandler):
    """Handles REST API calls and serves the production React SPA bundle."""

    def __init__(self, *args, **kwargs):
        self.dist_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
        super().__init__(*args, directory=str(self.dist_dir) if self.dist_dir.exists() else None, **kwargs)

    def _send_json(self, data: Any, status_code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/v1/health":
            self._send_json({
                "status": "healthy",
                "timestamp": time.time(),
                "system": "Network Anomaly Detection (Seq2Seq + KAN)",
                "research_phase": "FROZEN_AND_VERIFIED",
                "models_cached": list(inference_engine.model_cache.keys())
            })
            return

        elif path == "/api/v1/sample-bank":
            try:
                bank = inference_engine.get_sample_bank()
                self._send_json({"status": "success", "sample_bank": bank})
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, status_code=500)
            return

        elif path == "/api/v1/experiment-results":
            try:
                table_data = comparator.generate_comparison_table()
                self._send_json(table_data)
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, status_code=500)
            return

        elif path == "/api/v1/datasets-info":
            self._send_json({
                "datasets": [
                    {
                        "id": "CSECICIDS2018",
                        "name": "CSE-CIC-IDS2018",
                        "official_name": "Communications Security Establishment & CIC IDS 2018",
                        "provenance": "Official UNB AWS Dataset Repository",
                        "status": "COMPLETED & FROZEN",
                        "train_sequences": 33355,
                        "test_sequences": 49641,
                        "canonical_features": 78,
                        "attack_categories": ["Infiltration", "Brute Force", "DoS", "DDoS", "Botnet"],
                        "description": "Comprehensive enterprise network dataset spanning multi-day attack simulations across realistic topology."
                    },
                    {
                        "id": "CICIDS2017",
                        "name": "CIC-IDS2017",
                        "official_name": "Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017",
                        "provenance": "Official UNB MachineLearningCSV Repository",
                        "status": "COMPLETED",
                        "train_sequences": 318258,
                        "test_sequences": 424583,
                        "canonical_features": 78,
                        "attack_categories": ["PortScan", "DDoS", "DoS GoldenEye", "DoS Hulk", "Web Attacks", "Infiltration"],
                        "description": "Standard modern IDS benchmark with 2.83M network flows captured across diverse protocol stacks."
                    },
                    {
                        "id": "CICIDS2019",
                        "name": "CIC-DDoS2019",
                        "official_name": "CIC Distributed Denial of Service Dataset 2019",
                        "provenance": "Official UNB CSV-01-12 & CSV-03-11 Archives",
                        "status": "COMPLETED",
                        "train_sequences": 3838,
                        "test_sequences": 269971,
                        "canonical_features": 81,
                        "attack_categories": ["DNS Reflection", "LDAP", "MSSQL", "NetBIOS", "NTP", "SNMP", "SSDP", "UDP", "Syn Flood", "TFTP"],
                        "description": "Modern reflection and exploitation DDoS signature benchmark containing 18 discrete attack scenarios."
                    },
                    {
                        "id": "CICIDS2023",
                        "name": "CICIoT2023",
                        "official_name": "CIC Internet of Things Security Dataset 2023",
                        "provenance": "Official UNB IoT CSV Archive",
                        "status": "EXCLUDED",
                        "train_sequences": 0,
                        "test_sequences": 0,
                        "canonical_features": 46,
                        "attack_categories": ["IoT DDoS", "Mirai", "Reconnaissance", "Spoofing"],
                        "description": "Large-scale IoT network traffic dataset.",
                        "exclusion_note": "CICIoT2023 was not included in the final experimental evaluation."
                    }
                ]
            })
            return

        # Static SPA File Serving
        if self.dist_dir.exists():
            clean_path = path.lstrip("/")
            target_file = self.dist_dir / clean_path
            if target_file.exists() and target_file.is_file():
                mime_type, _ = mimetypes.guess_type(str(target_file))
                with open(target_file, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mime_type or "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                # SPA Fallback to index.html
                index_file = self.dist_dir / "index.html"
                if index_file.exists():
                    with open(index_file, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return

        self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/v1/detect":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                req_data = json.loads(body.decode("utf-8"))

                dataset_id = req_data.get("dataset_id", "CICIDS2017")
                model_name = req_data.get("model_name", "baseline_seq2seq")
                sample_id = req_data.get("sample_id")
                sequence_data = req_data.get("sequence")
                sample_meta = None

                if sample_id:
                    bank = inference_engine.get_sample_bank()
                    for ds, samples in bank.items():
                        for s in samples:
                            if s["id"] == sample_id:
                                sequence_data = s["sequence"]
                                sample_meta = s
                                break
                        if sequence_data is not None:
                            break

                if sequence_data is None:
                    self._send_json({"error": "No sequence or sample_id provided"}, status_code=400)
                    return

                result = inference_engine.predict(
                    sequence=sequence_data,
                    dataset_id=dataset_id,
                    model_name=model_name
                )

                if sample_meta:
                    result["sample_info"] = {
                        "id": sample_meta["id"],
                        "name": sample_meta["name"],
                        "ground_truth_label": sample_meta["label"],
                        "ground_truth": sample_meta["ground_truth"],
                        "is_correct_classification": (result["prediction"] == sample_meta["label"])
                    }

                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, status_code=500)
            return

        elif path.startswith("/api/v1/upload-csv"):
            try:
                query = urllib.parse.parse_qs(parsed.query)
                dataset_id = query.get("dataset_id", ["CICIDS2017"])[0]
                model_name = query.get("model_name", ["baseline_seq2seq"])[0]

                content_length = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_length)
                
                # Simple multipart / text parser
                lines = raw_body.decode("utf-8", errors="ignore").splitlines()
                csv_lines = [l for l in lines if "," in l and not l.startswith("---")]
                
                if len(csv_lines) < 30:
                    self._send_json({"error": "CSV must contain at least 30 flow rows."}, status_code=400)
                    return

                reader = csv.reader(csv_lines)
                headers = next(reader, [])
                rows = []
                for r in reader:
                    num_row = []
                    for val in r:
                        try:
                            num_row.append(float(val.strip()))
                        except ValueError:
                            pass
                    if num_row:
                        rows.append(num_row)
                    if len(rows) >= 30:
                        break

                input_dim = inference_engine._get_input_dim(dataset_id)
                seq = np.array(rows[:30], dtype=np.float32)
                if seq.shape[1] < input_dim:
                    pad = np.zeros((30, input_dim - seq.shape[1]), dtype=np.float32)
                    seq = np.hstack([seq, pad])
                else:
                    seq = seq[:, :input_dim]

                result = inference_engine.predict(seq, dataset_id, model_name)
                result["uploaded_filename"] = "custom_flow.csv"
                result["parsed_rows"] = len(rows)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": f"CSV parse error: {str(e)}"}, status_code=500)
            return

        self._send_json({"error": "Endpoint not found"}, status_code=404)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    server = ThreadingHTTPServer((host, port), AnomalyDetectionHTTPHandler)
    print("================================================================================")
    print("               NETWORK ANOMALY DETECTION: LIVE INFERENCE SERVER                 ")
    print(f" URL: http://localhost:{port}")
    print(f" Host: {host} | Port: {port}")
    print("================================================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Shutting down server...")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Anomaly Detection Web Application")
    parser.add_argument("--host", default="0.0.0.0", help="Host binding")
    parser.add_argument("--port", type=int, default=8000, help="Port binding")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
