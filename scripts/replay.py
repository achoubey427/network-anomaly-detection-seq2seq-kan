"""
Replay script: Reads recorded/synthetic Modbus/TCP traffic CSV and pushes events
into the streaming backend or SequenceBuffer to demonstrate real-time detection.
"""

import argparse
import csv
from pathlib import Path
import sys
import time
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def replay_csv(csv_path: str, server_url: str = "http://127.0.0.1:8000/api/v1/ingest", delay_sec: float = 0.05):
    path = Path(csv_path)
    if not path.exists():
        print(f"Error: File {csv_path} does not exist.")
        return

    print(f"Replaying packets from {csv_path} to {server_url} (delay: {delay_sec}s)...")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            count += 1
            try:
                resp = requests.post(server_url, json=row, timeout=2.0)
                if count % 50 == 0:
                    print(f"Ingested {count} packets... (Last status: {resp.status_code})")
            except Exception as e:
                print(f"Connection error on packet {count}: {e}")
                time.sleep(1.0)
            time.sleep(delay_sec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay Modbus/TCP traffic to live backend")
    parser.add_argument("--csv", default="data/synthetic/modbus_demo_traffic.csv", help="Path to CSV file")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/ingest", help="Backend ingest API URL")
    parser.add_argument("--delay", type=float, default=0.02, help="Delay between packets in seconds")
    args = parser.parse_args()
    replay_csv(args.csv, args.url, args.delay)
