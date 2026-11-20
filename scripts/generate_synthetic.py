"""
Generate synthetic Modbus/TCP traffic CSV for real-time pipeline demo.
Saved strictly to data/synthetic/ to avoid mixing with research benchmark data.
"""

import csv
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline.synthetic_generator import SyntheticModbusGenerator


def main():
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "modbus_demo_traffic.csv"

    generator = SyntheticModbusGenerator(seed=42)
    packets = []
    
    # 5,000 normal packets
    for _ in range(5000):
        packets.append(generator.generate_packet(is_attack=False))
        
    # 500 mixed evaluation packets (with attacks)
    for _ in range(500):
        is_atk = generator.rng.random() < 0.2
        atk_type = generator.rng.choice(["register_tamper", "unauthorized_scan", "flood_dos"])
        packets.append(generator.generate_packet(is_attack=is_atk, attack_type=atk_type))

    fieldnames = list(packets[0].keys())
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(packets)

    print(f"Successfully generated {len(packets)} synthetic packets -> {out_file}")


if __name__ == "__main__":
    main()
