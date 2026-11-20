"""
Synthetic Modbus/TCP Traffic Generator.
Generates normal baseline traffic across 6 industrial flows with periodic attack injection
for real-time pipeline demonstration and integration testing.
Clearly isolated in data/synthetic/ to avoid contaminating research benchmarks.
"""

import random
import time
from typing import Any, Dict, Generator, List


class SyntheticModbusGenerator:
    """Simulates multi-flow Modbus/TCP industrial control communication."""

    FLOWS = [
        {"src_ip": "192.168.1.10", "dst_ip": "192.168.1.50", "unit_id": 1, "desc": "PLC-1 to Sensor-Bank"},
        {"src_ip": "192.168.1.11", "dst_ip": "192.168.1.50", "unit_id": 2, "desc": "PLC-2 to Sensor-Bank"},
        {"src_ip": "192.168.1.20", "dst_ip": "192.168.1.60", "unit_id": 1, "desc": "SCADA-HMI to Actuator-1"},
        {"src_ip": "192.168.1.21", "dst_ip": "192.168.1.61", "unit_id": 1, "desc": "SCADA-HMI to Actuator-2"},
        {"src_ip": "192.168.1.30", "dst_ip": "192.168.1.70", "unit_id": 3, "desc": "Historian to Turbine"},
        {"src_ip": "192.168.1.35", "dst_ip": "192.168.1.75", "unit_id": 4, "desc": "Safety System to Valve-Rack"},
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tx_counter = 1000

    def generate_packet(self, is_attack: bool = False, attack_type: str = "register_tamper") -> Dict[str, Any]:
        """Generate a single synthetic Modbus/TCP packet event."""
        flow = self.rng.choice(self.FLOWS)
        self.tx_counter += 1
        
        # Base normal values
        function_code = self.rng.choice([3, 4, 1, 2])  # Standard read operations
        register_address = self.rng.randint(100, 150)
        register_value = round(self.rng.gauss(50.0, 5.0), 2)
        packet_length = self.rng.choice([64, 72, 80])
        
        if is_attack:
            if attack_type == "register_tamper":
                # Stealthy out-of-bound write
                function_code = self.rng.choice([6, 16])  # Force write
                register_value = round(self.rng.uniform(9000.0, 9999.0), 2)
            elif attack_type == "unauthorized_scan":
                register_address = self.rng.randint(9000, 9999)
                function_code = 125  # Illegal function code
            elif attack_type == "flood_dos":
                packet_length = self.rng.randint(500, 1500)
                
        return {
            "timestamp": time.time(),
            "src_ip": flow["src_ip"],
            "dst_ip": flow["dst_ip"],
            "function_code": function_code,
            "unit_id": flow["unit_id"],
            "protocol_id": 0,
            "direction": "request",
            "transaction_id": self.tx_counter,
            "packet_length": packet_length,
            "register_address": register_address,
            "register_value": register_value,
            "is_anomaly": 1 if is_attack else 0,
            "attack_type": attack_type if is_attack else "normal"
        }

    def generate_stream(self, n_packets: int = 1000, attack_prob: float = 0.05) -> Generator[Dict[str, Any], None, None]:
        """Stream generator yielding synthetic packets."""
        for _ in range(n_packets):
            is_attack = self.rng.random() < attack_prob
            yield self.generate_packet(is_attack=is_attack)
