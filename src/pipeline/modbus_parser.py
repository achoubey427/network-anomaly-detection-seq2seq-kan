"""
Modbus/TCP Network Packet Parser.
Extracts 11 core features (6 categorical, 5 numerical) from Modbus/TCP frames or event payloads.
"""

from typing import Any, Dict, Optional
import time


class ModbusParser:
    """Parser for industrial Modbus/TCP protocol frames."""

    FUNCTION_NAMES = {
        1: "Read Coils",
        2: "Read Discrete Inputs",
        3: "Read Holding Registers",
        4: "Read Input Registers",
        5: "Write Single Coil",
        6: "Write Single Register",
        15: "Write Multiple Coils",
        16: "Write Multiple Registers"
    }

    def __init__(self):
        self.last_packet_time: Dict[str, float] = {}

    def parse_event_dict(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a JSON or dictionary event into canonical feature representation."""
        src_ip = str(event.get("src_ip", "192.168.1.10"))
        dst_ip = str(event.get("dst_ip", "192.168.1.50"))
        flow_key = f"{src_ip}->{dst_ip}"
        
        current_time = float(event.get("timestamp", time.time()))
        last_time = self.last_packet_time.get(flow_key, current_time)
        inter_arrival_ms = max(0.0, (current_time - last_time) * 1000.0)
        self.last_packet_time[flow_key] = current_time

        parsed = {
            # Categorical features
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "function_code": int(event.get("function_code", 3)),
            "unit_id": int(event.get("unit_id", 1)),
            "protocol_id": int(event.get("protocol_id", 0)),
            "direction": str(event.get("direction", "request")),
            # Numerical features
            "transaction_id": float(event.get("transaction_id", 0)),
            "packet_length": float(event.get("packet_length", 64)),
            "register_address": float(event.get("register_address", 0)),
            "register_value": float(event.get("register_value", 0)),
            "inter_arrival_ms": float(inter_arrival_ms),
            # Metadata
            "flow_key": flow_key,
            "timestamp": current_time,
            "is_anomaly_ground_truth": int(event.get("is_anomaly", 0))
        }
        return parsed
