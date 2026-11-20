"""
Unit tests for Modbus/TCP packet parser.
"""

import unittest
from src.pipeline.modbus_parser import ModbusParser


class TestModbusParser(unittest.TestCase):
    def test_parse_event_dict(self):
        parser = ModbusParser()
        raw_event = {
            "src_ip": "192.168.1.10",
            "dst_ip": "192.168.1.50",
            "function_code": 3,
            "unit_id": 1,
            "transaction_id": 1001,
            "packet_length": 64,
            "register_address": 102,
            "register_value": 45.2,
            "is_anomaly": 0
        }
        parsed = parser.parse_event_dict(raw_event)
        self.assertEqual(parsed["src_ip"], "192.168.1.10")
        self.assertEqual(parsed["dst_ip"], "192.168.1.50")
        self.assertEqual(parsed["flow_key"], "192.168.1.10->192.168.1.50")
        self.assertEqual(parsed["function_code"], 3)
        self.assertEqual(parsed["register_value"], 45.2)


if __name__ == "__main__":
    unittest.main()
