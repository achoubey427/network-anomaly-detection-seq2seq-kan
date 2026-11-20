"""
Per-flow rolling Sequence Buffer for real-time streaming anomaly detection.
Buffers incoming network packets into fixed-length temporal windows.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class FlowBuffer:
    """Buffer for a single network flow."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.packet_count = 0

    def add_packet(self, packet_features: Dict[str, Any]) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
        """Add packet to buffer. Returns (is_ready, sequence) if buffer reached window_size."""
        self.buffer.append(packet_features)
        self.packet_count += 1
        if len(self.buffer) == self.window_size:
            return True, list(self.buffer)
        return False, None


class SequenceBufferManager:
    """Manages rolling buffers across multiple independent network flows."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.flows: Dict[str, FlowBuffer] = {}

    def push_packet(self, flow_key: str, packet_features: Dict[str, Any]) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
        """Push packet to the flow's buffer and return ready sequence if full window is available."""
        if flow_key not in self.flows:
            self.flows[flow_key] = FlowBuffer(window_size=self.window_size)
        return self.flows[flow_key].add_packet(packet_features)

    def reset_flow(self, flow_key: str):
        """Reset buffer for a given flow."""
        if flow_key in self.flows:
            del self.flows[flow_key]

    def reset_all(self):
        """Reset all active flow buffers."""
        self.flows.clear()
