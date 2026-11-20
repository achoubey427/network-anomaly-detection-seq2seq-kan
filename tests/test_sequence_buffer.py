"""
Unit tests for streaming sequence buffer manager.
"""

import unittest
from src.pipeline.sequence_buffer import FlowBuffer, SequenceBufferManager


class TestSequenceBuffer(unittest.TestCase):
    def test_flow_buffer_window(self):
        buf = FlowBuffer(window_size=5)
        for i in range(4):
            ready, seq = buf.add_packet({"id": i})
            self.assertFalse(ready)
            self.assertIsNone(seq)
            
        ready, seq = buf.add_packet({"id": 4})
        self.assertTrue(ready)
        self.assertEqual(len(seq), 5)
        self.assertEqual(seq[0]["id"], 0)
        self.assertEqual(seq[-1]["id"], 4)

    def test_multi_flow_manager(self):
        mgr = SequenceBufferManager(window_size=3)
        # Flow 1
        mgr.push_packet("flow1", {"val": 1})
        mgr.push_packet("flow1", {"val": 2})
        ready1, seq1 = mgr.push_packet("flow1", {"val": 3})
        self.assertTrue(ready1)
        self.assertEqual(len(seq1), 3)

        # Flow 2 should still have only 1 packet
        ready2, seq2 = mgr.push_packet("flow2", {"val": 10})
        self.assertFalse(ready2)
        self.assertIsNone(seq2)


if __name__ == "__main__":
    unittest.main()
