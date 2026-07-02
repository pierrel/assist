"""Unit tests for the durable inbound-message log (persist-before-200 + dedup)."""
import json
import os
import tempfile
import unittest

from assist.events.inbound import InboundLog


class TestInboundLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.log = InboundLog(self.tmp)

    def test_claims_and_persists_record(self):
        assert self.log.claim("abc123", "+15551234567", "hi there") is True
        path = os.path.join(self.tmp, "inbound", "abc123.json")
        assert os.path.isfile(path)                       # persisted permanently
        rec = json.load(open(path))
        assert rec["sender"] == "+15551234567"
        assert rec["text"] == "hi there"
        assert rec["message_id"] == "abc123"
        assert rec["received_at"]                          # server-stamped timestamp

    def test_duplicate_returns_false_without_overwrite(self):
        assert self.log.claim("dup", "+1", "first") is True
        # a re-POST with different content must NOT overwrite the original record
        assert self.log.claim("dup", "+1", "second") is False
        rec = json.load(open(os.path.join(self.tmp, "inbound", "dup.json")))
        assert rec["text"] == "first"

    def test_rejects_unsafe_message_id(self):
        with self.assertRaises(ValueError):
            self.log.claim("../escape", "+1", "x")
        assert InboundLog.valid_id("a1B2-_c3") and not InboundLog.valid_id("a/b")


if __name__ == "__main__":
    unittest.main()
