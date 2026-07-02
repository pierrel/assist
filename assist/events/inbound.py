"""Durable inbound-message log — records every inbound message BEFORE the 200, keyed by a
content-hash ``message_id``, so the ack means "assist owns this text" (not "the LLM was
reachable"): a crash before the triage turn leaves a recoverable record, never a lost text.
The same record is the dedup key — the phone re-POSTs only what it never got a 2xx for, and
an atomic ``O_EXCL`` create claims a message_id exactly once (no lock, no TOCTOU) so the
daemon's own retry race can't double-dispatch.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class InboundLog:
    def __init__(self, root_dir: str):
        self._dir = os.path.join(root_dir, "inbound")

    @staticmethod
    def valid_id(message_id: str) -> bool:
        """A message_id must be a single safe filename segment (the phone sends a sha256
        hex). Rejecting anything else keeps the record path traversal-proof by construction."""
        return bool(_SAFE_ID.match(message_id or ""))

    def claim(self, message_id: str, sender: str, text: str) -> bool:
        """Atomically record + claim ``message_id``. Returns True if newly recorded (the
        caller should dispatch it), False if already seen (duplicate — do nothing). Raises
        ValueError for an unsafe message_id (the route maps that to 400)."""
        if not self.valid_id(message_id):
            raise ValueError(f"unsafe message_id: {message_id!r}")
        os.makedirs(self._dir, exist_ok=True)
        path = os.path.join(self._dir, f"{message_id}.json")
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump({"message_id": message_id, "sender": sender, "text": text,
                       "received_at": datetime.now(timezone.utc).isoformat()}, f)
        return True
