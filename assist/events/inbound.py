"""Legacy inbound-only JSON history retained for reading and migration.

The active SMS routes now use SmsArchive outside Git and admit matching durable
Runs before acknowledgment. This class neither owns current ingestion nor
migrates old records automatically; existing files remain untouched.
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
