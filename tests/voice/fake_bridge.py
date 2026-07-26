"""Protocol-level fake for the phone-side voice bridge."""
from __future__ import annotations

import json
from typing import Any


class FakeBridge:
    def __init__(self, websocket):
        self.websocket = websocket

    def send_control(self, kind: str, **fields: Any) -> None:
        self.websocket.send_text(json.dumps({"type": kind, **fields}))

    def ring(
        self, call_id: str = "boot-1", caller: str = "+15555550100"
    ) -> None:
        self.send_control("ring", call_id=call_id, caller=caller)

    def send_pcm(self, pcm: bytes) -> None:
        self.websocket.send_bytes(pcm)

    def receive_control(self) -> dict[str, Any]:
        return json.loads(self.websocket.receive_text())

    def receive_pcm(self) -> bytes:
        return self.websocket.receive_bytes()
