"""Durable, server-owned pins for visible assistant responses."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime


MAX_PINS = 50
MAX_PIN_TEXT_BYTES = 32 * 1024
MAX_STORE_BYTES = 2 * 1024 * 1024
_LOCK = threading.Lock()
_FILENAME = "phone-pins.json"


@dataclass(frozen=True)
class PhonePin:
    """An immutable assistant response saved by a phone client."""

    response_id: str
    text: str
    created_at: str


def _path(thread_dir: str) -> str:
    return os.path.join(thread_dir, _FILENAME)


def _read(thread_dir: str) -> list[PhonePin]:
    try:
        if os.stat(_path(thread_dir)).st_size > MAX_STORE_BYTES:
            raise ValueError("phone pin store is too large")
        with open(_path(thread_dir), encoding="utf-8") as source:
            raw = json.load(source)
    except FileNotFoundError:
        return []
    if not isinstance(raw, list):
        raise ValueError("invalid phone pin store")
    pins: list[PhonePin] = []
    for value in raw:
        if not isinstance(value, dict):
            raise ValueError("invalid phone pin record")
        response_id = value.get("response_id")
        text = value.get("text")
        created_at = value.get("created_at")
        if not all(isinstance(item, str) for item in (response_id, text, created_at)):
            raise ValueError("invalid phone pin record")
        if len(text.encode("utf-8")) > MAX_PIN_TEXT_BYTES:
            raise ValueError("phone pin record is too large")
        pins.append(PhonePin(response_id, text, created_at))
    if len(pins) > MAX_PINS:
        raise ValueError("phone pin store has too many pins")
    return pins


def _write(thread_dir: str, pins: list[PhonePin]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".phone-pins-", dir=thread_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            json.dump([asdict(pin) for pin in pins], destination, ensure_ascii=False)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, _path(thread_dir))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def list_pins(thread_dir: str) -> list[PhonePin]:
    """Return pins newest first."""
    with _LOCK:
        return list(reversed(_read(thread_dir)))


def create_pin(thread_dir: str, response_id: str, text: str) -> PhonePin:
    """Persist one response pin, returning an existing matching pin unchanged."""
    with _LOCK:
        if len(text.encode("utf-8")) > MAX_PIN_TEXT_BYTES:
            raise ValueError("pinned response is too large")
        pins = _read(thread_dir)
        existing = next((pin for pin in pins if pin.response_id == response_id), None)
        if existing is not None:
            return existing
        if len(pins) >= MAX_PINS:
            raise ValueError("pin limit reached")
        pin = PhonePin(response_id=response_id, text=text,
                       created_at=datetime.now(UTC).isoformat())
        pins.append(pin)
        _write(thread_dir, pins)
        return pin


def delete_pin(thread_dir: str, response_id: str) -> bool:
    """Remove a pin and report whether it was present."""
    with _LOCK:
        pins = _read(thread_dir)
        kept = [pin for pin in pins if pin.response_id != response_id]
        if len(kept) == len(pins):
            return False
        _write(thread_dir, kept)
        return True
