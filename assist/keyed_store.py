"""Shared base for the single-file keyed JSON stores.

The repo grew two shape-frozen copies of this pattern (``assist/geo/``'s
``RegionRegistry`` and ``ProposalStore``) whose comments name the trigger:
a THIRD instance factors a shared base. ``EgressStore`` is the third — this
is that base. The two geo stores stay un-migrated for now (behavior-frozen;
migrating them is a mechanical follow-up, out of the egress feature's scope).

Shape: one JSON object file, key -> record dict; a ``threading.Lock`` around
read-modify-write; lenient reads (missing/corrupt/wrong-shape ⇒ empty, so a
damaged file can't wedge its callers); atomic same-dir tmp + ``os.replace``
writes (readers see whole-old or whole-new, never partial — which is also
what makes a LOCK-FREE ``peek`` safe for the event-loop render path).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class KeyedJsonStore(Generic[T]):
    """Subclasses set ``FILENAME`` and implement ``_key``/``_to_dict``/``_from_dict``
    (return ``None`` from ``_from_dict`` to skip a malformed record)."""

    FILENAME: str = ""

    def __init__(self, root_dir: str):
        self._path = os.path.join(root_dir, self.FILENAME)
        self._lock = threading.Lock()

    # --- record adapters (subclass contract) --------------------------------
    def _key(self, rec: T) -> str:
        raise NotImplementedError

    def _to_dict(self, rec: T) -> dict:
        raise NotImplementedError

    def _from_dict(self, d: dict) -> T | None:
        raise NotImplementedError

    # --- unlocked internals (callers hold self._lock) -----------------------
    def _load(self) -> dict[str, T]:
        recs, _ = self._load_raw()
        return recs

    def _load_raw(self) -> tuple[dict[str, T], bool]:
        """(records, dirty) — dirty when malformed entries were skipped, so a
        caller that intends to write anyway persists the cleanup."""
        try:
            with open(self._path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}, False
        if not isinstance(data, dict):
            logger.warning("%s is not an object; ignoring", self._path)
            return {}, False
        out: dict[str, T] = {}
        dirty = False
        for _, d in data.items():
            rec = self._from_dict(d) if isinstance(d, dict) else None
            if rec is None:
                dirty = True
                continue
            out[self._key(rec)] = rec
        return out, dirty

    def _write(self, recs: dict[str, T]) -> None:
        tmp = f"{self._path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({k: self._to_dict(r) for k, r in recs.items()}, f)
        os.replace(tmp, self._path)

    # --- locked API ---------------------------------------------------------
    def all(self) -> list[T]:
        with self._lock:
            return list(self._load().values())

    def get(self, key: str) -> T | None:
        with self._lock:
            return self._load().get(key)

    def put(self, rec: T) -> T:
        """Insert or replace by key."""
        with self._lock:
            recs = self._load()
            recs[self._key(rec)] = rec
            self._write(recs)
            return rec

    def remove(self, key: str) -> bool:
        with self._lock:
            recs = self._load()
            if recs.pop(key, None) is None:
                return False
            self._write(recs)
            return True

    def peek(self) -> list[T]:
        """LOCK-FREE, side-effect-free read for the event-loop render path
        (atomic-replace writes ⇒ whole-old-or-new; the ``MESSAGE_BACKLOG.peek``
        discipline). Any problem degrades to [] — never a 500."""
        try:
            with open(self._path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return []
            return [r for r in (self._from_dict(d) for d in data.values()
                                if isinstance(d, dict)) if r is not None]
        except Exception:
            return []
