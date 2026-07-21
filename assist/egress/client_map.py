"""The proxy's client-attribution map: egress-network IP → thread id.

Written by ``SandboxManager`` (the one component that creates per-turn
sandbox containers and knows their thread), read by the egress proxy on
every allowlist miss to enforce THREAD-scoped grants. Lives in the mounted
``approvals/`` subdir beside the projection. Atomic replace under a module
lock; the create→write race (a container connecting before its entry lands)
degrades to a fail-closed transient deny that self-heals on retry.
"""
from __future__ import annotations

import json
import logging
import os
import threading

from assist.egress.store import APPROVALS_SUBDIR

logger = logging.getLogger(__name__)

CLIENT_MAP_FILE = "client-map.json"
_LOCK = threading.Lock()


def _path(egress_dir: str) -> str:
    return os.path.join(egress_dir, APPROVALS_SUBDIR, CLIENT_MAP_FILE)


def _read(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write(path: str, entries: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(entries, f)
    os.replace(tmp, path)


def record_client(egress_dir: str, ip: str, tid: str) -> None:
    """Map a just-started sandbox's IP to its thread (overwrite-on-reuse:
    docker reassigns IPs across per-turn containers; newest writer wins).
    Best-effort — a failure here only costs thread-scoped grants for this
    turn (fail-closed denies), never the turn itself."""
    try:
        with _LOCK:
            path = _path(egress_dir)
            entries = _read(path)
            entries[ip] = tid
            _write(path, entries)
    except Exception:
        logger.warning("egress: client-map write failed for %s -> %s "
                       "(grants deny fail-closed this turn)", ip, tid,
                       exc_info=True)


def forget_client(egress_dir: str, ip: str) -> None:
    """Reap-path cleanup; best-effort (a stale entry is overwritten at the
    IP's next reuse anyway)."""
    try:
        with _LOCK:
            path = _path(egress_dir)
            entries = _read(path)
            if entries.pop(ip, None) is not None:
                _write(path, entries)
    except Exception:
        logger.warning("egress: client-map cleanup failed for %s", ip,
                       exc_info=True)
