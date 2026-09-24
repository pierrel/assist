"""Generation-bound proxy client attribution shared by shell and browser containers."""
from __future__ import annotations

import fcntl
import ipaddress
import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

from assist.egress.store import APPROVALS_SUBDIR, approvals_dir_is_safe


CLIENT_MAP_FILE = "client-map.json"


def configured_directory() -> str | None:
    """Return a proxy-only map directory outside every thread workspace."""
    explicit = os.environ.get("ASSIST_EGRESS_CLIENT_MAP_DIR")
    directory = explicit
    if not directory:
        approvals = os.environ.get("ASSIST_EGRESS_APPROVALS_DIR")
        directory = os.path.join(approvals, APPROVALS_SUBDIR) if approvals else None
    if directory is None:
        return None
    if not os.path.isabs(directory) or not approvals_dir_is_safe(directory):
        if not explicit:
            return None
        raise RuntimeError("proxy client map must be outside the thread root")
    os.makedirs(directory, mode=0o700, exist_ok=True)
    stat = os.stat(directory)
    if stat.st_uid != os.getuid() or stat.st_mode & 0o022:
        raise RuntimeError("proxy client map directory has unsafe ownership or permissions")
    return os.path.realpath(directory)


@dataclass(frozen=True)
class ClientRecord:
    thread_id: str
    generation: str
    kind: str
    browser_mode: str | None = None
    internal_host: str | None = None

    def __post_init__(self) -> None:
        if (not self.thread_id or len(self.thread_id) > 128
                or not self.generation or len(self.generation) > 128
                or self.kind not in {"sandbox", "browser"}):
            raise ValueError("invalid proxy client identity")
        if self.kind == "sandbox":
            if self.browser_mode is not None or self.internal_host is not None:
                raise ValueError("sandbox has browser policy fields")
        elif (self.browser_mode not in {"public", "internal"}
              or (self.browser_mode == "internal") != bool(self.internal_host)):
            raise ValueError("invalid browser policy")

    def to_dict(self) -> dict[str, str]:
        value = {"thread_id": self.thread_id, "generation": self.generation,
                 "kind": self.kind}
        if self.browser_mode is not None:
            value["browser_mode"] = self.browser_mode
        if self.internal_host is not None:
            value["internal_host"] = self.internal_host
        return value


def _validate_ip(ip: str) -> str:
    return str(ipaddress.ip_address(ip))


@contextmanager
def _locked(directory: str, timeout: float = 5):
    os.makedirs(directory, mode=0o700, exist_ok=True)
    with open(os.path.join(directory, ".client-map.lock"), "a+b") as lock:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("proxy client-map lock timed out")
                time.sleep(0.01)
        try:
            yield os.path.join(directory, CLIENT_MAP_FILE)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read(path: str) -> dict[str, dict[str, str]]:
    try:
        with open(path, encoding="utf-8") as stream:
            entries = json.load(stream)
    except FileNotFoundError:
        return {}
    if not isinstance(entries, dict):
        raise ValueError("invalid proxy client map")
    for ip, value in entries.items():
        if not isinstance(ip, str) or not isinstance(value, dict):
            raise ValueError("invalid proxy client map entry")
        _validate_ip(ip)
        if value != ClientRecord(**value).to_dict():
            raise ValueError("invalid proxy client map record")
    return entries


def _write(path: str, entries: dict[str, dict[str, str]]) -> None:
    fd, temp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".client-map-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(entries, stream, separators=(",", ":"), sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def record_client(directory: str, ip: str, record: ClientRecord) -> None:
    """Publish an exact running generation before its container is exposed."""
    ip = _validate_ip(ip)
    with _locked(directory) as path:
        entries = _read(path)
        entries[ip] = record.to_dict()
        _write(path, entries)


def forget_client(directory: str, ip: str, generation: str) -> None:
    """A delayed cleanup cannot erase a later owner of a recycled IP."""
    ip = _validate_ip(ip)
    with _locked(directory) as path:
        entries = _read(path)
        if entries.get(ip, {}).get("generation") == generation:
            del entries[ip]
            _write(path, entries)


def prune_absent_browser_clients(
        directory: str,
        running_endpoints: Callable[[], set[tuple[str, str]]]) -> int:
    """Compare-delete stale browser IP+generation records after an unlocked scan."""
    with _locked(directory) as path:
        snapshot = {ip: value for ip, value in _read(path).items()
                    if value["kind"] == "browser"}
    # Docker must never run while holding the map lock needed by admission.
    live = running_endpoints()
    if (not isinstance(live, set) or any(
            not isinstance(item, tuple) or len(item) != 2
            or not isinstance(item[0], str) or not item[0]
            or not isinstance(item[1], str) or _validate_ip(item[1]) != item[1]
            for item in live)):
        raise ValueError("invalid running browser endpoint scan")
    with _locked(directory) as path:
        entries = _read(path)
        stale = [ip for ip, value in snapshot.items()
                 if (value["generation"], ip) not in live
                 and entries.get(ip) == value]
        for ip in stale:
            del entries[ip]
        if stale:
            _write(path, entries)
        return len(stale)


def browser_records(directory: str, thread_id: str) -> dict[str, ClientRecord]:
    """Read exact browser client-map records for one thread."""
    with _locked(directory) as path:
        return {ip: ClientRecord(**value) for ip, value in _read(path).items()
                if value["kind"] == "browser" and value["thread_id"] == thread_id}


def read_client(directory: str, ip: str) -> ClientRecord | None:
    """Inspect the current published record under the interprocess lock."""
    with _locked(directory) as path:
        value = _read(path).get(_validate_ip(ip))
    return ClientRecord(**value) if value else None
