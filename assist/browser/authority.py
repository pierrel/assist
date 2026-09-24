"""Host-only per-thread browser lease and short admission/publication fence.

The file lives beside, never inside, the thread's mounted workspace, scratch
or agent directories. Missing state on an older thread is not a clean proof.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from contextlib import contextmanager


STATE_FILE = "browser-authority.json"
LOCK_FILE = ".browser-authority.lock"


def _thread_dir(root: str, thread_id: str) -> str:
    if (not isinstance(thread_id, str) or not thread_id
            or thread_id in {".", ".."} or "/" in thread_id
            or "\\" in thread_id or "\0" in thread_id):
        raise ValueError("invalid browser thread ID")
    directory = os.path.join(root, thread_id)
    if os.path.islink(directory) or not os.path.isdir(directory):
        raise RuntimeError("browser thread directory unavailable")
    return directory


def _read(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as stream:
            state = json.load(stream)
    except FileNotFoundError:
        return {"covered": False, "lease": None}
    if (not isinstance(state, dict) or set(state) != {"covered", "lease"}
            or not isinstance(state["covered"], bool)):
        raise RuntimeError("browser authority state is invalid")
    lease = state["lease"]
    if lease is not None:
        if (not isinstance(lease, dict)
                or set(lease) != {"owner_run_id", "sequence", "generations"}
                or not isinstance(lease["owner_run_id"], str)
                or not lease["owner_run_id"]
                or len(lease["owner_run_id"]) > 256
                or not isinstance(lease["sequence"], int)
                or lease["sequence"] < 0
                or not isinstance(lease["generations"], list)
                or any(not isinstance(item, str) or not item or len(item) > 128
                       for item in lease["generations"])):
            raise RuntimeError("browser lease is invalid")
    return state


def _write(path: str, state: dict) -> None:
    directory = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=".browser-authority-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Authority:
    def __init__(self, path: str):
        self.path = path
        self.state = _read(path)

    @property
    def covered(self) -> bool:
        return self.state["covered"]

    @property
    def lease(self) -> dict | None:
        return self.state["lease"]

    def save(self) -> None:
        _write(self.path, self.state)

    def mark_covered(self) -> None:
        self.state["covered"] = True
        self.save()

    def begin(self, owner_run_id: str, sequence: int) -> None:
        if not self.covered:
            raise RuntimeError("browser startup reconciliation is incomplete")
        lease = self.lease
        if lease is None:
            self.state["lease"] = {"owner_run_id": owner_run_id,
                                   "sequence": sequence, "generations": []}
            self.save()
        elif (lease["owner_run_id"], lease["sequence"]) != (owner_run_id, sequence):
            raise RuntimeError("previous browser lease remains active")

    def add_generation(self, owner_run_id: str, generation: str) -> None:
        lease = self.lease
        if lease is None or lease["owner_run_id"] != owner_run_id:
            raise RuntimeError("browser lease changed during startup")
        if generation not in lease["generations"]:
            lease["generations"].append(generation)
            self.save()

    def remove_generation(self, owner_run_id: str, generation: str) -> None:
        lease = self.lease
        if lease is None or lease["owner_run_id"] != owner_run_id:
            raise RuntimeError("browser lease changed during teardown")
        if generation in lease["generations"]:
            lease["generations"].remove(generation)
            self.save()

    def clear(self, owner_run_id: str) -> None:
        lease = self.lease
        if lease is None:
            return
        if lease["owner_run_id"] != owner_run_id or lease["generations"]:
            raise RuntimeError("browser lease teardown is unconfirmed")
        self.state["lease"] = None
        self.save()

    def clear_reconciled(self, owner_run_id: str) -> None:
        """A successful scoped scan/kill and map check supersede cached IDs."""
        lease = self.lease
        if lease is None:
            return
        if lease["owner_run_id"] != owner_run_id:
            raise RuntimeError("browser lease owner changed")
        self.state["lease"] = None
        self.save()


@contextmanager
def fence(root: str, thread_id: str, timeout: float = 5):
    """Serialize only short local journal, lease and map-publication work."""
    directory = _thread_dir(root, thread_id)
    fd = os.open(os.path.join(directory, LOCK_FILE),
                 os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("browser admission fence timed out")
                time.sleep(0.01)
        try:
            yield Authority(os.path.join(directory, STATE_FILE))
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def mark_new_thread(root: str, thread_id: str) -> None:
    """Mark a new private thread directory clean before it is published."""
    with fence(root, thread_id) as authority:
        if not os.path.exists(authority.path):
            authority.mark_covered()
