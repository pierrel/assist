"""Private bounded Playwright state between disposable browser turns."""
from __future__ import annotations

import json
import os
import stat
from uuid import uuid4

from assist.browser import authority


FILE = "browser-storage.json"
MAX_BYTES = 48 * 1024


def _directory(root: str, thread_id: str) -> int:
    path = authority._thread_dir(root, thread_id)
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def _encoded(state: dict) -> bytes:
    if (not isinstance(state, dict) or set(state) != {"cookies", "origins"}
            or not isinstance(state["cookies"], list)
            or not isinstance(state["origins"], list)):
        raise ValueError("invalid Playwright storage state")
    encoded = json.dumps(state, separators=(",", ":")).encode()
    if len(encoded) > MAX_BYTES:
        raise ValueError("Playwright storage state exceeds limit")
    return encoded


def load(root: str, thread_id: str) -> dict | None:
    """Read only a private regular file outside every sandbox mount."""
    directory = _directory(root, thread_id)
    try:
        try:
            fd = os.open(FILE, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or info.st_nlink != 1
                    or info.st_size > MAX_BYTES):
                raise ValueError("unsafe Playwright storage file")
            data = stream.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("Playwright storage state exceeds limit")
        state = json.loads(data)
        _encoded(state)
        return state
    finally:
        os.close(directory)


def save(root: str, thread_id: str, run_id: str,
         generation: str, state: dict) -> None:
    """Replace state only while its exact browser lease still owns the turn."""
    data = _encoded(state)
    with authority.fence(root, thread_id) as current:
        lease = current.lease
        if (lease is None or lease["owner_run_id"] != run_id
                or generation not in lease["generations"]):
            raise RuntimeError("browser storage owner changed")
        directory = _directory(root, thread_id)
        temporary = f".browser-storage-{uuid4().hex}"
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         os.O_NOFOLLOW, 0o600, dir_fd=directory)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, FILE, src_dir_fd=directory,
                       dst_dir_fd=directory)
            os.fsync(directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
            os.close(directory)
