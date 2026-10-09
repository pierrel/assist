"""Host-only retirement fence for shared Docker egress generations."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager


def _outside_mounts(directory: str, mounts: tuple[str | None, ...]) -> None:
    for mount in mounts:
        if not mount:
            continue
        mounted = os.path.realpath(mount)
        if os.path.commonpath((directory, mounted)) == mounted:
            raise RuntimeError("egress runtime directory is inside container mounts")


def runtime_directory() -> str:
    """Use a private host directory outside every configured container mount."""
    root = os.path.realpath(os.getenv("ASSIST_THREADS_DIR", "/tmp/assist_threads"))
    default = os.path.join(
        os.path.dirname(root),
        ".assist-egress-runtime-" + hashlib.sha256(root.encode()).hexdigest()[:16],
    )
    requested = os.environ.get("ASSIST_EGRESS_RUNTIME_DIR", default)
    if not os.path.isabs(requested):
        raise RuntimeError("egress runtime directory must be absolute")
    directory = os.path.realpath(requested)
    _outside_mounts(directory, (
        root,
        os.environ.get("ASSIST_EGRESS_CLIENT_MAP_DIR"),
        os.environ.get("ASSIST_EGRESS_APPROVALS_DIR"),
    ))
    os.makedirs(directory, mode=0o700, exist_ok=True)
    stat = os.stat(directory)
    if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise RuntimeError("egress runtime directory has unsafe ownership or permissions")
    return directory


def assert_outside_mounts(*mounts: str | None) -> None:
    """Reject a ledger location inside a per-Run bind mount before Docker setup."""
    _outside_mounts(runtime_directory(), mounts)


def _read(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as stream:
            state = json.load(stream)
    except FileNotFoundError as error:
        raise RuntimeError("egress retirement ledger disappeared") from error
    if (not isinstance(state, dict) or set(state) != {"retired", "pending"}
            or not isinstance(state["retired"], dict)
            or not isinstance(state["pending"], dict)):
        raise RuntimeError("invalid egress retirement ledger")
    for mapping in state.values():
        if any(not isinstance(key, str) or not key or len(key) > 256
               or not isinstance(value, str) or not value or len(value) > 256
               for key, value in mapping.items()):
            raise RuntimeError("invalid egress retirement record")
    return state


def _write(path: str, state: dict) -> None:
    directory = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=".egress-state-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(directory, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _locked():
    directory = runtime_directory()
    lock_path = os.path.join(directory, ".lock")
    with open(lock_path, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            path = os.path.join(directory, "state.json")
            marker = os.path.join(directory, ".initialized")
            if not os.path.exists(path):
                if os.path.exists(marker):
                    raise RuntimeError("egress retirement ledger disappeared")
                _write(path, {"retired": {}, "pending": {}})
            if not os.path.exists(marker):
                fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                dir_fd = os.open(directory, os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            yield path, _read(path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def assert_admissible(kind: str, identity: str) -> None:
    with _locked() as (_, state):
        if f"{kind}:{identity}" in state["retired"]:
            raise RuntimeError(f"{kind} generation retired after uncertain Docker mutation")


def retirement(kind: str, identity: str) -> str | None:
    with _locked() as (_, state):
        return state["retired"].get(f"{kind}:{identity}")


def retire(kind: str, identity: str, operation: str) -> None:
    with _locked() as (path, state):
        state["retired"][f"{kind}:{identity}"] = operation
        _write(path, state)


def settle_retirement(kind: str, identity: str, operation: str) -> None:
    """Clear only a terminal, verified non-destructive mutation."""
    with _locked() as (path, state):
        key = f"{kind}:{identity}"
        if state["retired"].get(key) != operation:
            raise RuntimeError(f"{kind} retirement changed")
        del state["retired"][key]
        _write(path, state)


def begin_creation(kind: str, name: str, token: str) -> None:
    with _locked() as (path, state):
        key = f"{kind}:{name}"
        if key in state["pending"]:
            raise RuntimeError(f"{kind} creation outcome is uncertain")
        state["pending"][key] = token
        _write(path, state)


def pending_creation(kind: str, name: str) -> str | None:
    with _locked() as (_, state):
        return state["pending"].get(f"{kind}:{name}")


def settle_creation(kind: str, name: str, token: str) -> None:
    with _locked() as (path, state):
        key = f"{kind}:{name}"
        if state["pending"].get(key) != token:
            raise RuntimeError(f"{kind} creation token changed")
        del state["pending"][key]
        _write(path, state)
