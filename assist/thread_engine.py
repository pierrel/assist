"""Immutable engine identity for visible Assist web threads.

The engine marker belongs to the thread directory rather than Run metadata: a
client must not be able to change a thread from Deep Agents to Pi by crafting a
later request.  A missing marker denotes a legacy Deep Agents thread; malformed
markers never select Pi.
"""
from __future__ import annotations

import json
import os
import errno
import stat
from dataclasses import dataclass
from typing import Literal


EngineName = Literal["deepagents", "pi"]
ENGINE_MARKER = "engine.json"
_MARKER_VERSION = 1
_VISIBLE_ORIGIN = "manual-web"
_MAX_MARKER_BYTES = 1024


class ThreadEngineError(ValueError):
    """A thread engine marker is missing where required or is not trustworthy."""


@dataclass(frozen=True)
class ThreadEngine:
    """The immutable engine and origin selected for one visible thread."""

    name: EngineName
    origin: Literal["manual-web", "legacy"]


LEGACY_DEEP = ThreadEngine("deepagents", "legacy")


def _marker_path(thread_dir: str) -> str:
    return os.path.join(thread_dir, ENGINE_MARKER)


def _validated(value: object) -> ThreadEngine:
    if not isinstance(value, dict) or set(value) != {"version", "engine", "origin"}:
        raise ThreadEngineError("thread engine marker has an invalid shape")
    if value["version"] != _MARKER_VERSION:
        raise ThreadEngineError("thread engine marker version is unsupported")
    engine = value["engine"]
    if not isinstance(engine, str) or engine not in {"deepagents", "pi"}:
        raise ThreadEngineError("thread engine marker has an unknown engine")
    origin = value["origin"]
    if not isinstance(origin, str) or origin != _VISIBLE_ORIGIN:
        raise ThreadEngineError("thread engine marker has an invalid origin")
    return ThreadEngine(engine, origin)


def read_thread_engine(thread_dir: str) -> ThreadEngine:
    """Read a thread identity; absent marker is the safe legacy Deep default."""
    path = _marker_path(thread_dir)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return LEGACY_DEEP
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ThreadEngineError("thread engine marker is not a regular file") from error
        raise ThreadEngineError("thread engine marker is unreadable") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ThreadEngineError("thread engine marker is not a regular file")
        if metadata.st_size > _MAX_MARKER_BYTES:
            raise ThreadEngineError("thread engine marker exceeds its size bound")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            return _validated(json.load(stream))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ThreadEngineError("thread engine marker is unreadable") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_new_thread_engine(thread_dir: str, engine: EngineName) -> ThreadEngine:
    """Publish a new manual-web identity once, refusing overwrite or mutation."""
    if not isinstance(engine, str) or engine not in {"deepagents", "pi"}:
        raise ThreadEngineError(f"unknown thread engine: {engine!r}")
    identity = ThreadEngine(engine, _VISIBLE_ORIGIN)
    payload = json.dumps(
        {"version": _MARKER_VERSION, "engine": identity.name, "origin": identity.origin},
        separators=(",", ":"),
    ).encode("utf-8")
    path = _marker_path(thread_dir)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ThreadEngineError("thread engine identity already exists") from error
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("thread engine marker write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except BaseException:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(descriptor)
    directory = os.open(thread_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    observed = read_thread_engine(thread_dir)
    if observed != identity:
        raise ThreadEngineError("thread engine identity changed while publishing")
    return identity
