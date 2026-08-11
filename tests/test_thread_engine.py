from __future__ import annotations

import json
import os

import pytest

from assist.thread_engine import (
    ENGINE_MARKER,
    LEGACY_DEEP,
    ThreadEngine,
    ThreadEngineError,
    read_thread_engine,
    write_new_thread_engine,
)


def test_missing_marker_is_legacy_deep(tmp_path):
    assert read_thread_engine(str(tmp_path)) == LEGACY_DEEP


def test_new_marker_is_manual_web_and_immutable(tmp_path):
    assert write_new_thread_engine(str(tmp_path), "pi") == ThreadEngine("pi", "manual-web")
    assert read_thread_engine(str(tmp_path)) == ThreadEngine("pi", "manual-web")

    with pytest.raises(ThreadEngineError, match="already exists"):
        write_new_thread_engine(str(tmp_path), "deepagents")


def test_writer_rejects_non_string_engine(tmp_path):
    with pytest.raises(ThreadEngineError, match="unknown"):
        write_new_thread_engine(str(tmp_path), [])


@pytest.mark.parametrize("payload", [
    {},
    {"version": 2, "engine": "pi", "origin": "manual-web"},
    {"version": 1, "engine": "unknown", "origin": "manual-web"},
    {"version": 1, "engine": [], "origin": "manual-web"},
    {"version": 1, "engine": "pi", "origin": "phone"},
])
def test_malformed_marker_fails_closed(tmp_path, payload):
    (tmp_path / ENGINE_MARKER).write_text(json.dumps(payload))

    with pytest.raises(ThreadEngineError):
        read_thread_engine(str(tmp_path))


def test_symlink_marker_fails_closed(tmp_path):
    target = tmp_path / "target"
    target.write_text('{}')
    os.symlink(target, tmp_path / ENGINE_MARKER)

    with pytest.raises(ThreadEngineError, match="regular"):
        read_thread_engine(str(tmp_path))
