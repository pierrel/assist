"""Durable, redacted Pi activity trace resource tests."""
from __future__ import annotations

import json

import pytest

from assist.pi_trace import PiTraceError, PiTraceEvent, PiTraceRecorder, PiTraceStore


def test_recorder_persists_a_closed_redacted_operation(tmp_path) -> None:
    recorder = PiTraceRecorder(PiTraceStore(), tmp_path, "run-1")

    operation = recorder.start("tool", "read")
    recorder.settle(operation, True)

    assert PiTraceStore().get_events(tmp_path) == [
        PiTraceEvent("run-1", 1, 1, "tool", "read", "started"),
        PiTraceEvent("run-1", 2, 1, "tool", "read", "completed"),
    ]
    raw = (tmp_path / "pi-trace.jsonl").read_text()
    assert "/workspace" not in raw
    assert "command" not in raw


def test_store_rejects_an_invalid_event_before_it_persists(tmp_path) -> None:
    store = PiTraceStore()

    with pytest.raises(PiTraceError):
        store.append(tmp_path, PiTraceEvent("run-1", 1, 1, "tool", "not a label", "started"))  # type: ignore[arg-type]

    assert not (tmp_path / "pi-trace.jsonl").exists()


def test_store_rejects_an_invalid_lifecycle_before_it_persists(tmp_path) -> None:
    store = PiTraceStore()
    store.append(tmp_path, PiTraceEvent("run-1", 1, 1, "tool", "read", "started"))

    with pytest.raises(PiTraceError, match="outcome"):
        store.append(tmp_path, PiTraceEvent("run-1", 2, 2, "tool", "read", "completed"))

    assert PiTraceStore().get_events(tmp_path) == [
        PiTraceEvent("run-1", 1, 1, "tool", "read", "started"),
    ]


def test_store_rejects_a_malformed_or_unsettled_trace(tmp_path) -> None:
    path = tmp_path / "pi-trace.jsonl"
    path.write_text(json.dumps({"version": 1}) + "\n")
    path.chmod(0o600)

    with pytest.raises(PiTraceError, match="malformed"):
        PiTraceStore().get_events(tmp_path)


def test_store_rejects_a_truncated_but_otherwise_valid_record(tmp_path) -> None:
    (tmp_path / "pi-trace.jsonl").write_text(
        '{"version":1,"ts":"","run_id":"run-1","sequence":1,'
        '"operation":1,"kind":"tool","name":"read","outcome":"started"}')
    (tmp_path / "pi-trace.jsonl").chmod(0o600)

    with pytest.raises(PiTraceError, match="malformed"):
        PiTraceStore().get_events(tmp_path)


def test_recorder_stops_recording_after_a_write_failure(tmp_path, monkeypatch) -> None:
    store = PiTraceStore()
    recorder = PiTraceRecorder(store, tmp_path, "run-1")
    original = store.append
    calls = 0

    def fail_second_write(directory, event):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PiTraceError("disk unavailable")
        return original(directory, event)

    monkeypatch.setattr(store, "append", fail_second_write)
    operation = recorder.start("model", "model request")
    recorder.settle(operation, True)
    recorder.finish_unsettled()

    assert PiTraceStore().get_events(tmp_path) == [
        PiTraceEvent("run-1", 1, 1, "model", "model request", "started"),
    ]
