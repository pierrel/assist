from __future__ import annotations

import os
import json

import pytest

from assist.pi_conversation import PiConversationError, PiConversationStore, PiMessage


def test_host_transcript_is_run_keyed_and_idempotent(tmp_path):
    store = PiConversationStore()
    assert store.append(tmp_path, "run-1", "user", "hello").text == "hello"
    assert store.append(tmp_path, "run-1", "user", "hello").text == "hello"
    store.append(tmp_path, "run-1", "assistant", "hi")
    assert store.get_messages(tmp_path) == [
        PiMessage("run-1", "user", "hello"),
        PiMessage("run-1", "assistant", "hi"),
    ]
    with pytest.raises(PiConversationError):
        store.append(tmp_path, "run-1", "user", "different")
    with pytest.raises(PiConversationError):
        store.append(tmp_path, "run-2", "assistant", "cannot lead")


def test_malformed_or_unsafe_transcript_fails_closed(tmp_path):
    store = PiConversationStore()
    path = tmp_path / "pi-conversation.jsonl"
    path.write_text("not json\n")
    os.chmod(path, 0o600)
    with pytest.raises(PiConversationError):
        store.get_messages(tmp_path)
    path.unlink()
    target = tmp_path / "target.jsonl"
    target.write_text("")
    path.symlink_to(target)
    with pytest.raises(PiConversationError):
        store.get_messages(tmp_path)


def test_non_string_role_and_failed_write_leave_the_transcript_usable(tmp_path, monkeypatch):
    store = PiConversationStore()
    path = tmp_path / "pi-conversation.jsonl"
    path.write_text(
        '{"version":1,"ts":"now","run_id":"run","role":[],"text":"x"}\n')
    os.chmod(path, 0o600)
    with pytest.raises(PiConversationError):
        store.get_messages(tmp_path)
    path.unlink()
    store.append(tmp_path, "run-1", "user", "hello")
    original_write = os.write
    writes = 0

    def short_then_fail(descriptor, data):
        nonlocal writes
        if writes == 0:
            writes += 1
            return original_write(descriptor, data[:1])
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr("assist.pi_conversation.os.write", short_then_fail)
        with pytest.raises(PiConversationError):
            store.append(tmp_path, "run-1", "assistant", "hi")
    assert store.get_messages(tmp_path) == [PiMessage("run-1", "user", "hello")]
    store.append(tmp_path, "run-1", "assistant", "hi")
    before = store.get_messages(tmp_path)

    def fail_chmod(*_):
        raise OSError("permission denied")

    with monkeypatch.context() as patch:
        patch.setattr("assist.pi_conversation.os.fchmod", fail_chmod)
        with pytest.raises(PiConversationError):
            store.append(tmp_path, "run-2", "user", "next")
    assert store.get_messages(tmp_path) == before
    assert not list(tmp_path.glob(".pi-conversation-*"))


def test_forged_oversized_duplicate_or_out_of_order_records_are_rejected(tmp_path):
    store = PiConversationStore()
    path = tmp_path / "pi-conversation.jsonl"
    oversized = {"version": 1, "ts": "now", "run_id": "run", "role": "user",
                 "text": "x" * (128 * 1024 + 1)}
    path.write_text(json.dumps(oversized) + "\n")
    os.chmod(path, 0o600)
    with pytest.raises(PiConversationError):
        store.get_messages(tmp_path)
    user = {"version": 1, "ts": "now", "run_id": "run", "role": "user", "text": "x"}
    assistant = user | {"role": "assistant", "text": "y"}
    path.write_text(json.dumps(assistant) + "\n" + json.dumps(user) + "\n")
    with pytest.raises(PiConversationError):
        store.get_messages(tmp_path)
    path.write_text(json.dumps(user) + "\n" + json.dumps(user) + "\n")
    with pytest.raises(PiConversationError):
        store.get_messages(tmp_path)
