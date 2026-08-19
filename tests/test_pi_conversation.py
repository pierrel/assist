from __future__ import annotations

import os
import json
import hashlib

import pytest

from assist.pi_conversation import (PiConversationError, PiConversationStore,
                                    PiHistorySummary, PiMessage)
from assist.pi_skills import PiLoadedSkill


def _render(run_id: str, body: str = "render rules") -> PiLoadedSkill:
    return PiLoadedSkill("render", body, hashlib.sha256(body.encode()).hexdigest(),
                         ("map_data",), run_id)


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
    user = {"version": 1, "ts": "now", "run_id": "run", "role": "user", "text": "x"}
    assistant = user | {"role": "assistant", "text": "y"}
    path.write_text(json.dumps(assistant) + "\n" + json.dumps(user) + "\n")
    with pytest.raises(PiConversationError):
        store.get_messages(tmp_path)
    path.write_text(json.dumps(user) + "\n" + json.dumps(user) + "\n")
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


def test_completed_run_retains_complete_skill_record_and_refreshes_it(tmp_path):
    store = PiConversationStore()
    store.append(tmp_path, "run-1", "user", "render this")
    store.complete(tmp_path, "run-1", "Done", (_render("run-1"),), ())

    context = store.context(tmp_path, max_messages=32)
    assert [skill.name for skill in context.loaded_skills] == ["render"]
    assert store.completed_reply(tmp_path, "run-1") == PiMessage("run-1", "assistant", "Done")
    assert store.get_messages(tmp_path) == [
        PiMessage("run-1", "user", "render this"), PiMessage("run-1", "assistant", "Done"),
    ]

    store.append(tmp_path, "run-2", "user", "refresh render")
    store.complete(tmp_path, "run-2", "Refreshed", (_render("run-2", "new rules"),), ())
    refreshed = store.context(tmp_path, max_messages=32).loaded_skills
    assert refreshed == (_render("run-2", "new rules"),)


def test_context_evicts_whole_skill_record_with_its_load_run(tmp_path):
    store = PiConversationStore()
    store.append(tmp_path, "run-1", "user", "first")
    store.complete(tmp_path, "run-1", "first done", (_render("run-1"),), ())
    for number in range(2, 18):
        run_id = f"run-{number}"
        store.append(tmp_path, run_id, "user", run_id)
        store.append(tmp_path, run_id, "assistant", "done")

    context = store.context(tmp_path, max_messages=32)
    assert len(context.messages) == 32
    assert context.loaded_skills == ()
    assert context.evicted_skills == (_render("run-1"),)


def test_context_excludes_the_current_run_before_selecting_complete_prior_runs(tmp_path):
    store = PiConversationStore()
    for number in range(1, 18):
        run_id = f"run-{number}"
        store.append(tmp_path, run_id, "user", f"user {number}")
        store.complete(tmp_path, run_id, "done", (_render(run_id),) if number == 1 else (), ())
    store.append(tmp_path, "run-18", "user", "current")

    context = store.context(tmp_path, max_messages=32, exclude_run_id="run-18")

    assert [(message.run_id, message.role) for message in context.messages] == [
        (f"run-{number}", role) for number in range(2, 18) for role in ("user", "assistant")]
    assert context.loaded_skills == ()
    assert context.evicted_skills == (_render("run-1"),)


def test_skill_record_must_immediately_follow_its_own_assistant_and_bad_names_fail_closed(tmp_path):
    store = PiConversationStore()
    path = tmp_path / "pi-conversation.jsonl"
    record = lambda run_id, role, **value: {  # noqa: E731
        "version": 1, "ts": "now", "run_id": run_id, "role": role, **value}
    render = _render("run-1")
    misplaced = record("run-1", "skill", name="render", body=render.body,
                       body_sha256=render.body_sha256, declared_tools=["map_data"])
    path.write_text("\n".join(json.dumps(value) for value in [
        record("run-1", "user", text="one"), record("run-1", "assistant", text="done"),
        record("run-2", "user", text="two"), record("run-2", "assistant", text="done"), misplaced,
    ]) + "\n")
    os.chmod(path, 0o600)
    with pytest.raises(PiConversationError):
        store.get_messages(tmp_path)

    invalid_timestamp = misplaced | {"ts": "x" * 129}
    path.write_text("\n".join(json.dumps(value) for value in [
        record("run-1", "user", text="one"), record("run-1", "assistant", text="done"), invalid_timestamp,
    ]) + "\n")
    with pytest.raises(PiConversationError):
        store.get_messages(tmp_path)

    malformed = misplaced | {"name": []}
    path.write_text("\n".join(json.dumps(value) for value in [
        record("run-1", "user", text="one"), record("run-1", "assistant", text="done"), malformed,
    ]) + "\n")
    with pytest.raises(PiConversationError):
        store.get_messages(tmp_path)


def test_compaction_preserves_existing_event_timestamps(tmp_path):
    store = PiConversationStore()
    store.append(tmp_path, "run-1", "user", "first")
    path = tmp_path / "pi-history" / "segment-0000000000000001.jsonl"
    original = json.loads(path.read_text().splitlines()[0])["ts"]
    store.complete(tmp_path, "run-1", "done", (), ())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["ts"] == original


def test_context_keeps_an_earlier_unpaired_user_visible(tmp_path):
    store = PiConversationStore()
    store.append(tmp_path, "run-1", "user", "please retry that")
    store.append(tmp_path, "run-2", "user", "next")
    store.append(tmp_path, "run-2", "assistant", "done")

    context = store.context(tmp_path, max_messages=32)

    assert context.messages == (
        PiMessage("run-1", "user", "please retry that"),
        PiMessage("run-2", "user", "next"), PiMessage("run-2", "assistant", "done"),
    )


def test_context_offers_only_a_complete_prefix_for_checkpointing(tmp_path):
    store = PiConversationStore()
    for number in range(1, 18):
        run_id = f"run-{number}"
        store.append(tmp_path, run_id, "user", f"user {number}")
        store.append(tmp_path, run_id, "assistant", f"assistant {number}")

    context = store.context(tmp_path, max_messages=32)

    assert [(message.run_id, message.role) for message in context.messages] == [
        (f"run-{number}", role) for number in range(2, 18) for role in ("user", "assistant")]
    assert context.compaction_candidate is not None
    assert context.compaction_candidate.last_run_id == "run-1"
    assert "Run run-1 user:\nuser 1" in context.compaction_candidate.body
    assert "run-2" not in context.compaction_candidate.body

    summary_body = "The first exchange established the starting point."
    summary = PiHistorySummary(summary_body, hashlib.sha256(summary_body.encode()).hexdigest(), "run-1")
    store.append(tmp_path, "run-18", "user", "next")
    store.complete(tmp_path, "run-18", "done", (), (), summary)
    after = store.context(tmp_path, max_messages=32)
    assert after.summary == summary
    assert after.compaction_candidate is not None
    assert after.compaction_candidate.last_run_id == "run-2"


def test_segmented_history_migrates_legacy_and_keeps_full_visible_transcript(tmp_path):
    legacy = tmp_path / "pi-conversation.jsonl"
    legacy.write_text("\n".join(json.dumps({
        "version": 1, "ts": "now", "run_id": f"run-{number}", "role": role,
        "text": f"{role} {number}",
    }) for number in range(1, 3) for role in ("user", "assistant")) + "\n")
    os.chmod(legacy, 0o600)
    store = PiConversationStore()

    assert len(store.get_messages(tmp_path)) == 4
    assert not legacy.exists()
    assert (tmp_path / "pi-history" / "manifest.json").exists()
    assert (tmp_path / "pi-history" / "state.json").exists()
    store.append(tmp_path, "run-3", "user", "x" * (128 * 1024))
    store.append(tmp_path, "run-3", "assistant", "done")
    store.append(tmp_path, "run-4", "user", "y" * (128 * 1024))
    assert len(list((tmp_path / "pi-history").glob("segment-*.jsonl"))) >= 2
    assert len(store.get_messages(tmp_path)) == 7


def test_receipt_recovers_assistant_and_projection_after_state_write_failure(tmp_path, monkeypatch):
    store = PiConversationStore()
    store.append(tmp_path, "run-1", "user", "first")
    body = "first checkpoint"
    summary = PiHistorySummary(body, hashlib.sha256(body.encode()).hexdigest(), "run-1")
    original = PiConversationStore._atomic_write

    def fail_state(path, raw, subject):
        if path.name == "state.json":
            raise OSError("disk full")
        return original(path, raw, subject)

    with monkeypatch.context() as patch:
        patch.setattr(PiConversationStore, "_atomic_write", staticmethod(fail_state))
        with pytest.raises(OSError):
            store.complete(tmp_path, "run-1", "done", (), (), summary)

    recovered = PiConversationStore()
    assert recovered.completed_reply(tmp_path, "run-1") == PiMessage("run-1", "assistant", "done")
    assert recovered.context(tmp_path, max_messages=32).summary == summary
    assert not (tmp_path / "pi-history" / "pending.json").exists()


def test_invalid_persisted_summary_boundary_fails_closed(tmp_path):
    store = PiConversationStore()
    store.append(tmp_path, "run-1", "user", "first")
    store.append(tmp_path, "run-1", "assistant", "done")
    body = "bad boundary"
    state = {"version": 1, "summary": {
        "body": body, "body_sha256": hashlib.sha256(body.encode()).hexdigest(), "last_run_id": "absent",
    }, "skills": []}
    path = tmp_path / "pi-history" / "state.json"
    path.write_text(json.dumps(state))
    os.chmod(path, 0o600)

    with pytest.raises(PiConversationError, match="summary boundary"):
        store.context(tmp_path, max_messages=32)


def test_incomplete_migration_restarts_from_the_legacy_transcript(tmp_path):
    legacy = tmp_path / "pi-conversation.jsonl"
    legacy.write_text(json.dumps({
        "version": 1, "ts": "now", "run_id": "run-1", "role": "user", "text": "hello",
    }) + "\n")
    os.chmod(legacy, 0o600)
    (tmp_path / "pi-history").mkdir(mode=0o700)

    assert PiConversationStore().get_messages(tmp_path) == [PiMessage("run-1", "user", "hello")]
    assert not legacy.exists()


def test_unpaired_omitted_event_prevents_a_checkpoint_gap(tmp_path):
    store = PiConversationStore()
    for number in range(1, 3):
        store.append(tmp_path, f"run-{number}", "user", "user")
        store.append(tmp_path, f"run-{number}", "assistant", "assistant")
    store.append(tmp_path, "orphan", "user", "retry me")
    for number in range(3, 21):
        store.append(tmp_path, f"run-{number}", "user", "user")
        store.append(tmp_path, f"run-{number}", "assistant", "assistant")

    context = store.context(tmp_path, max_messages=32)

    assert context.compaction_candidate is None
