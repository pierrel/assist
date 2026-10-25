"""Synthetic history and receipt contracts, with real SQLite durability."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import pytest

from assist.events.sms_archive import SmsArchive
from assist.events.sms_tools import sms_archive_tools


@pytest.fixture
def archive(tmp_path):
    return SmsArchive(tmp_path / "private-sms", thread_root=tmp_path / "threads")


def test_private_non_git_location(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: elsewhere")
    with pytest.raises(ValueError, match="outside every Git"):
        SmsArchive(repo / "nested" / "archive", thread_root=tmp_path / "threads")
    bare = tmp_path / "bare.git"
    bare.mkdir()
    (bare / "HEAD").write_text("ref: refs/heads/main")
    (bare / "objects").mkdir()
    with pytest.raises(ValueError, match="outside every Git"):
        SmsArchive(bare / "archive", thread_root=tmp_path / "threads")
    with pytest.raises(ValueError, match="outside the thread"):
        SmsArchive(tmp_path / "threads" / "archive", thread_root=tmp_path / "threads")


def test_archive_refuses_linked_database_and_symlink_directory(tmp_path):
    outside = tmp_path / "target"
    outside.mkdir(mode=0o700)
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        SmsArchive(link, thread_root=tmp_path / "threads")
    (outside / "messages.sqlite3").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="private regular"):
        SmsArchive(outside, thread_root=tmp_path / "threads")


def test_independent_identical_messages_and_exact_content_survive_reopen(archive, tmp_path):
    body = 'Same "quote"\n雪 and apostrophe\'s'
    for ident in ("receipt_one", "receipt_two"):
        _, created = archive.record(ident, "incoming", "+15550001001", body, "received")
        assert created
    reopened = SmsArchive(archive.directory, thread_root=tmp_path / "threads")
    assert len(reopened.search(text='"quote"')) == 2
    assert reopened.read("receipt_one")["text"] == body
    assert os.stat(archive.path).st_mode & 0o777 == 0o600
    assert os.stat(archive.directory).st_mode & 0o777 == 0o700


def test_conflicting_id_and_reordered_outcomes(archive):
    archive.record("attempt", "outgoing", "+15550001001", "message", "unknown")
    archive.record("attempt", "outgoing", "+15550001001", "message", "sent")
    _, created = archive.record("attempt", "outgoing", "+15550001001", "message", "unknown")
    assert not created and archive.read("attempt")["status"] == "sent"
    for direction, number, text, status in (
        ("outgoing", "+15550001002", "message", "sent"),
        ("outgoing", "+15550001001", "changed", "sent"),
        ("incoming", "+15550001001", "message", "received"),
        ("outgoing", "+15550001001", "message", "not_sent"),
    ):
        with pytest.raises(ValueError, match="conflicts"):
            archive.record("attempt", direction, number, text, status)


def test_concurrent_attempt_claim_only_one_creator(archive):
    def claim(_):
        return archive.record("attempt", "outgoing", "+15550001001", "message", "unknown")[1]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(claim, range(12))) == 1


def test_incoming_handling_identity_durable_and_immutable(archive, tmp_path):
    row, _ = archive.record("receipt", "incoming", "+15550001001", "hi", "received",
                            handling=("thread1", "handle hi"))
    reopened = SmsArchive(archive.directory, thread_root=tmp_path / "threads")
    duplicate, _ = reopened.record("receipt", "incoming", "+15550001001", "hi", "received",
                                    handling=("different-thread", "different prompt"))
    assert (duplicate["thread_id"], duplicate["prompt"], duplicate["run_id"]) == (
        "thread1", "handle hi", row["run_id"])
    assert not duplicate["admitted"]
    reopened.mark_admitted("receipt")
    admitted, _ = reopened.record("receipt", "incoming", "+15550001001", "hi", "received")
    assert admitted["admitted"]


def test_bounded_literal_search_and_read_tools(archive, monkeypatch):
    for i in range(25):
        archive.record(f"receipt_{i}", "incoming", "+15550001001", "% literal_ " + "x" * 500, "received")
    assert len(archive.search(text="% literal_")) == 20
    assert all(len(row["text"]) == 240 for row in archive.search())
    assert archive.search(text="' OR 1=1 --") == []
    with pytest.raises(ValueError):
        archive.search(limit=21)
    with pytest.raises(ValueError):
        archive.read("../secret")
    monkeypatch.setenv("ASSIST_SMS_ARCHIVE_DIR", str(archive.directory))
    monkeypatch.setenv("ASSIST_THREADS_DIR", str(archive.directory.parent / "threads"))
    search, read = sms_archive_tools()
    assert "Untrusted" in search(text="% literal_")
    assert "x" * 500 in read("receipt_0")
    assert "invalid" in read("../secret")


def test_nonexistent_dotdot_prefix_cannot_hide_git_ancestor(tmp_path):
    repository = tmp_path / "repo"
    (repository / ".git" / "objects").mkdir(parents=True)
    (repository / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(ValueError):
        SmsArchive(tmp_path / "missing" / ".." / "repo" / "private", thread_root=tmp_path / "threads")
    assert not (repository / "private").exists()


def test_incoming_sender_is_exact_untrusted_address_not_outbound_dial_validation(archive):
    for index, number in enumerate(('T-Mobile', 'CITY BANK', '通知', '')):
        row, _ = archive.record('sender' + str(index), 'incoming', number, '', 'received')
        assert row['number'] == number and row['text'] == ''
    with pytest.raises(ValueError): archive.record('bad_out', 'outgoing', 'T-Mobile', 'hello', 'unknown')


def test_long_multipart_text_is_not_lost_at_small_message_threshold(archive):
    text='multipart ' * 4000
    row,_=archive.record('multipart','incoming','CITY BANK',text,'received')
    assert archive.read('multipart')['text']==text
