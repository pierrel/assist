"""Tests for ``assist.eviction_saver.EvictionSaver``.

Layer 3 of the threads.db growth plan
(docs/2026-05-04-threads-db-layer-3-tool-result-eviction.org).

Real sqlite + real upstream SqliteSaver behind the wrapper.  The
on-disk eviction directory is a per-test ``tempfile.TemporaryDirectory``
so the round-trip (write → evict → checkpoint → read → rehydrate) is
exercised end-to-end without mocks.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
from unittest import TestCase
from unittest.mock import patch

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.base import empty_checkpoint

from assist.eviction_saver import (
    DEFAULT_EVICT_THRESHOLD_KB,
    EVICT_HASH_KEY,
    EVICT_PATH_KEY,
    EVICT_SIZE_KEY,
    EvictionFileMissingError,
    EvictionSaver,
    _resolve_evict_threshold_kb,
)


def _config(thread_id: str, ns: str = "") -> dict:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns}}


def _ckpt(seq: int, *, messages=None, files=None) -> dict:
    """Build a Checkpoint with deterministic id and the given channel values."""
    c = empty_checkpoint()
    c["id"] = f"{seq:08d}"
    cv = c.get("channel_values") or {}
    if messages is not None:
        cv["messages"] = messages
    if files is not None:
        cv["files"] = files
    c["channel_values"] = cv
    return c


def _tool_msg(content: str, tool_call_id: str = "call_x") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tool_call_id, name="t")


def _new_saver(
    *, threshold_kb: int = 20, root: str | None = None
) -> tuple[EvictionSaver, str]:
    """Return (saver, eviction_root).  Caller is responsible for cleaning
    up ``eviction_root`` (use ``TemporaryDirectory`` in the test).
    """
    if root is None:
        root = tempfile.mkdtemp()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    return (
        EvictionSaver(conn, evict_threshold_kb=threshold_kb, eviction_root=root),
        root,
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip(TestCase):
    """Write a large ToolMessage; verify it goes to disk and rehydrates."""

    def test_round_trip_messages(self):
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            big = "x" * 5000  # 5 KB > 1 KB threshold
            tm = _tool_msg(big)
            saver.put(_config("t1"), _ckpt(0, messages=[tm]), {}, {})

            # Eviction file written under <root>/<tid>/large_tool_results/
            evict_dir = os.path.join(root, "t1", "large_tool_results")
            self.assertTrue(os.path.isdir(evict_dir))
            files = os.listdir(evict_dir)
            self.assertEqual(len(files), 1)
            with open(os.path.join(evict_dir, files[0]), "rb") as f:
                self.assertEqual(f.read(), big.encode("utf-8"))

            # Checkpoint in DB has stub content (does NOT contain the
            # original 5 KB).  Smoking gun is byte count.
            cur = saver.conn.cursor()
            cur.execute("SELECT checkpoint FROM checkpoints WHERE thread_id='t1'")
            blob = cur.fetchone()[0]
            self.assertNotIn(big.encode("utf-8"), bytes(blob))

            # get_tuple rehydrates back to the original content.
            tup = saver.get_tuple(_config("t1"))
            self.assertIsNotNone(tup)
            messages = tup.checkpoint["channel_values"]["messages"]
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].content, big)
            # Sentinel keys are scrubbed from rehydrated messages.
            self.assertNotIn(EVICT_PATH_KEY, messages[0].additional_kwargs)

    def test_round_trip_files_channel(self):
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            big = "y" * 5000
            files = {
                "/large_tool_results/abc": {
                    "content": big, "encoding": "utf-8",
                },
            }
            saver.put(_config("t1"), _ckpt(0, files=files), {}, {})

            tup = saver.get_tuple(_config("t1"))
            restored = tup.checkpoint["channel_values"]["files"]
            self.assertEqual(restored["/large_tool_results/abc"]["content"], big)

    def test_unevicted_messages_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=20, root=root)
            small = "hello"
            tm = _tool_msg(small)
            saver.put(_config("t1"), _ckpt(0, messages=[tm]), {}, {})
            self.assertFalse(
                os.path.isdir(os.path.join(root, "t1", "large_tool_results"))
            )
            tup = saver.get_tuple(_config("t1"))
            self.assertEqual(
                tup.checkpoint["channel_values"]["messages"][0].content, small
            )


# ---------------------------------------------------------------------------
# Threshold semantics
# ---------------------------------------------------------------------------


class TestThresholdBoundary(TestCase):
    """Exactly-at-threshold is NOT evicted; threshold+1 is."""

    def test_exact_boundary_not_evicted(self):
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            content = "x" * 1024  # exactly 1 KB
            tm = _tool_msg(content)
            saver.put(_config("t1"), _ckpt(0, messages=[tm]), {}, {})
            self.assertFalse(
                os.path.isdir(os.path.join(root, "t1", "large_tool_results"))
            )

    def test_threshold_plus_one_evicted(self):
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            content = "x" * 1025  # 1 KB + 1 byte
            tm = _tool_msg(content)
            saver.put(_config("t1"), _ckpt(0, messages=[tm]), {}, {})
            evict_dir = os.path.join(root, "t1", "large_tool_results")
            self.assertTrue(os.path.isdir(evict_dir))
            self.assertEqual(len(os.listdir(evict_dir)), 1)

    def test_files_channel_only_under_large_tool_results_prefix(self):
        """Only ``/large_tool_results/`` entries are eligible for eviction."""
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            big = "z" * 5000
            files = {
                "/large_tool_results/should_evict": {
                    "content": big, "encoding": "utf-8",
                },
                "/other_path/should_keep": {
                    "content": big, "encoding": "utf-8",
                },
            }
            saver.put(_config("t1"), _ckpt(0, files=files), {}, {})
            evict_dir = os.path.join(root, "t1", "large_tool_results")
            # Only one file evicted (the /large_tool_results/ one).
            self.assertEqual(len(os.listdir(evict_dir)), 1)


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


class TestKillSwitch(TestCase):
    def test_threshold_zero_disables_eviction(self):
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=0, root=root)
            big = "x" * 100_000
            tm = _tool_msg(big)
            saver.put(_config("t1"), _ckpt(0, messages=[tm]), {}, {})
            # No eviction dir.
            self.assertFalse(os.path.isdir(os.path.join(root, "t1")))
            # Content survives in DB unchanged.
            tup = saver.get_tuple(_config("t1"))
            self.assertEqual(
                tup.checkpoint["channel_values"]["messages"][0].content, big
            )

    def test_no_eviction_root_disables_eviction(self):
        """If eviction_root is None but threshold > 0, eviction still
        gets disabled (no place to write) and pass-through works."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        saver = EvictionSaver(conn, evict_threshold_kb=1, eviction_root=None)
        self.assertEqual(saver.evict_threshold_kb, 0)
        big = "x" * 5000
        tm = _tool_msg(big)
        saver.put(_config("t1"), _ckpt(0, messages=[tm]), {}, {})
        tup = saver.get_tuple(_config("t1"))
        self.assertEqual(
            tup.checkpoint["channel_values"]["messages"][0].content, big
        )


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


class TestContentHashDedup(TestCase):
    def test_same_content_two_threads_one_file(self):
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            big = "shared content " * 1000  # > 1 KB, identical bytes
            tm_a = _tool_msg(big, tool_call_id="call_a")
            tm_b = _tool_msg(big, tool_call_id="call_b")
            saver.put(_config("ta"), _ckpt(0, messages=[tm_a]), {}, {})
            saver.put(_config("tb"), _ckpt(0, messages=[tm_b]), {}, {})

            # Each thread has its own dir; files have the same hash.
            ta_dir = os.path.join(root, "ta", "large_tool_results")
            tb_dir = os.path.join(root, "tb", "large_tool_results")
            self.assertEqual(os.listdir(ta_dir), os.listdir(tb_dir))

            # Both rehydrate correctly.
            for tid in ("ta", "tb"):
                tup = saver.get_tuple(_config(tid))
                self.assertEqual(
                    tup.checkpoint["channel_values"]["messages"][0].content, big
                )

    def test_two_puts_same_thread_same_content_dedup(self):
        """Putting the same content twice in one thread reuses the file
        (O_EXCL hits, treated as success)."""
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            big = "dup " * 1000
            for seq in range(2):
                saver.put(
                    _config("t1"),
                    _ckpt(seq, messages=[_tool_msg(big)]),
                    {}, {},
                )
            evict_dir = os.path.join(root, "t1", "large_tool_results")
            self.assertEqual(len(os.listdir(evict_dir)), 1)


# ---------------------------------------------------------------------------
# Idempotency: a checkpoint with already-evicted stubs is a no-op.
# ---------------------------------------------------------------------------


class TestIdempotency(TestCase):
    def test_already_evicted_message_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            big = "x" * 5000
            saver.put(
                _config("t1"),
                _ckpt(0, messages=[_tool_msg(big)]),
                {}, {},
            )
            evict_dir = os.path.join(root, "t1", "large_tool_results")
            files_before = sorted(os.listdir(evict_dir))

            # Read back and put again — the rehydrated message goes
            # back to full content, gets re-evicted to the same hash.
            tup = saver.get_tuple(_config("t1"))
            saver.put(
                _config("t1"),
                _ckpt(1, messages=tup.checkpoint["channel_values"]["messages"]),
                {}, {},
            )
            files_after = sorted(os.listdir(evict_dir))
            self.assertEqual(files_before, files_after)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestRehydrationFailure(TestCase):
    def test_missing_eviction_file_raises(self):
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            big = "x" * 5000
            saver.put(
                _config("t1"),
                _ckpt(0, messages=[_tool_msg(big)]),
                {}, {},
            )
            evict_dir = os.path.join(root, "t1", "large_tool_results")
            for f in os.listdir(evict_dir):
                os.unlink(os.path.join(evict_dir, f))
            with self.assertRaises(EvictionFileMissingError):
                saver.get_tuple(_config("t1"))


class TestDiskWriteFallback(TestCase):
    """Disk failure during put() falls back to writing un-evicted checkpoint."""

    def test_makedirs_failure_falls_back(self):
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            big = "x" * 5000
            tm = _tool_msg(big)
            with patch("os.makedirs", side_effect=OSError("ENOSPC")):
                saver.put(_config("t1"), _ckpt(0, messages=[tm]), {}, {})
            # Eviction dir was never created.
            self.assertFalse(
                os.path.isdir(os.path.join(root, "t1", "large_tool_results"))
            )
            # But the checkpoint did get persisted (un-evicted).
            tup = saver.get_tuple(_config("t1"))
            self.assertEqual(
                tup.checkpoint["channel_values"]["messages"][0].content, big
            )


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


class TestNonToolMessagesUnchanged(TestCase):
    def test_human_message_with_huge_content_not_evicted(self):
        """Eviction targets ToolMessage only; HumanMessage / AIMessage
        with large content stays in the DB.  The threshold protects
        against tool-result blowup, not against user pasting a novel."""
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            big = "x" * 5000
            hm = HumanMessage(content=big)
            saver.put(_config("t1"), _ckpt(0, messages=[hm]), {}, {})
            self.assertFalse(
                os.path.isdir(os.path.join(root, "t1", "large_tool_results"))
            )
            tup = saver.get_tuple(_config("t1"))
            self.assertEqual(
                tup.checkpoint["channel_values"]["messages"][0].content, big
            )


class TestListWrapsRehydration(TestCase):
    def test_list_returns_rehydrated_checkpoints(self):
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            big = "x" * 5000
            for seq in range(3):
                saver.put(
                    _config("t1"),
                    _ckpt(seq, messages=[_tool_msg(big + str(seq))]),
                    {}, {},
                )
            tuples = list(saver.list(_config("t1")))
            self.assertEqual(len(tuples), 3)
            for tup in tuples:
                content = tup.checkpoint["channel_values"]["messages"][0].content
                self.assertTrue(content.startswith("x"))
                self.assertGreater(len(content), 4000)


class TestAputStillRaises(TestCase):
    """The async path is intentionally not enabled by Layer 3."""

    def test_aput_raises_not_implemented(self):
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(root=root)
            with self.assertRaises(NotImplementedError):
                asyncio.run(saver.aput(_config("t"), _ckpt(0), {}, {}))


# ---------------------------------------------------------------------------
# Env var parser
# ---------------------------------------------------------------------------


class TestResolveEvictThresholdKb(TestCase):
    def test_unset_uses_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ASSIST_EVICT_THRESHOLD_KB", None)
            self.assertEqual(
                _resolve_evict_threshold_kb(), DEFAULT_EVICT_THRESHOLD_KB
            )

    def test_zero_disables(self):
        with patch.dict(os.environ, {"ASSIST_EVICT_THRESHOLD_KB": "0"}):
            self.assertEqual(_resolve_evict_threshold_kb(), 0)

    def test_disabled_keyword(self):
        for v in ("disabled", "off", "false", ""):
            with patch.dict(os.environ, {"ASSIST_EVICT_THRESHOLD_KB": v}):
                self.assertEqual(_resolve_evict_threshold_kb(), 0)

    def test_positive_int(self):
        with patch.dict(os.environ, {"ASSIST_EVICT_THRESHOLD_KB": "50"}):
            self.assertEqual(_resolve_evict_threshold_kb(), 50)

    def test_negative_int_falls_back_to_default(self):
        with patch.dict(os.environ, {"ASSIST_EVICT_THRESHOLD_KB": "-5"}):
            self.assertEqual(
                _resolve_evict_threshold_kb(), DEFAULT_EVICT_THRESHOLD_KB
            )

    def test_garbage_falls_back_to_default(self):
        with patch.dict(os.environ, {"ASSIST_EVICT_THRESHOLD_KB": "abc"}):
            self.assertEqual(
                _resolve_evict_threshold_kb(), DEFAULT_EVICT_THRESHOLD_KB
            )


# ---------------------------------------------------------------------------
# Compatibility with Layer 0's hard_delete (smoke).  Eviction files live
# inside the per-thread dir so an rmtree on the dir takes them with it.
# ---------------------------------------------------------------------------


class TestEvictionFilesUnderThreadDir(TestCase):
    def test_eviction_path_is_under_root(self):
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            saver.put(
                _config("tid_xyz"),
                _ckpt(0, messages=[_tool_msg("x" * 5000)]),
                {}, {},
            )
            evict_dir = os.path.join(root, "tid_xyz", "large_tool_results")
            self.assertTrue(os.path.isdir(evict_dir))
            # Every eviction file is under <root>/<tid>/.
            for f in os.listdir(evict_dir):
                full = os.path.join(evict_dir, f)
                self.assertTrue(
                    os.path.commonpath([full, os.path.join(root, "tid_xyz")])
                    == os.path.join(root, "tid_xyz")
                )


# ---------------------------------------------------------------------------
# Stub-message metadata: stubs carry size + hash for forensics.
# ---------------------------------------------------------------------------


class TestStubMetadata(TestCase):
    def test_stub_records_size_and_hash(self):
        """Inspect the in-DB checkpoint (not rehydrated) to confirm the
        stub message carries the size + hash that Layer 0 / VACUUM
        observability can later inspect."""
        with tempfile.TemporaryDirectory() as root:
            saver, _ = _new_saver(threshold_kb=1, root=root)
            big = "x" * 5000
            saver.put(
                _config("t1"),
                _ckpt(0, messages=[_tool_msg(big)]),
                {}, {},
            )
            # Use the upstream get_tuple via a parallel SqliteSaver so
            # we see the un-rehydrated checkpoint.  Any saver pointed
            # at the same conn works.
            from langgraph.checkpoint.sqlite import SqliteSaver
            raw = SqliteSaver(saver.conn).get_tuple(_config("t1"))
            stub = raw.checkpoint["channel_values"]["messages"][0]
            ak = stub.additional_kwargs
            self.assertIn(EVICT_PATH_KEY, ak)
            self.assertEqual(ak[EVICT_SIZE_KEY], 5000)
            self.assertEqual(len(ak[EVICT_HASH_KEY]), 16)
            self.assertNotIn("x" * 5000, str(stub.content))
