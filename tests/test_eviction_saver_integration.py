"""Integration test: EvictionSaver wired into a real ThreadManager.

Layer 3 of the threads.db growth plan
(docs/2026-05-04-threads-db-layer-3-tool-result-eviction.org).

The unit tests in ``test_eviction_saver.py`` exercise eviction with
synthetic checkpoints constructed via ``empty_checkpoint()``.  This
test covers the wiring: that a ``ThreadManager`` instantiated normally
gets an ``EvictionSaver``, and that the round-trip works against a
real ``threads.db`` SQLite file (not ``:memory:``) the same way prod
will use it.

No LLM calls — the saver is exercised directly via its public
``put`` / ``get_tuple`` API, so this test is fast and doesn't depend
on a running model.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from unittest import TestCase

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.base import empty_checkpoint

from assist.eviction_saver import EvictionSaver
from assist.thread import ThreadManager


def _config(tid: str) -> dict:
    return {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}


class TestThreadManagerWiring(TestCase):
    def test_thread_manager_uses_eviction_saver(self):
        with tempfile.TemporaryDirectory() as root:
            tm = ThreadManager(root_dir=root)
            try:
                self.assertIsInstance(tm.checkpointer, EvictionSaver)
                # eviction_root must be ThreadManager.root_dir so Layer 0's
                # rmtree(thread_dir) cleans up eviction blobs for free.
                self.assertEqual(tm.checkpointer.eviction_root, tm.root_dir)
                # default threshold is 20 KB (env var unset in test process).
                self.assertEqual(tm.checkpointer.evict_threshold_kb, 20)
            finally:
                tm.close()

    def test_real_db_round_trip(self):
        """Drive a real ``threads.db`` (file-backed, not in-memory) through
        a put + get_tuple cycle with a 100 KB ``ToolMessage``.  Asserts:

        - The eviction blob lands at ``<root>/<tid>/large_tool_results/<sha256_16>``.
        - The serialized checkpoint row in ``threads.db`` does NOT contain
          the original 100 KB bytes (the load-bearing DB-bytes claim).
        - ``get_tuple`` rehydrates the message back to its full content.
        """
        with tempfile.TemporaryDirectory() as root:
            tm = ThreadManager(root_dir=root)
            try:
                tid = "tid_int_test"
                big = "Z" * 100_000  # 100 KB > 20 KB threshold

                tm.checkpointer.put(
                    _config(tid),
                    self._build_checkpoint(big),
                    {}, {},
                )

                # Eviction blob on disk, under per-thread dir.
                evict_dir = os.path.join(root, tid, "large_tool_results")
                self.assertTrue(os.path.isdir(evict_dir))
                files = os.listdir(evict_dir)
                self.assertEqual(len(files), 1)
                with open(os.path.join(evict_dir, files[0]), "rb") as f:
                    self.assertEqual(f.read(), big.encode("utf-8"))

                # threads.db row does NOT contain the 100 KB original.
                # This is the load-bearing claim — Layer 3's reason to exist.
                db_path = os.path.join(root, "threads.db")
                self.assertTrue(os.path.exists(db_path))
                with sqlite3.connect(db_path) as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT checkpoint FROM checkpoints WHERE thread_id = ?",
                        (tid,),
                    )
                    blob = cur.fetchone()[0]
                self.assertNotIn(big.encode("utf-8"), bytes(blob))

                # get_tuple rehydrates back to full content.
                tup = tm.checkpointer.get_tuple(_config(tid))
                self.assertIsNotNone(tup)
                msgs = tup.checkpoint["channel_values"]["messages"]
                self.assertEqual(msgs[0].content, big)
            finally:
                tm.close()

    def test_hard_delete_cascades_to_eviction_blobs(self):
        """Layer 0's ``hard_delete`` rmtrees the per-thread directory.
        Eviction blobs live INSIDE that directory, so they go too —
        no separate cleanup path needed."""
        with tempfile.TemporaryDirectory() as root:
            tm = ThreadManager(root_dir=root)
            try:
                tid = "tid_for_hard_delete"
                # Bootstrap a thread directory the way ``ThreadManager.new()``
                # would, then put a checkpoint with eviction.
                os.makedirs(os.path.join(root, tid), exist_ok=True)
                tm.checkpointer.put(
                    _config(tid),
                    self._build_checkpoint("X" * 100_000),
                    {}, {},
                )
                evict_dir = os.path.join(root, tid, "large_tool_results")
                self.assertTrue(os.path.isdir(evict_dir))

                tm.hard_delete(tid)

                # The whole thread dir is gone; eviction blobs with it.
                self.assertFalse(os.path.exists(os.path.join(root, tid)))
            finally:
                tm.close()

    def _build_checkpoint(self, content: str) -> dict:
        c = empty_checkpoint()
        c["id"] = "00000001"
        c["channel_values"] = {
            "messages": [
                ToolMessage(content=content, tool_call_id="call_int", name="t")
            ],
        }
        return c
