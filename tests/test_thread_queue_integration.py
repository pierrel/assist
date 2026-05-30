"""Integration tests for Thread.message / Thread.stream_message and the queue.

Covers the reviewer's gap items #3 and #4: the queue must release on
exceptions raised from inside the agent invocation, and the
``stream_message`` generator must hold the queue for the duration of
iteration (and release on iterator close / exception).
"""
import os
import shutil
import tempfile
import threading
from unittest import TestCase
from unittest.mock import patch

from langchain.messages import AIMessage

from assist.thread import Thread, ThreadManager
from assist.thread_queue import THREAD_QUEUE


class _ThreadQueueIntegrationBase(TestCase):
    def setUp(self):
        # The conftest fixture stubs `_probe_endpoint` so model
        # selection doesn't network, but `_get_config` first checks
        # ASSIST_MODEL_URL and raises if unset.  Set a placeholder.
        self._prev_url = os.environ.get("ASSIST_MODEL_URL")
        os.environ["ASSIST_MODEL_URL"] = "http://test.local/v1"
        self.root = tempfile.mkdtemp()
        self.tm = ThreadManager(self.root)

    def tearDown(self):
        try:
            self.tm.close()
        except Exception:
            pass
        shutil.rmtree(self.root, ignore_errors=True)
        if self._prev_url is None:
            os.environ.pop("ASSIST_MODEL_URL", None)
        else:
            os.environ["ASSIST_MODEL_URL"] = self._prev_url
        # Belt-and-suspenders: confirm no test leaked the queue holder.
        self.assertIsNone(
            THREAD_QUEUE.current_handle(),
            "test leaked queue holder; clean-up failed",
        )


class TestThreadMessageReleasesQueueOnException(_ThreadQueueIntegrationBase):
    def test_message_releases_queue_when_invoke_raises(self):
        chat = self.tm.new()
        with patch(
            "assist.thread.invoke_with_rollback",
            side_effect=RuntimeError("boom from inside the agent"),
        ):
            with self.assertRaises(RuntimeError):
                chat.message("hello")
        # Queue is empty: a subsequent thread should be free to acquire.
        self.assertIsNone(THREAD_QUEUE.current_handle())
        self.assertEqual(THREAD_QUEUE.waiter_count(), 0)

    def test_message_releases_queue_after_normal_completion(self):
        chat = self.tm.new()
        fake_result = {"messages": [AIMessage(content="ok")]}
        with patch(
            "assist.thread.invoke_with_rollback",
            return_value=fake_result,
        ):
            resp = chat.message("hello")
        self.assertEqual(resp, "ok")
        self.assertIsNone(THREAD_QUEUE.current_handle())


class TestThreadStreamMessageHoldsAndReleasesQueue(_ThreadQueueIntegrationBase):
    def _patch_stream(self, chunks, raise_on=None):
        """Replace ``self.agent.stream`` on the thread with a generator that
        yields ``chunks`` and optionally raises after a given index.
        """

        def fake_stream(*args, **kwargs):
            for i, chunk in enumerate(chunks):
                if raise_on is not None and i == raise_on:
                    raise RuntimeError("boom mid-stream")
                yield chunk

        return fake_stream

    def test_queue_acquired_only_when_iteration_starts(self):
        chat = self.tm.new()
        gen = None
        with patch.object(chat.agent, "stream", self._patch_stream(["a", "b"])):
            gen = chat.stream_message("hello")
            # No iteration yet — the queue should be untouched.
            self.assertIsNone(THREAD_QUEUE.current_handle())
            chunks = list(gen)
        self.assertEqual(chunks, ["a", "b"])
        self.assertIsNone(THREAD_QUEUE.current_handle())

    def test_queue_held_during_iteration_released_on_exhaustion(self):
        chat = self.tm.new()
        seen_holder: list = []
        with patch.object(chat.agent, "stream", self._patch_stream(["a", "b", "c"])):
            for chunk in chat.stream_message("hello"):
                seen_holder.append(THREAD_QUEUE.current_handle())
        # During iteration the queue had a holder for this thread.
        self.assertTrue(all(h is not None for h in seen_holder))
        self.assertTrue(all(h.thread_id == chat.thread_id for h in seen_holder))
        # Released after exhaustion.
        self.assertIsNone(THREAD_QUEUE.current_handle())

    def test_queue_released_when_stream_iteration_raises(self):
        chat = self.tm.new()
        with patch.object(
            chat.agent, "stream", self._patch_stream(["a", "b"], raise_on=1)
        ):
            with self.assertRaises(RuntimeError):
                list(chat.stream_message("hello"))
        self.assertIsNone(THREAD_QUEUE.current_handle())

    def test_queue_released_when_iterator_is_closed_early(self):
        chat = self.tm.new()
        with patch.object(chat.agent, "stream", self._patch_stream(["a", "b", "c"])):
            gen = chat.stream_message("hello")
            next(gen)
            self.assertIsNotNone(THREAD_QUEUE.current_handle())
            gen.close()
        # Generator close runs the finally → exits the `with` → releases queue.
        self.assertIsNone(THREAD_QUEUE.current_handle())

    def test_iterator_can_be_driven_from_different_os_thread(self):
        # The streaming iterator binds to a captured contextvars.Context at
        # construction time, so iterating from a different OS thread does NOT
        # trigger a ValueError on THREAD_QUEUE's `_active_handle.reset(token)`.
        # Without that binding, a consumer using `run_in_executor → next()`
        # (or any cross-thread iteration) exits the with-block in a different
        # Context than entered — and the queue holder leaks until the
        # watchdog catches it.  The 2026-05-28 incident was this exact bug.
        chat = self.tm.new()
        chunks: list = []
        errors: list = []

        with patch.object(chat.agent, "stream", self._patch_stream(["a", "b", "c"])):
            stream_iter = chat.stream_message("hello")

            def consumer():
                try:
                    for chunk in stream_iter:
                        chunks.append(chunk)
                except Exception as e:
                    errors.append(e)

            t = threading.Thread(target=consumer)
            t.start()
            t.join(timeout=2.0)

        # Worker must have finished — without this guard a hang inside
        # `ctx.run` would let `join` return on timeout with empty errors,
        # a partial `chunks`, and a released `current_handle()` (because
        # the holder hasn't been touched yet), masking the very bug this
        # test is supposed to catch.
        self.assertFalse(t.is_alive(), "consumer thread did not finish")
        self.assertFalse(errors, f"cross-thread iteration raised: {errors}")
        self.assertEqual(chunks, ["a", "b", "c"])
        # Queue released cleanly — no ValueError on with-exit means no leak.
        self.assertIsNone(THREAD_QUEUE.current_handle())
