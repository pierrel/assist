"""Regression: web-server module load must not probe the LLM endpoint.

The user starts the web server and vLLM in parallel; the server must
boot even when vLLM is unreachable.  This works today only because
``ChatOpenAI(...)`` construction is lazy at the SDK layer — the first
``invoke()`` is what touches the network.

Adding HTTP-probe-based model discovery to ``select_chat_model``
threatened that property: ``ThreadManager.__init__`` used to call
``select_chat_model`` eagerly, and ``ThreadManager`` is constructed at
module-load time in ``manage/web.py``.  The fix made
``ThreadManager.model`` a lazy ``@property``; this test guards against
the regression.

See docs/2026-04-28-dynamic-model-plan.org §"Lazy ThreadManager.model".
"""
from __future__ import annotations

import os
import tempfile
from unittest import TestCase
from unittest.mock import patch

from assist.thread import ThreadManager


class TestThreadManagerLazy(TestCase):
    def test_init_does_not_call_select_chat_model(self):
        """Constructing a ``ThreadManager`` must not touch the model."""
        with patch(
            "assist.thread.select_chat_model",
            side_effect=AssertionError("select_chat_model called at init"),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                ThreadManager(root_dir=tmp)

    def test_first_model_access_calls_select(self):
        """First read of ``.model`` triggers ``select_chat_model``; the
        result is cached for subsequent reads."""
        sentinel = object()
        with patch(
            "assist.thread.select_chat_model", return_value=sentinel
        ) as fake_select:
            with tempfile.TemporaryDirectory() as tmp:
                manager = ThreadManager(root_dir=tmp)
                self.assertEqual(fake_select.call_count, 0)
                first = manager.model
                second = manager.model
        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        self.assertEqual(fake_select.call_count, 1)
