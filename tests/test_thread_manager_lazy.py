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
from unittest.mock import patch, MagicMock

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

    def test_new_leaves_working_dir_empty(self):
        """``MANAGER.new()`` must leave the default working dir empty.

        Regression: prod thread 20260504091127-183e8f35 lost the user's
        work because ``Thread.__init__`` (called eagerly by
        ``MANAGER.new()``) wired the research sub-agent via
        ``create_references_backend``, which used to ``os.makedirs``
        the ``references/`` dir on the host.  That left the workspace
        non-empty, so the background ``_initialize_thread``'s
        ``DomainManager(...)`` saw ``is_empty=False`` and silently
        skipped the git clone.  No ``.git/`` ever existed and every
        post-run ``dm.changes()`` blew up.

        The contract this test pins: between the moment ``MANAGER.new()``
        returns and the moment ``_initialize_thread`` begins its clone,
        the working dir must be empty.  Eager filesystem side effects
        in any agent factory called from ``Thread.__init__`` violate
        that contract.
        """
        # Patch create_deep_agent — its real implementation calls
        # init_chat_model() which requires a real model string.  We
        # don't care about the agent itself here; we care about the
        # filesystem side effects of the wiring around it.
        with patch("assist.thread.select_chat_model", return_value=MagicMock()), \
             patch("assist.agent.create_deep_agent", return_value=MagicMock()):
            with tempfile.TemporaryDirectory() as tmp:
                manager = ThreadManager(root_dir=tmp)
                chat = manager.new()
                working_dir = chat.working_dir
                self.assertTrue(os.path.isdir(working_dir))
                self.assertEqual(
                    os.listdir(working_dir), [],
                    f"MANAGER.new() left files in {working_dir}: "
                    f"{os.listdir(working_dir)} — DomainManager.is_empty "
                    "will return False and skip the git clone.",
                )
