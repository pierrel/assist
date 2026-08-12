"""Regression: web-server module load must not probe the LLM endpoint.

The user starts the web server and vLLM in parallel; the server must
boot even when vLLM is unreachable.  This works today only because
``ChatOpenAI(...)`` construction is lazy at the SDK layer — the first
``invoke()`` is what touches the network.

Adding HTTP-probe-based model discovery to ``select_chat_model``
threatened that property: ``ThreadManager.__init__`` used to call
``select_chat_model`` eagerly, and ``ThreadManager`` is constructed at
module-load time in ``manage/web/state.py``.  The fix made
``ThreadManager.model`` a lazy ``@property``; this test guards against
the regression.

See docs/2026-04-28-dynamic-model-plan.org §"Lazy ThreadManager.model".
"""
from __future__ import annotations

import os
import tempfile
from unittest import TestCase
from unittest.mock import patch, MagicMock

from assist.thread_engine import ThreadEngine, ThreadEngineError, read_thread_engine
from assist.thread_manager import ThreadManager


class TestThreadManagerLazy(TestCase):
    def test_reserve_visible_publishes_engine_before_returning(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = ThreadManager(root_dir=tmp)
            try:
                tid = manager.reserve_visible("pi", thread_id="pi-thread")
                self.assertEqual(tid, "pi-thread")
                self.assertEqual(
                    ThreadEngine("pi", "manual-web"),
                    read_thread_engine(manager.thread_dir(tid)),
                )
            finally:
                manager.close()

    def test_reserve_visible_rejects_engine_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = ThreadManager(root_dir=tmp)
            try:
                manager.reserve_visible("pi", thread_id="pi-thread")
                with self.assertRaises(ThreadEngineError):
                    manager.reserve_visible("deepagents", thread_id="pi-thread")
            finally:
                manager.close()

    def test_reserve_visible_never_replaces_generic_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = ThreadManager(root_dir=tmp)
            try:
                manager.reserve("existing")
                with self.assertRaises(ThreadEngineError):
                    manager.reserve_visible("pi", thread_id="existing")
                self.assertTrue(os.path.isdir(manager.thread_dir("existing")))
                self.assertFalse(os.path.exists(
                    os.path.join(manager.thread_dir("existing"), "engine.json")))
            finally:
                manager.close()

    def test_hidden_visible_reservation_staging_is_not_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = ThreadManager(root_dir=tmp)
            try:
                os.mkdir(os.path.join(tmp, ".thread-pending"))
                self.assertEqual(manager.list(), [])
            finally:
                manager.close()

    def test_list_preserves_existing_dot_prefixed_thread_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = ThreadManager(root_dir=tmp)
            try:
                manager.reserve(".visible")
                self.assertEqual(manager.list(), [".visible"])
            finally:
                manager.close()
    def test_thread_agent_dir_is_a_sibling_of_workspace_and_isolated_by_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = ThreadManager(root_dir=tmp)
            try:
                assert manager.thread_agent_dir("one") == os.path.join(tmp, "one", "agent")
                assert manager.thread_agent_dir("two") == os.path.join(tmp, "two", "agent")
                assert manager.thread_agent_dir("one") != manager.thread_agent_dir("two")
            finally:
                manager.close()

    def test_visible_main_get_receives_agent_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = ThreadManager(root_dir=tmp)
            manager.reserve("visible")
            manager._model = MagicMock()
            try:
                with patch("assist.thread_manager.Thread", return_value=MagicMock()) as thread:
                    manager.get("visible")
                assert thread.call_args.kwargs["agent_dir"] == manager.thread_agent_dir(
                    "visible")
                assert os.path.isdir(manager.thread_agent_dir("visible"))
            finally:
                manager.close()

    def test_specialized_child_get_does_not_receive_agent_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = ThreadManager(root_dir=tmp)
            manager.reserve("child", hidden={"parent_thread_id": "visible"})
            manager._model = MagicMock()
            try:
                with patch("assist.thread_manager.create_context_agent",
                           return_value=MagicMock()), \
                     patch("assist.thread_manager.Thread", return_value=MagicMock()) as thread:
                    manager.get("child", working_dir=os.path.join(tmp, "visible", "domain"),
                                assistant_id="context-agent")
                assert "agent_dir" not in thread.call_args.kwargs
            finally:
                manager.close()

    def test_init_does_not_call_select_assistant_model(self):
        """Constructing a ``ThreadManager`` must not touch the model."""
        with patch(
            "assist.thread_manager.select_assistant_model",
            side_effect=AssertionError("select_assistant_model called at init"),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                ThreadManager(root_dir=tmp)

    def test_first_model_access_calls_select(self):
        """First read of ``.model`` triggers ``select_assistant_model``;
        the result is cached for subsequent reads."""
        sentinel = object()
        with patch(
            "assist.thread_manager.select_assistant_model", return_value=sentinel
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
        with patch("assist.thread_manager.select_assistant_model", return_value=MagicMock()), \
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

    def test_delegate_profile_omits_web_supervisor_capabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = ThreadManager(root_dir=tmp)
            manager._model = MagicMock()
            manager.reserve("sub-delegate")
            with patch("assist.thread_manager.Thread") as thread:
                manager.get("sub-delegate", assistant_id="delegate-agent")

        spec = thread.call_args.kwargs["spec"]
        self.assertEqual(spec.role, "delegate")
        self.assertIsNone(spec.async_subagent_tools)
        self.assertEqual(spec.tools, ())
        self.assertEqual(dict(spec.skill_sources), {})
        self.assertIsNone(spec.interrupt_on)
        self.assertNotIn("agent_dir", thread.call_args.kwargs)

    def test_triage_profile_preserves_pre_disclosure_skill_composition(self):
        from assist.backends import (
            BundledSkillsBackend, LegacySkillsBackend, SKILLS_ROUTE)
        from deepagents.backends import FilesystemBackend

        with tempfile.TemporaryDirectory() as tmp:
            manager = ThreadManager(root_dir=tmp)
            manager._model = MagicMock()
            manager.reserve("triage")
            with patch("assist.thread_manager.Thread") as thread:
                manager.get("triage", triage=True)
            manager.close()

        spec = thread.call_args.kwargs["spec"]
        self.assertIn(SKILLS_ROUTE, spec.skill_sources)
        self.assertIsInstance(spec.skill_sources[SKILLS_ROUTE], LegacySkillsBackend)
        for backend in spec.skill_sources.values():
            self.assertIsInstance(backend, FilesystemBackend)
            self.assertNotIsInstance(backend, BundledSkillsBackend)
