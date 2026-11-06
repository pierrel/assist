"""Natural web-main requests should update the current thread description."""
import os
import tempfile
from unittest import TestCase, mock

from assist.agent import AgentHarness
from assist.model_manager import select_assistant_model
from assist.thread_manager import ThreadManager
from manage.web import state

from .utils import agent_tool_calls, stub_research_subagent


class ThreadDescriptionEval(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def _assert_updated(self, prompt, expected):
        with tempfile.TemporaryDirectory(prefix="thread-description-eval-") as root:
            manager = ThreadManager(root)
            self.addCleanup(manager.conn.close)
            tid = manager.reserve()
            self.addCleanup(state.DESCRIPTION_CACHE.pop, tid, None)
            with mock.patch.object(state, "MANAGER", manager):
                thread = manager.get(tid)
                with stub_research_subagent():
                    thread.message(prompt)

                harness = AgentHarness(thread.agent, thread_id=tid)
                calls = agent_tool_calls(harness, "update_thread_description")
                self.assertEqual(len(calls), 1,
                                 f"expected one update call, got {agent_tool_calls(harness)}")
                self.assertEqual(calls[0]["args"]["description"], expected)

                description_file = os.path.join(
                    manager.thread_dir(tid), "description.txt")
                with open(description_file) as saved:
                    self.assertEqual(saved.read(), expected)

                # Force the normal cached getter to read the saved file.
                state.DESCRIPTION_CACHE.pop(tid, None)
                self.assertEqual(state.get_cached_description(tid), expected)
                state.DESCRIPTION_CACHE.pop(tid, None)

    def test_update_this_threads_description_to(self):
        self._assert_updated(
            "Update this thread’s description to Weekend meal planning",
            "Weekend meal planning")

    def test_update_the_description_to(self):
        self._assert_updated(
            "update the description to Road trip checklist",
            "Road trip checklist")
