"""Eval: the agent maps natural-language recurrence to the right create_schedule args.

The cadence math + the tool are unit-tested (deterministic); the small-model risk is the
mapping from "on the 25th of every 2 months" to day_of_month=25, month_interval=2. We
assert on the create_schedule TOOL-CALL args the agent emits (not the store) — that isolates
the NL→args mapping and doesn't need the run config's tz to be wired.
"""
import os
import tempfile
from types import SimpleNamespace
from unittest import TestCase, mock

from assist.agent import create_agent, AgentHarness
from assist.context_rider import CONTEXT_RIDER_KEY
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec
from assist.schedule.tools import schedule_tools
from assist.schedule.store import ScheduleStore
from .utils import (
    agent_tool_calls, create_filesystem, skill_was_loaded,
    prompt_rewrite_web_main_spec,
    stub_research_subagent,
)

os.environ.setdefault("ASSIST_MODEL_URL", "http://127.0.0.1:8000/v1")


class TestScheduleAgent(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def _agent(self):
        root = tempfile.mkdtemp()
        create_filesystem(root, {"README.org": "Personal workspace."})
        store = ScheduleStore(tempfile.mkdtemp())
        return AgentHarness(create_agent(self.model, root,
                                         spec=AgentSpec(tools=tuple(schedule_tools(store)))))

    def _create_calls(self, agent) -> list:
        """args of every create_schedule call the agent emitted."""
        return [call.get("args") or call.get("arguments") or {}
                for call in agent_tool_calls(agent, "create_schedule")]

    def test_agent_maps_day_of_month(self):
        # Not a research eval — stub the research subagent so a stray dispatch can't hit
        # SearXNG (AGENTS.md #5). Build the agent INSIDE the stub (create_agent binds it).
        with stub_research_subagent():
            agent = self._agent()
            agent.message("Set up a recurring reminder on the 25th of each month to pay rent.")
        calls = self._create_calls(agent)
        self.assertTrue(skill_was_loaded(agent, "schedule"))
        self.assertTrue(
            any(c.get("day_of_month") == 25 for c in calls),
            f"expected a create_schedule with day_of_month=25; calls: {calls}")

    def test_agent_maps_skip_months(self):
        with stub_research_subagent():
            agent = self._agent()
            agent.message("Remind me on the 25th of every 2 months to review my finances.")
        calls = self._create_calls(agent)
        self.assertTrue(skill_was_loaded(agent, "schedule"))
        self.assertTrue(
            any(c.get("day_of_month") == 25 and c.get("month_interval") == 2 for c in calls),
            f"expected create_schedule with day_of_month=25, month_interval=2; calls: {calls}")

    def test_referential_followup_reloads_skill_and_modifies_schedule(self):
        """A new invocation resets disclosure, but the active task still routes."""
        with stub_research_subagent():
            agent = self._agent()
            agent.message("Remind me every day at 7 AM to take my vitamins.")
            before = len(agent_tool_calls(agent, "load_skill"))
            agent.message("Actually, make it 8 AM instead.")

        later_loads = agent_tool_calls(agent, "load_skill")[before:]
        self.assertTrue(any(
            (call.get("args") or {}).get("name") == "schedule"
            for call in later_loads), later_loads)
        self.assertTrue(agent_tool_calls(agent, "modify_schedule"),
                        "follow-up did not modify the existing schedule")


class TestPromptRewriteScheduleOutcome(TestCase):
    """Natural web-main comparison with a persisted recurring schedule outcome."""

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_creates_requested_recurring_reminder(self):
        thread_id = "schedule-eval"
        with tempfile.TemporaryDirectory(prefix="schedule_web_main_store_") as store_root, \
                tempfile.TemporaryDirectory(prefix="schedule_web_main_workspace_") as root:
            os.makedirs(os.path.join(store_root, thread_id))
            create_filesystem(root, {"README.org": "Personal workspace."})
            store = ScheduleStore(store_root)
            config = {"configurable": {
                "thread_id": thread_id,
                CONTEXT_RIDER_KEY: SimpleNamespace(tz="America/Los_Angeles"),
            }}
            with mock.patch("assist.schedule.tools.get_config", return_value=config), \
                 mock.patch("assist.tools.requests.get",
                            side_effect=AssertionError("schedule eval must not fetch URLs")) as get, \
                 stub_research_subagent():
                agent = AgentHarness(create_agent(
                    self.model, root,
                    spec=prompt_rewrite_web_main_spec(
                        tools=tuple(schedule_tools(store)))),
                    thread_id=thread_id)
                reply = agent.message(
                    "Remind me at 9 AM on the 25th of every two months to review my finances.")
            get.assert_not_called()
            saved = store.for_thread(thread_id)

        self.assertEqual(len(saved), 1, saved)
        cadence = saved[0].cadence
        self.assertEqual(cadence.day_of_month, 25)
        self.assertEqual(cadence.month_interval, 2)
        self.assertEqual(cadence.hour, 9)
        self.assertIn("financ", saved[0].prompt.lower())
        self.assertRegex(str(reply).lower(), r"(reminder|scheduled|next run)")
