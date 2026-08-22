"""Evals for natural schedule selection and persisted outcomes.

The capability probes cover cadence mapping through ``create_schedule`` arguments
and referential changes through reload/modify tool traces. Production-shaped
web-main rows additionally assert on the persisted store so a plausible completion
response cannot substitute for the requested state change.
"""
import os
import tempfile
from types import SimpleNamespace
from unittest import TestCase, mock

from langchain_core.messages import ToolMessage

from assist.agent import create_agent, AgentHarness
from assist.context_rider import CONTEXT_RIDER_KEY
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec
from assist.schedule.model import Cadence, Schedule
from assist.schedule.tools import schedule_tools
from assist.schedule.store import ScheduleStore
from .test_async_subagents import reset_task_fixture
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
    """Natural web-main comparisons with persisted recurring-schedule outcomes."""

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.enterContext(mock.patch.dict(os.environ, {
            "ASSIST_PROMPT_REWRITE_GUIDANCE_SKILLS": "1",
        }, clear=False))
        reset_task_fixture()

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

    def test_deletes_named_recurring_reminder(self):
        """A natural removal request loads scheduling and changes persisted state."""
        thread_id = "schedule-delete-eval"
        target = Schedule(
            id="a17c9e4b23d1",
            thread_id=thread_id,
            prompt="Evening meditation check-in: ask whether today's session happened.",
            cadence=Cadence(hour=18, minute=0),
            tz="America/Los_Angeles",
            next_fire_at="2030-01-02T02:00:00+00:00",
            created_at="2026-08-21T16:00:00+00:00",
        )
        control = Schedule(
            id="f04d8a6c91e2",
            thread_id=thread_id,
            prompt="Morning meditation intention check-in.",
            cadence=Cadence(hour=8, minute=0),
            tz="America/Los_Angeles",
            next_fire_at="2030-01-01T16:00:00+00:00",
            created_at="2026-08-21T16:00:00+00:00",
        )

        with tempfile.TemporaryDirectory(prefix="schedule_delete_store_") as store_root, \
                tempfile.TemporaryDirectory(prefix="schedule_delete_workspace_") as root:
            os.makedirs(os.path.join(store_root, thread_id))
            create_filesystem(root, {"README.org": "Personal workspace."})
            store = ScheduleStore(store_root)
            store.add(target)
            store.add(control)
            config = {"configurable": {
                "thread_id": thread_id,
                CONTEXT_RIDER_KEY: SimpleNamespace(tz="America/Los_Angeles"),
            }}
            with mock.patch("assist.schedule.tools.get_config", return_value=config), \
                 mock.patch("assist.tools.requests.get",
                            side_effect=AssertionError(
                                "schedule eval must not fetch URLs")) as get, \
                 stub_research_subagent():
                agent = AgentHarness(create_agent(
                    self.model, root,
                    spec=prompt_rewrite_web_main_spec(
                        tools=tuple(schedule_tools(store)))),
                    thread_id=thread_id)
                reply = str(agent.message(
                    "Please remove the nightly scheduled reminder. "
                    "I don't think I need it anymore."))
            get.assert_not_called()
            saved = store.for_thread(thread_id)

        calls = agent_tool_calls(agent)
        call_names = [call.get("name") for call in calls]
        messages = agent.all_messages()
        diagnostics = {
            "calls": calls,
            "saved": saved,
            "reply": reply,
            "messages": messages,
        }
        self.assertTrue(skill_was_loaded(agent, "schedule"), diagnostics)
        self.assertTrue(calls, diagnostics)
        self.assertEqual(call_names[0], "load_skill", diagnostics)
        self.assertEqual(
            (calls[0].get("args") or {}).get("name"),
            "schedule",
            diagnostics,
        )
        schedule_loads = [
            i for i, call in enumerate(calls)
            if call.get("name") == "load_skill"
            and (call.get("args") or {}).get("name") == "schedule"]
        schedule_lists = [
            i for i, call in enumerate(calls)
            if call.get("name") == "list_schedules"]
        target_deletes = [
            i for i, call in enumerate(calls)
            if call.get("name") == "delete_schedule"
            and (call.get("args") or {}).get("schedule_id") == target.id]
        self.assertTrue(schedule_loads, diagnostics)
        self.assertTrue(schedule_lists, diagnostics)
        self.assertTrue(target_deletes, diagnostics)
        self.assertLess(schedule_loads[0], schedule_lists[0], diagnostics)
        self.assertLess(schedule_lists[0], target_deletes[0], diagnostics)
        self.assertFalse(any(
            call.get("name") == "delete_schedule"
            and (call.get("args") or {}).get("schedule_id") == control.id
            for call in calls), diagnostics)
        self.assertEqual(saved, [control], diagnostics)
        delete_call = calls[target_deletes[0]]
        delete_results = [
            message for message in messages
            if isinstance(message, ToolMessage)
            and message.tool_call_id == delete_call.get("id")]
        self.assertEqual(len(delete_results), 1, diagnostics)
        self.assertEqual(
            delete_results[0].content,
            f"Deleted schedule {target.id}.",
            diagnostics,
        )
        self.assertTrue(reply.strip(), diagnostics)
