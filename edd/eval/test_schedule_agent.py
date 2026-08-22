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

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import PrivateAttr

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


class _CountingSummaryModel(FakeMessagesListChatModel):
    """Deterministic summary model that exposes the number of compactions."""

    _summary_calls: int = PrivateAttr(default=0)

    @property
    def summary_calls(self) -> int:
        return self._summary_calls

    def _generate(self, *args, **kwargs):
        self._summary_calls += 1
        return super()._generate(*args, **kwargs)


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

    DELETE_PROMPT = (
        "Please remove the nightly scheduled reminder. "
        "I don't think I need it anymore."
    )
    COMPACTION_SUMMARY = (
        "The user reviewed two recurring meditation check-ins, one nightly and "
        "one in the morning. The assistant inspected them, and neither reminder "
        "was changed."
    )

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.enterContext(mock.patch.dict(os.environ, {
            "ASSIST_PROMPT_REWRITE_GUIDANCE_SKILLS": "1",
        }, clear=False))
        reset_task_fixture()

    def _run_delete_case(self, *, prior_schedule_turn: bool, compact: bool):
        """Run one natural removal case and verify its persisted outcome."""
        from contextlib import nullcontext
        from deepagents.middleware.summarization import SummarizationMiddleware

        thread_id = f"schedule-delete-{'compact' if compact else 'plain'}-eval"
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

        summary_model = _CountingSummaryModel(
            responses=[AIMessage(content=self.COMPACTION_SUMMARY)])
        delete_prompt = self.DELETE_PROMPT

        class DeleteTurnSummarizationMiddleware(SummarizationMiddleware):
            """Compact once, immediately before the deletion turn is modeled."""

            def _should_summarize(self, messages, _total_tokens):
                return bool(
                    messages
                    and isinstance(messages[-1], HumanMessage)
                    and messages[-1].content == delete_prompt
                )

        def delete_turn_summary(_model, backend):
            return DeleteTurnSummarizationMiddleware(
                summary_model,
                backend=backend,
                keep=("messages", 1),
            )

        summary_patch = (
            mock.patch(
                "deepagents.graph.create_summarization_middleware",
                side_effect=delete_turn_summary,
            )
            if compact else nullcontext()
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
                 stub_research_subagent(), summary_patch:
                agent = AgentHarness(create_agent(
                    self.model, root,
                    spec=prompt_rewrite_web_main_spec(
                        tools=tuple(schedule_tools(store)))),
                    thread_id=thread_id)

                initial_schedule_load_ids = []
                if prior_schedule_turn:
                    agent.message("Which meditation check-ins are currently scheduled?")
                    initial_calls = agent_tool_calls(agent)
                    initial_loads = [
                        call for call in initial_calls
                        if call.get("name") == "load_skill"
                        and (call.get("args") or {}).get("name") == "schedule"]
                    self.assertTrue(initial_loads, agent.all_messages())
                    self.assertTrue(
                        agent_tool_calls(agent, "list_schedules"),
                        agent.all_messages(),
                    )
                    initial_schedule_load_ids = [
                        call.get("id") for call in initial_loads]
                    self.assertEqual(store.for_thread(thread_id), [target, control])

                before_calls = len(agent_tool_calls(agent))
                before_state = agent.agent.get_state({
                    "configurable": {"thread_id": thread_id},
                }).values
                self.assertIsNone(
                    before_state.get("_summarization_event"), before_state)

                reply = str(agent.message(self.DELETE_PROMPT))
                state = agent.agent.get_state({
                    "configurable": {"thread_id": thread_id},
                }).values
            get.assert_not_called()
            saved = store.for_thread(thread_id)

        calls = agent_tool_calls(agent)[before_calls:]
        call_names = [call.get("name") for call in calls]
        messages = agent.all_messages()
        event = state.get("_summarization_event")
        diagnostics = {
            "calls": calls,
            "saved": saved,
            "reply": reply,
            "event": event,
            "messages": messages,
        }

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
        self.assertTrue(target_deletes, diagnostics)
        if compact or not prior_schedule_turn:
            self.assertTrue(schedule_lists, diagnostics)
            self.assertLess(schedule_loads[0], schedule_lists[0], diagnostics)
            self.assertLess(schedule_lists[0], target_deletes[0], diagnostics)
        else:
            self.assertLess(schedule_loads[0], target_deletes[0], diagnostics)
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

        if compact:
            self.assertIsNotNone(event, diagnostics)
            self.assertEqual(summary_model.summary_calls, 1, diagnostics)
            summary_message = event["summary_message"]
            self.assertEqual(
                summary_message.additional_kwargs.get("lc_source"),
                "summarization",
                diagnostics,
            )
            self.assertIn(self.COMPACTION_SUMMARY, str(summary_message.content))
            self.assertRegex(
                str(event.get("file_path")),
                r"^/conversation_history/.+\.md$",
                diagnostics,
            )
            initial_load_results = [
                i for i, message in enumerate(messages)
                if isinstance(message, ToolMessage)
                and message.tool_call_id in initial_schedule_load_ids]
            self.assertEqual(
                len(initial_load_results), len(initial_schedule_load_ids), diagnostics)
            self.assertTrue(all(
                result_index < event["cutoff_index"]
                for result_index in initial_load_results
            ), diagnostics)
        else:
            self.assertIsNone(event, diagnostics)

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
        self._run_delete_case(prior_schedule_turn=False, compact=False)

    def test_deletes_named_recurring_reminder_after_prior_schedule_turn(self):
        """An uncompacted end-to-end follow-up reloads before changing state."""
        self._run_delete_case(prior_schedule_turn=True, compact=False)

    def test_deletes_named_recurring_reminder_after_compaction(self):
        """A compacted follow-up reloads after prior instructions leave active context."""
        self._run_delete_case(prior_schedule_turn=True, compact=True)
