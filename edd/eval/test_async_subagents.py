"""Small-model contract for all-background delegation.

The subagents are deterministic stubs.  This suite evaluates the supervisor's
decisions and first visible reply, not child research quality or queue plumbing.
"""
from __future__ import annotations

import hashlib
import tempfile
from unittest import TestCase

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from assist.agent import AgentHarness, create_agent
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

from .utils import create_filesystem


class _TaskInput(BaseModel):
    description: str
    subagent_type: str


def _start(description: str, subagent_type: str) -> str:
    """Return the future production tool's observable scheduling result."""
    suffix = hashlib.sha256(description.encode()).hexdigest()[:12]
    return (f"Launched async subagent. task_id: task-eval-{suffix}. "
            "The result will be posted as a follow-up.")


class _TaskIdInput(BaseModel):
    task_id: str


class _UpdateInput(_TaskIdInput):
    instructions: str


def _check(task_id: str) -> str:
    return (f'{{"task_id":"{task_id}","status":"success",'
            '"result":"Weekend rail service resumes May 2 according to the report."}')


def _list() -> str:
    return ('{"task_id":"task-eval-outstanding0001","agent_name":"research-agent",'
            '"description":"Research the old destination","status":"pending"}')


def _update(task_id: str, instructions: str) -> str:
    return f"Task updated: {task_id}"


def _cancel(task_id: str) -> str:
    return f"Task cancelled: {task_id}"


_START = StructuredTool.from_function(
    name="start_async_task",
    func=_start,
    description=("Start a background task and return its task ID immediately. "
                 "Do not wait for or poll the result in this turn."),
    infer_schema=False,
    args_schema=_TaskInput,
)
_CHECK = StructuredTool.from_function(
    name="check_async_task", func=_check,
    description="Fetch fresh task status and result.",
    infer_schema=False, args_schema=_TaskIdInput)
_LIST = StructuredTool.from_function(
    name="list_async_tasks", func=_list,
    description="List all current tasks for this conversation.")
_UPDATE = StructuredTool.from_function(
    name="update_async_task", func=_update,
    description="Redirect pending task work.",
    infer_schema=False, args_schema=_UpdateInput)
_CANCEL = StructuredTool.from_function(
    name="cancel_async_task", func=_cancel,
    description="Cancel pending task work.",
    infer_schema=False, args_schema=_TaskIdInput)
_TOOLS = (_START, _CHECK, _UPDATE, _CANCEL, _LIST)


class TestAsyncSubagentSupervisor(TestCase):
    def setUp(self):
        self.model = select_assistant_model(0.1)

    def _agent(self) -> AgentHarness:
        root = tempfile.mkdtemp()
        create_filesystem(root, {
            "README.org": "Personal notes live here.",
            "trip.org": "* Possible trip\nNo destination research yet.\n",
        })
        spec = AgentSpec(async_subagent_tools=_TOOLS)
        return AgentHarness(create_agent(self.model, root, spec=spec))

    @staticmethod
    def _calls(agent: AgentHarness) -> list[dict]:
        return [call for message in agent.all_messages()
                if isinstance(message, AIMessage)
                for call in (message.tool_calls or [])]

    def test_starts_background_work_and_returns_without_polling(self):
        agent = self._agent()
        reply = agent.message(
            "Look through my trip notes and research current train options. "
            "Tell me what you started, then let me keep using this conversation.")
        calls = self._calls(agent)
        starts = [call for call in calls if call.get("name") == "start_async_task"]
        checks = [call for call in calls if call.get("name") == "check_async_task"]

        self.assertGreaterEqual(len(starts), 1, calls)
        self.assertEqual(checks, [], "the launch turn must not poll")
        self.assertRegex(reply, r"task-[A-Za-z0-9_-]+",
                         "the visible reply must include the full task ID")
        self.assertRegex(reply, r"(?i)(follow.?up|when .*finish|background|started)")
        self.assertNotRegex(reply, r"(?i)(I found|the train options are)",
                            "the launch reply must not invent unfinished results")

    def test_starts_independent_context_and_research_without_polling(self):
        agent = self._agent()
        reply = agent.message(
            "Look through my trip notes and independently research current national "
            "rail discount programs. Both briefs are already self-contained. Start "
            "every useful background task, tell me their IDs, and return.")
        calls = self._calls(agent)
        starts = [call for call in calls if call.get("name") == "start_async_task"]
        self.assertGreaterEqual(len(starts), 2, calls)
        self.assertFalse(any(call.get("name") == "check_async_task" for call in calls))
        for call in starts:
            description = call["args"]["description"]
            task_id = "task-eval-" + hashlib.sha256(
                description.encode()).hexdigest()[:12]
            self.assertIn(task_id, reply)

    def test_completion_wake_checks_then_uses_result(self):
        agent = self._agent()
        agent.message(
            "Look only through my local trip notes to find when weekend rail "
            "service resumes. Start the context background task and return while "
            "it runs; do not do external research.")
        start = next(call for call in reversed(self._calls(agent))
                     if call.get("name") == "start_async_task")
        task_id = "task-eval-" + hashlib.sha256(
            start["args"]["description"].encode()).hexdigest()[:12]
        reply = agent.message(
            "[Background task finished] [Async task completion] Task ID: "
            f"{task_id}. Status: success. This is orchestration metadata. "
            "Check the exact task before responding.")
        calls = self._calls(agent)
        checks = [call for call in calls if call.get("name") == "check_async_task"]
        self.assertTrue(checks, calls)
        self.assertEqual(checks[-1]["args"]["task_id"], task_id)
        self.assertRegex(reply, r"(?i)weekend.*May 2")

    def test_user_stop_request_lists_then_cancels_outstanding_task(self):
        agent = self._agent()
        agent.message("Hello")
        reply = agent.message(
            "I changed my mind. Stop the outstanding background work from my "
            "earlier request and tell me what you stopped.")
        calls = self._calls(agent)
        names = [call.get("name") for call in calls]
        self.assertIn("list_async_tasks", names, calls)
        self.assertIn("cancel_async_task", names, calls)
        self.assertLess(names.index("list_async_tasks"), names.index("cancel_async_task"))
        self.assertRegex(reply, r"task-eval-outstanding0001")
