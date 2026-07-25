"""Small-model contract for subagent delegation.

The subagents are deterministic stubs.  This suite evaluates the supervisor's
decisions and first visible reply, not child research quality or queue plumbing.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from unittest import TestCase, mock

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel

from assist.agent import AgentHarness, create_agent
from assist.async_subagents import SUBAGENTS
from assist.middleware.loop_detection import _messages_from_state
from assist.middleware.url_provenance import (
    delegated_general_purpose_body, delegated_general_purpose_description)
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

from .utils import create_filesystem


class _TaskInput(BaseModel):
    description: str
    subagent_type: str


class _TaskIdInput(BaseModel):
    task_id: str


class _UpdateInput(_TaskIdInput):
    instructions: str


_AVAILABLE = "Available types:\n" + "\n".join(
    f"- {name}: {description}" for name, description in SUBAGENTS.items())


def _task_id(description: str) -> str:
    return "task-eval-" + hashlib.sha256(description.encode()).hexdigest()[:12]


def _started_task(stub: "_TaskStub", call: dict) -> dict:
    description = call["args"]["description"]
    return next(
        task for task in stub.tasks.values()
        if task["agent_name"] == "general-purpose"
        and task["description"].endswith(description))


class _TaskStub:
    """Stateful deterministic stand-in for the production task lifecycle."""

    def __init__(self):
        self.tasks: dict[str, dict] = {}

    def start(self, description: str, subagent_type: str) -> str:
        if subagent_type not in SUBAGENTS:
            allowed = ", ".join(f"`{name}`" for name in SUBAGENTS)
            return f"Unknown subagent type `{subagent_type}`. Available: {allowed}"
        task_id = _task_id(description)
        self.tasks[task_id] = {
            "task_id": task_id,
            "agent_name": subagent_type,
            "description": description,
            "status": "pending",
        }
        return (f"Started subagent. task_id: {task_id}. In the user reply, call it a "
                "subagent or task, never background or async. Report this full ID "
                "and return now; the result will trigger a follow-up.")

    def check(self, task_id: str) -> str:
        task = dict(self.tasks.get(task_id, {
            "task_id": task_id, "status": "not_found"}))
        if task.get("agent_name") == "general-purpose":
            task["description"] = delegated_general_purpose_body(
                str(task.get("description") or ""))
        if task.get("status") in {"error", "timeout", "cancelled"}:
            task["instruction"] = (
                "This status is terminal. Report the failure and any blocked "
                "dependent work. Do not retry, redispatch, update, or complete "
                "this todo with direct tools.")
        return json.dumps(task)

    def list(self) -> str:
        tasks = []
        for source in self.tasks.values():
            task = dict(source)
            if task.get("agent_name") == "general-purpose":
                task["description"] = delegated_general_purpose_body(
                    str(task.get("description") or ""))
            tasks.append(task)
        return "\n".join(json.dumps(task) for task in tasks) \
            or "No subagent tasks exist for this conversation."

    def update(self, task_id: str, instructions: str, runtime: ToolRuntime) -> str:
        task = self.tasks.get(task_id)
        if task is None:
            return f"Task `{task_id}` was not found in this conversation."
        if task["status"] in {"success", "error", "timeout", "cancelled"}:
            return (f"Task `{task_id}` already completed with status "
                    f"{task['status']}.")
        try:
            task["description"] = (
                delegated_general_purpose_description(
                    instructions, _messages_from_state(runtime))
                if task["agent_name"] == "general-purpose" else instructions)
        except ValueError as exc:
            return str(exc)
        return f"Task updated: {task_id}"

    def cancel(self, task_id: str) -> str:
        task = self.tasks.get(task_id)
        if task is None:
            return f"Task `{task_id}` was not found in this conversation."
        if task["status"] in {"success", "error", "timeout"}:
            return (f"Task `{task_id}` already completed with status "
                    f"{task['status']}.")
        if task["status"] == "cancelled":
            return f"Task `{task_id}` is already cancelled."
        task["status"] = "cancelled"
        return f"Task cancelled: {task_id}"

    def complete(self, task_id: str, result: str) -> None:
        self.tasks[task_id].update(status="success", result=result)

    def fail(self, task_id: str, error: str) -> None:
        self.tasks[task_id].update(status="error", error=error)

    def tools(self):
        return (
            StructuredTool.from_function(
                name="start_async_task", func=self.start,
                description=("Start a subagent and return its task ID immediately. "
                             "Never poll in the launch turn. " + _AVAILABLE),
                infer_schema=False, args_schema=_TaskInput),
            StructuredTool.from_function(
                name="check_async_task", func=self.check,
                description=("Fetch fresh task status and result. On a task-completion "
                             "wake, this MUST be the first tool called with the exact "
                             "ID from the latest wake; never answer from wake metadata."),
                infer_schema=False, args_schema=_TaskIdInput),
            StructuredTool.from_function(
                name="update_async_task", func=self.update,
                description="Redirect pending task work.",
                infer_schema=False, args_schema=_UpdateInput),
            StructuredTool.from_function(
                name="cancel_async_task", func=self.cancel,
                description="Cancel pending task work.",
                infer_schema=False, args_schema=_TaskIdInput),
            StructuredTool.from_function(
                name="list_async_tasks", func=self.list,
                description="List all current tasks for this conversation."),
    )


def _compiled_leaf_stub(root: str):
    def complete(state):
        brief = str(state["messages"][-1].content)
        if "/announcement.md" in brief:
            create_filesystem(root, {"announcement.md": (
                "# Operator announcement\n\nThe release will follow Preflight, Deploy, "
                "Rollback, and Post-check phases. Operators should monitor health "
                "checks, be ready to execute rollback, and report anomalies during "
                "post-check monitoring.\n")})
            result = "Announcement completed exactly as requested at /announcement.md"
        elif "/runbook.md" in brief:
            create_filesystem(root, {"runbook.md": (
                "# Runbook\n## Preflight\nVerify backups and health.\n## Deploy\n"
                "Deploy artifacts and check services.\n## Rollback\nRestore the prior "
                "release and verify health.\n## Post-check\nRun smoke tests and monitor.\n")})
            result = "Runbook completed exactly as requested at /runbook.md"
        elif "/risks.md" in brief:
            create_filesystem(root, {"risks.md": (
                "| Risk | Likelihood | Impact | Mitigation |\n"
                "|---|---|---|---|\n| Expired certificate | Medium | High | Renew early |\n"
                "| Full disk | Medium | High | Monitor and clean |\n"
                "| Failed migration | Low | High | Test and retain rollback |\n"
                "| Stale cache | Medium | Medium | Purge after deploy |\n")})
            result = "Risk register completed exactly as requested at /risks.md"
        else:
            result = "Task completed."
        return {"messages": [AIMessage(content=result)]}

    graph = StateGraph(MessagesState)
    graph.add_node("complete", complete)
    graph.add_edge(START, "complete")
    graph.add_edge("complete", END)
    return graph.compile()


class TestAsyncSubagentSupervisor(TestCase):
    def setUp(self):
        self.model = select_assistant_model(0.1)

    def _agent(self) -> tuple[AgentHarness, _TaskStub]:
        root = tempfile.mkdtemp()
        create_filesystem(root, {
            "README.org": "Personal notes live here.",
            "trip.org": "* Possible trip\nNo destination research yet.\n",
        })
        stub = _TaskStub()
        stub.root = root
        spec = AgentSpec(async_subagent_tools=stub.tools())
        return AgentHarness(create_agent(self.model, root, spec=spec)), stub

    @staticmethod
    def _calls(agent: AgentHarness) -> list[dict]:
        return [call for message in agent.all_messages()
                if isinstance(message, AIMessage)
                for call in (message.tool_calls or [])]

    def test_starts_subagents_and_returns_without_polling(self):
        agent, _ = self._agent()
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
        self.assertRegex(reply, r"(?i)(follow.?up|when .*finish|started)")
        self.assertNotRegex(reply, r"(?i)\b(background|async)\b",
                            "all web subagents should be named simply as subagents")
        self.assertNotRegex(reply, r"(?i)(I found|the train options are)",
                            "the launch reply must not invent unfinished results")

    def test_starts_independent_context_and_research_without_polling(self):
        agent, _ = self._agent()
        reply = agent.message(
            "Look through my trip notes and independently research current national "
            "rail discount programs. Both briefs are already self-contained. Start "
            "every useful subagent task, tell me their IDs, and return.")
        calls = self._calls(agent)
        starts = [call for call in calls if call.get("name") == "start_async_task"]
        self.assertGreaterEqual(len(starts), 2, calls)
        self.assertFalse(any(call.get("name") == "check_async_task" for call in calls))
        for call in starts:
            description = call["args"]["description"]
            task_id = _task_id(description)
            self.assertIn(task_id, reply)

    def test_completion_wake_checks_then_uses_result(self):
        agent, stub = self._agent()
        agent.message(
            "Look only through my local trip notes to find when weekend rail "
            "service resumes. Start the context subagent and return while "
            "it runs; do not do external research.")
        start = next(call for call in reversed(self._calls(agent))
                     if call.get("name") == "start_async_task")
        task_id = _task_id(start["args"]["description"])
        stub.complete(task_id, "Weekend rail service resumes May 2 according to the report.")
        reply = agent.message(
            "[Background task finished] Task ID: "
            f"{task_id}. Status: success. This is orchestration metadata. "
            "Check the exact task before responding.")
        calls = self._calls(agent)
        checks = [call for call in calls if call.get("name") == "check_async_task"]
        self.assertTrue(checks, calls)
        self.assertEqual(checks[-1]["args"]["task_id"], task_id)
        self.assertRegex(reply, r"(?i)weekend.*May 2")

    def test_user_stop_request_lists_then_cancels_outstanding_task(self):
        agent, stub = self._agent()
        agent.message("Hello")
        agent.message(
            "Research the latest overnight train schedule from Oakland to Portland.")
        task_id = next(
            task["task_id"] for task in stub.tasks.values()
            if task["agent_name"] == "research-agent"
            and task["status"] == "pending")
        reply = agent.message(
            "I changed my mind. Stop the train research task you just started and "
            "tell me what you stopped.")
        calls = self._calls(agent)
        names = [call.get("name") for call in calls]
        self.assertIn("cancel_async_task", names, calls)
        self.assertTrue(any(
            call.get("name") == "cancel_async_task"
            and call.get("args", {}).get("task_id") == task_id
            for call in calls), calls)
        self.assertIn(task_id, reply)

    def test_parallelizes_independent_todos_into_specific_general_tasks(self):
        agent, stub = self._agent()
        agent.message("Hello")
        before = len(agent.all_messages())
        reply = agent.message(
            "Everything needed is in this message; there are no related files. I need "
            "three deliverables. Build a detailed release runbook with preflight, "
            "deploy, rollback, and post-check phases from these constraints: owner is "
            "Sam, maintenance window is 02:00-03:00, rollback after two failed health "
            "checks. Separately turn these risks into a likelihood/impact/mitigation "
            "register: expired certificate, full disk, failed migration, stale cache. "
            "After the runbook is finished, draft an operator announcement from it. "
            "No outside facts are needed.")
        messages = agent.all_messages()[before:]
        ai_messages = [message for message in messages if isinstance(message, AIMessage)]
        calls_by_message = [message.tool_calls or [] for message in ai_messages]
        flat = [call for calls in calls_by_message for call in calls]
        general_starts_by_message = [[
            call for call in calls
            if call.get("name") == "start_async_task"
            and call.get("args", {}).get("subagent_type") == "general-purpose"
        ] for calls in calls_by_message]
        launch_index, launch_batch = max(
            enumerate(general_starts_by_message),
            key=lambda item: len(item[1]), default=(-1, []))

        todo_index = next((index for index, calls in enumerate(calls_by_message)
                           if any(call.get("name") == "write_todos" for call in calls)),
                          -1)
        self.assertGreaterEqual(todo_index, 0, flat)
        self.assertIn(launch_index, {todo_index, todo_index + 1}, calls_by_message)
        self.assertGreaterEqual(len(launch_batch), 2, calls_by_message)
        self.assertFalse(
            {call.get("name") for call in calls_by_message[launch_index]}
            & {"write_file", "edit_file", "execute"},
            calls_by_message[launch_index])
        self.assertFalse(
            {call.get("name") for call in flat}
            & {"write_file", "edit_file", "execute"}, flat)
        descriptions = [call["args"]["description"] for call in launch_batch]
        self.assertEqual(sum("runbook" in text.lower() for text in descriptions), 1,
                         descriptions)
        self.assertEqual(sum("risk" in text.lower() and "register" in text.lower()
                             for text in descriptions), 1,
                         descriptions)
        runbook_brief = next(text for text in descriptions
                             if "runbook" in text.lower())
        for detail in ("preflight", "deploy", "rollback", "post-check", "Sam",
                       "02:00", "two", "failed health checks"):
            self.assertIn(detail.lower(), runbook_brief.lower(), runbook_brief)
        risk_brief = next(text for text in descriptions
                          if "risk" in text.lower() and "register" in text.lower())
        for detail in ("likelihood", "impact", "mitigation", "expired certificate",
                       "full disk", "failed migration", "stale cache"):
            self.assertIn(detail, risk_brief.lower(), risk_brief)
        self.assertFalse(any("announcement" in text.lower() for text in descriptions),
                         "dependent work must wait for the runbook")
        launched_tasks = [
            task for task in stub.tasks.values()
            if task["agent_name"] == "general-purpose"
        ]
        self.assertEqual(len(launched_tasks), 2, launched_tasks)
        for task in launched_tasks:
            self.assertIn(task["task_id"], reply)

    def test_completion_starts_newly_unblocked_general_task(self):
        agent, stub = self._agent()
        agent.message("Hello")
        agent.message(
            "Everything needed is below; there are no related files. I need three "
            "deliverables. Build a detailed release runbook at /runbook.md with "
            "preflight, deploy, rollback, and post-check phases; owner Sam; window "
            "02:00-03:00; rollback after two failed health checks. Separately turn "
            "expired certificate, full disk, failed migration, and stale cache into "
            "a risk register at /risks.md. After the runbook succeeds, draft an "
            "operator announcement from it into /announcement.md. No outside facts "
            "are needed.")
        runbook_call = next(
            call for call in self._calls(agent)
            if call.get("name") == "start_async_task"
            and call.get("args", {}).get("subagent_type") == "general-purpose"
            and "/runbook.md" in call.get("args", {}).get("description", ""))
        runbook = _started_task(stub, runbook_call)
        stub.complete(runbook["task_id"],
                      "Release runbook completed at /runbook.md")
        create_filesystem(stub.root, {
            "runbook.md": (
                "# Release runbook\nPreflight\nDeploy\nRollback\nPost-check\n"),
        })
        before = len(agent.all_messages())
        reply = agent.message(
            "[Background task finished] Task ID: "
            f"{runbook['task_id']}. Status: success. This is orchestration metadata. "
            "Check the exact task before responding.")
        calls = [call for message in agent.all_messages()[before:]
                 if isinstance(message, AIMessage)
                 for call in (message.tool_calls or [])]
        self.assertTrue(any(call.get("name") == "check_async_task"
                            and call.get("args", {}).get("task_id") == runbook["task_id"]
                            for call in calls), calls)
        announcement_starts = [
            call for call in calls
            if call.get("name") == "start_async_task"
            and call.get("args", {}).get("subagent_type") == "general-purpose"
            and "announcement" in call.get("args", {}).get("description", "").lower()
        ]
        self.assertEqual(len(announcement_starts), 1, calls)
        announcement_brief = announcement_starts[0]["args"]["description"]
        self.assertIn("/runbook.md", announcement_brief, announcement_brief)
        self.assertIn("/announcement.md", announcement_brief, announcement_brief)
        announcement_id = _started_task(
            stub, announcement_starts[0])["task_id"]
        self.assertIn(announcement_id, reply)

    def test_context_dependent_general_tasks_wait_then_launch_together(self):
        agent, stub = self._agent()
        agent.message(
            "Update my existing release runbook with a rollback section and, "
            "separately, reformat my existing risk register into its usual table. "
            "I do not know their paths or current formats. No outside "
            "facts are needed.")
        first_calls = self._calls(agent)
        context_start = next(
            call for call in first_calls
            if call.get("name") == "start_async_task"
            and call.get("args", {}).get("subagent_type") == "context-agent")
        self.assertFalse(any(
            call.get("name") == "start_async_task"
            and call.get("args", {}).get("subagent_type") == "general-purpose"
            for call in first_calls), first_calls)
        context_id = _task_id(context_start["args"]["description"])
        stub.complete(
            context_id,
            "Runbook: /runbook.md, Markdown with phase headings. Risk register: "
            "/risks.md, Markdown table with Risk/Likelihood/Impact/Mitigation columns.")
        create_filesystem(stub.root, {
            "runbook.md": "# Runbook\n## Deploy\n",
            "risks.md": "| Risk | Likelihood | Impact | Mitigation |\n",
        })

        before = len(agent.all_messages())
        agent.message(
            "[Background task finished] Task ID: "
            f"{context_id}. Status: success. This is orchestration metadata. "
            "Check the exact task before responding.")
        messages = [message for message in agent.all_messages()[before:]
                    if isinstance(message, AIMessage)]
        calls_by_message = [message.tool_calls or [] for message in messages]
        general_starts_by_message = [[
            call for call in calls
            if call.get("name") == "start_async_task"
            and call.get("args", {}).get("subagent_type") == "general-purpose"
        ] for calls in calls_by_message]
        launch_index, launch_batch = max(
            enumerate(general_starts_by_message),
            key=lambda item: len(item[1]), default=(-1, []))
        todo_index = next((
            index for index, calls in enumerate(calls_by_message)
            if any(call.get("name") == "write_todos" for call in calls)), -1)

        self.assertGreaterEqual(todo_index, 0, calls_by_message)
        self.assertIn(launch_index, {todo_index, todo_index + 1}, calls_by_message)
        self.assertEqual(len(launch_batch), 2, calls_by_message)
        descriptions = [call["args"]["description"] for call in launch_batch]
        self.assertTrue(any("/runbook.md" in text for text in descriptions),
                        descriptions)
        self.assertTrue(any("/risks.md" in text for text in descriptions),
                        descriptions)
        self.assertFalse(
            {call.get("name") for calls in calls_by_message for call in calls}
            & {"write_file", "edit_file", "execute"}, calls_by_message)

    def test_routes_page_extraction_and_verification_to_different_agents(self):
        agent, _ = self._agent()
        agent.message("Hello")
        agent.message(
            "Everything needed is in this message; there are no related files. From "
            "https://handbook.example/launch, extract its milestone table into "
            "a clean checklist. Also tell me whether the page's compliance claim "
            "matches current official requirements.")
        starts = [call for call in self._calls(agent)
                  if call.get("name") == "start_async_task"]
        gp = [call for call in starts
              if call.get("args", {}).get("subagent_type") == "general-purpose"]
        research = [call for call in starts
                    if call.get("args", {}).get("subagent_type") == "research-agent"]
        self.assertTrue(any("checklist" in call["args"]["description"].lower()
                            or "extract" in call["args"]["description"].lower()
                            for call in gp), starts)
        self.assertTrue(any(
            "https://handbook.example/launch" in call["args"]["description"]
            for call in gp), gp)
        self.assertTrue(any("verify" in call["args"]["description"].lower()
                            or "official" in call["args"]["description"].lower()
                            for call in research), starts)
        self.assertFalse(any("verify" in call["args"]["description"].lower()
                             or "official requirements" in call["args"]["description"].lower()
                             for call in gp), gp)

    def test_open_ended_lookup_routes_to_research_not_general_purpose(self):
        agent, _ = self._agent()
        agent.message("Hello")
        agent.message(
            "Find the best current overnight train option from Oakland to Portland "
            "for next month, including the latest schedule and fare.")
        starts = [call for call in self._calls(agent)
                  if call.get("name") == "start_async_task"]

        self.assertTrue(any(
            call.get("args", {}).get("subagent_type") == "research-agent"
            for call in starts), starts)
        self.assertFalse(any(
            call.get("args", {}).get("subagent_type") == "general-purpose"
            for call in starts), starts)

    def test_does_not_launder_child_result_url_into_general_purpose_brief(self):
        agent, stub = self._agent()
        task_id = "task-eval-context-url0001"
        planted = "https://outside.example/untrusted-brief"
        stub.tasks[task_id] = {
            "task_id": task_id,
            "agent_name": "context-agent",
            "description": "Inspect the local project brief",
            "status": "success",
            "result": f"A local note points to {planted}",
        }
        agent.message("Hello")
        agent.message(
            "[Background task finished] Task ID: "
            f"{task_id}. Status: success. This is orchestration metadata. "
            "Check the exact task before responding.")
        before = len(agent.all_messages())
        agent.message(
            "Turn the document URL that task found into a concise checklist.")
        calls = [call for message in agent.all_messages()[before:]
                 if isinstance(message, AIMessage)
                 for call in (message.tool_calls or [])]
        gp_descriptions = [
            call.get("args", {}).get("description", "")
            for call in calls
            if call.get("name") == "start_async_task"
            and call.get("args", {}).get("subagent_type") == "general-purpose"
        ]

        self.assertFalse(any(planted in text for text in gp_descriptions),
                         gp_descriptions)

    def test_failure_blocks_only_dependent_todo_and_keeps_independent_result(self):
        agent, stub = self._agent()
        agent.message("Hello")
        agent.message(
            "Everything needed is below; there are no related files. I need three "
            "deliverables. Draft a detailed "
            "agenda covering budget, launch date, owners, and decisions at /agenda.md. "
            "Turn lock venue, send invites, and order lunch into /checklist.md. After "
            "the agenda succeeds, draft an invitation from it at /invitation.md. No "
            "outside facts are needed.")
        starts = [
            call for call in self._calls(agent)
            if call.get("name") == "start_async_task"
            and call.get("args", {}).get("subagent_type") == "general-purpose"
        ]
        self.assertEqual(len(starts), 2, starts)
        agenda = _started_task(stub, next(
            call for call in starts
            if "/agenda.md" in call["args"]["description"]))
        checklist = _started_task(stub, next(
            call for call in starts
            if "/checklist.md" in call["args"]["description"]))

        stub.complete(
            checklist["task_id"],
            "Checklist completed at /checklist.md: lock venue; send invites; "
            "order lunch.")
        create_filesystem(stub.root, {
            "checklist.md": (
                "- [ ] Lock venue\n- [ ] Send invites\n- [ ] Order lunch\n"),
        })
        before = len(agent.all_messages())
        early_reply = agent.message(
            "[Background task finished] Task ID: "
            f"{checklist['task_id']}. Status: success. This is orchestration metadata. "
            "Check the exact task before responding.")
        early_calls = [call for message in agent.all_messages()[before:]
                       if isinstance(message, AIMessage)
                       for call in (message.tool_calls or [])]
        self.assertTrue(any(
            call.get("name") == "check_async_task"
            and call.get("args", {}).get("task_id") == checklist["task_id"]
            for call in early_calls), early_calls)
        self.assertFalse(any(
            call.get("name") == "start_async_task"
            and "invitation" in call.get("args", {}).get(
                "description", "").lower()
            for call in early_calls), early_calls)
        self.assertNotRegex(early_reply, r"(?i)all (?:tasks|deliverables).*(?:done|complete)")

        stub.fail(agenda["task_id"], "agenda source was invalid")
        final_reply = agent.message(
            "[Background task finished] Task ID: "
            f"{agenda['task_id']}. Status: error. This is orchestration metadata. "
            "Check the exact task before responding.")
        invitation_tasks = [
            task for task in stub.tasks.values()
            if task["agent_name"] == "general-purpose"
            and "/invitation" in task["description"].lower()
        ]

        self.assertEqual(invitation_tasks, [],
                         (final_reply, self._calls(agent)[-8:]))
        self.assertRegex(final_reply, r"(?i)agenda.*(?:fail|error)")
        self.assertRegex(final_reply, r"(?i)invitation.*(?:block|cannot|could not)")
        self.assertRegex(final_reply, r"(?i)checklist.*(?:done|complete|success)")

    def test_legacy_mode_parallelizes_independent_general_tasks(self):
        root = tempfile.mkdtemp()
        with mock.patch(
                "assist.agent.create_general_purpose_subagent",
                side_effect=lambda *_args, **_kwargs: _compiled_leaf_stub(root)):
            agent = AgentHarness(create_agent(self.model, root))
        agent.message("Hello")
        before = len(agent.all_messages())
        agent.message(
            "Everything needed is here; there are no related files. Build a release "
            "runbook at /runbook.md with preflight, deploy, rollback, and post-check phases. "
            "Separately turn expired certificate, full disk, failed migration, and "
            "stale cache into a likelihood/impact/mitigation risk register at /risks.md. After "
            "the runbook succeeds, draft an operator announcement from it at "
            "/announcement.md. No "
            "outside facts are needed.")
        batches = [
            [call for call in (message.tool_calls or [])
             if call.get("name") == "task"
             and call.get("args", {}).get("subagent_type") == "general-purpose"]
            for message in agent.all_messages()[before:]
            if isinstance(message, AIMessage)
        ]
        launch_index, first_parallel_batch = max(
            enumerate(batches), key=lambda item: len(item[1]), default=(-1, []))
        ai_messages = [
            message for message in agent.all_messages()[before:]
            if isinstance(message, AIMessage)
        ]
        todo_index = next((
            index for index, message in enumerate(ai_messages)
            if any(call.get("name") == "write_todos"
                   for call in (message.tool_calls or []))), -1)

        self.assertGreaterEqual(todo_index, 0, ai_messages)
        self.assertIn(launch_index, {todo_index, todo_index + 1}, ai_messages)
        self.assertEqual(len(first_parallel_batch), 2, batches)
        self.assertFalse(
            {call.get("name") for call in (ai_messages[launch_index].tool_calls or [])}
            & {"write_file", "edit_file", "execute"},
            ai_messages[launch_index].tool_calls)
        descriptions = [call["args"]["description"]
                        for call in first_parallel_batch]
        self.assertTrue(any("runbook" in text.lower() for text in descriptions),
                        descriptions)
        self.assertTrue(any("risk" in text.lower() and "register" in text.lower()
                            for text in descriptions), descriptions)
        self.assertFalse(any("announcement" in text.lower()
                             for text in descriptions), descriptions)
