"""Natural, fixture-backed quiet evals with a real admitted web Run."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest import TestCase

from assist.agent import AgentHarness, create_agent
from assist.checkpoint_rollback import invoke_with_rollback
from assist.events.quiet import quiet_tools
from assist.model_manager import select_assistant_model
from assist.run_service import RunService
from .utils import agent_tool_calls, prompt_rewrite_web_main_spec, stub_research_subagent


_BACKUP_PROMPT = ("It's time for the daily backup check. Look at backup-status.txt "
                  "and tell me if anything needs attention.")


class TestQuiet(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="quiet_eval_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _run(self, *, fixture_name: str, fixture: str, prompt: str,
             require_read: bool = True) -> tuple[bool, dict]:
        Path(self.root, fixture_name).write_text(fixture)
        thread_id = "quiet-eval-thread"
        Path(self.root, thread_id).mkdir()
        runs = RunService(self.root)
        run = runs.create(thread_id, "general-agent", prompt)
        runs.claim(thread_id, run.id)
        spec = replace(prompt_rewrite_web_main_spec(
            tools=quiet_tools(runs.request_quiet)), web_main=True)
        with stub_research_subagent():
            agent = AgentHarness(create_agent(self.model, self.root, spec=spec),
                                 thread_id=thread_id)
            invoke_with_rollback(agent.agent,
                                 {"messages": [{"role": "user", "content": prompt}]},
                                 {"configurable": {"thread_id": thread_id,
                                                   "web_run_id": run.id},
                                  "recursion_limit": 5000})
        trace = {"calls": agent_tool_calls(agent),
                 "messages": [(type(message).__name__, str(message.content)[:300])
                              for message in agent.all_messages()]}
        if require_read:
            self.assertTrue(any(call["name"] == "read_file" for call in trace["calls"]),
                            f"fixture was not inspected: {trace}")
        return runs.get(thread_id, run.id).quiet_requested, trace

    def test_nightly_backup_without_change_is_quiet(self):
        requested, calls = self._run(
            fixture_name="backup-status.txt",
            fixture="Nightly backup completed at 03:14 UTC. No errors. No files changed. "
                    "No operator action is needed.\n",
            prompt=_BACKUP_PROMPT)
        self.assertTrue(requested, f"routine check did not successfully call quiet: {calls}")

    def test_capability_explicit_quiet(self):
        requested, trace = self._run(
            fixture_name="check.txt", fixture="No changes and no action needed.\n",
            prompt="This is a capability check: call quiet() now, then confirm it worked.",
            require_read=False)
        self.assertTrue(requested, f"quiet capability failed: {trace}")

    def test_failed_backup_stays_new(self):
        requested, calls = self._run(
            fixture_name="backup-status.txt",
            fixture="Nightly backup failed at 03:14 UTC. Error: repository disk full. "
                    "The backup has not completed; an operator must free space and retry.\n",
            prompt=_BACKUP_PROMPT)
        self.assertFalse(requested, f"actionable failure was suppressed: {calls}")

    def test_incomplete_backup_stays_new(self):
        requested, calls = self._run(
            fixture_name="backup-status.txt",
            fixture="Nightly backup started at 03:14 UTC. The status log ends before "
                    "any completion or failure record. Current outcome is unknown.\n",
            prompt=_BACKUP_PROMPT)
        self.assertFalse(requested, f"uncertain outcome was suppressed: {calls}")

    def test_useful_manual_answer_stays_new(self):
        requested, calls = self._run(
            fixture_name="meeting-note.txt",
            fixture="Planning note: the launch is Thursday. I own the release checklist. "
                    "The team needs to confirm the rollback contact by Wednesday.\n",
            prompt="Summarize meeting-note.txt for me.")
        self.assertFalse(requested, f"useful manual answer was suppressed: {calls}")

    def test_manual_routine_status_is_quiet(self):
        requested, calls = self._run(
            fixture_name="service-status.txt",
            fixture="Service check completed at 03:14 UTC. All services are healthy. "
                    "No changes or operator action are needed.\n",
            prompt="Please do the routine service check now. Look at service-status.txt "
                   "and tell me if anything needs attention.")
        self.assertTrue(requested, f"manual routine check was not quiet: {calls}")
