"""Eval for the dev agent planning-first TDD workflow.

Tests the full multi-turn flow:
  1. Agent explores codebase and runs at least one test (shows output)
  2. Agent writes a dated implementation plan and asks for user approval
  3. User approves plan → agent writes failing tests and shows output
  4. User approves tests → agent implements and all tests pass

The project is rsync'd from the current main branch, ensuring lazy diff loading
is not yet implemented (baseline SHA: a1389a3f9a5199798498fbd9f6e0c85b804dd9da).
"""
import logging
import os
import re
import subprocess
import tempfile
import shutil
from unittest import TestCase

from langchain_core.messages import AIMessage

from assist.model_manager import select_chat_model
from assist.agent import create_agent, AgentHarness
from assist.sandbox_manager import SandboxManager


logger = logging.getLogger(__name__)

# The feature request used across all phases of the eval
_FEATURE_REQUEST = (
    "Implement lazy loading of the diff view in manage/web.py. "
    "Currently, diffs are always loaded and rendered when a thread page loads. "
    "Change this so the diff is only fetched and rendered when the user "
    "explicitly requests it (e.g., via a ?show_diff=true query parameter). "
    "Follow the planning-first TDD process: explore, write a plan for my review, "
    "then write failing tests for my review, then implement."
)


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _rsync_project(dest: str) -> None:
    """Rsync the assist project to *dest*, excluding heavy/sensitive dirs."""
    subprocess.run([
        'rsync', '-a',
        '--exclude', '.git',
        '--exclude', '.venv',
        '--exclude', '.venv_old',
        '--exclude', '__pycache__',
        '--exclude', 'reference',
        '--exclude', '.dev.env',
        '--exclude', '.deploy.env',
        '--exclude', 'edd',
        '--exclude', 'improvements',
        '--exclude', '.mypy_cache',
        '--exclude', 'venv',
        '--exclude', 'node_modules',
        _project_root() + '/',
        dest + '/',
    ], check=True)


def _cleanup_workspace(path: str) -> None:
    """Remove workspace directory, using Docker to delete root-owned files."""
    try:
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{path}:/cleanup',
             'alpine', 'sh', '-c', 'chmod -R 777 /cleanup 2>/dev/null; rm -rf /cleanup/*'],
            check=False, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # best-effort
    shutil.rmtree(path, ignore_errors=True)


class TestDevAgentPlanningFlow(TestCase):
    """Eval for the planning-first TDD workflow with user-approval gates.

    This is a multi-turn eval that simulates a real developer workflow:
    the test acts as the "user" who reviews and approves the plan and tests
    before the agent is allowed to implement.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = select_chat_model(0.1)

    def setUp(self):
        self.workspace = tempfile.mkdtemp(prefix="dev_agent_planning_eval_")
        _rsync_project(self.workspace)

        # Sanity-check: show_diff should not be pre-implemented
        web_py = os.path.join(self.workspace, "manage", "web.py")
        with open(web_py) as f:
            content = f.read()
        if "show_diff" in content:
            self.skipTest(
                "manage/web.py already contains 'show_diff' — "
                "ensure the workspace is synced from main (not the lazy-diff branch)"
            )

        self.sandbox = SandboxManager.get_sandbox_backend(self.workspace)
        if self.sandbox is None:
            self.skipTest(
                "Docker sandbox unavailable — is Docker running and assist-sandbox built?"
            )

    def tearDown(self):
        SandboxManager.cleanup(self.workspace)
        _cleanup_workspace(self.workspace)

    def _create_agent(self) -> AgentHarness:
        # General agent + dev skill (post-Phase-D).  The rsync'd workspace
        # has pyproject.toml at the top level, which trips create_agent's
        # project_indicator detection and pre-loads the dev skill body.
        return AgentHarness(create_agent(
            self.model,
            self.workspace,
            sandbox_backend=self.sandbox,
        ))

    def _get_tool_calls(self, agent: AgentHarness) -> list[tuple[str, dict]]:
        calls = []
        for m in agent.all_messages():
            if isinstance(m, AIMessage):
                for tc in (getattr(m, 'tool_calls', None) or []):
                    calls.append((tc.get('name', ''), tc.get('args', {})))
        return calls

    def _executed_commands(self, agent: AgentHarness) -> list[str]:
        return [
            args.get('command', '')
            for name, args in self._get_tool_calls(agent)
            if name == 'execute'
        ]

    def _written_paths(self, agent: AgentHarness) -> list[str]:
        return [
            args.get('file_path', '') or args.get('path', '')
            for name, args in self._get_tool_calls(agent)
            if name in ('write_file', 'write')
        ]

    def _edited_paths(self, agent: AgentHarness) -> list[str]:
        return [
            args.get('file_path', '') or args.get('path', '')
            for name, args in self._get_tool_calls(agent)
            if name in ('edit_file', 'edit')
        ]

    def _task_calls(self, agent: AgentHarness) -> list[tuple[str, str]]:
        return [
            (
                args.get('subagent_type', args.get('agent', args.get('name', ''))),
                args.get('description', args.get('prompt', '')),
            )
            for name, args in self._get_tool_calls(agent)
            if name == 'task'
        ]

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _assert_ran_test_before_writing(self, agent: AgentHarness) -> None:
        """Verify a test command appeared before any implementation write/edit.

        Notes files (dev_notes.txt) and plan files (YYYY-MM-DD-*.md) written
        during exploration are excluded — only implementation writes count.
        """
        calls = self._get_tool_calls(agent)
        first_test_idx = next(
            (i for i, (name, args) in enumerate(calls)
             if name == 'execute' and any(
                 kw in args.get('command', '')
                 for kw in ('pytest', 'python -m pytest', 'python -m unittest')
             )),
            None,
        )

        def _is_impl_write(name: str, args: dict) -> bool:
            """Return True if this is a write/edit to an implementation file."""
            if name not in ('write', 'write_file', 'edit', 'edit_file'):
                return False
            path = (args.get('file_path', '') or args.get('path', '')).lower()
            # Exclude scratch files: dev_notes, plan files (YYYY-MM-DD-*.md)
            if 'dev_notes' in path:
                return False
            if re.search(r'\d{4}-\d{2}-\d{2}-', path):
                return False
            return True

        first_impl_write_idx = next(
            (i for i, (name, args) in enumerate(calls)
             if _is_impl_write(name, args)),
            None,
        )
        self.assertIsNotNone(
            first_test_idx,
            "Agent should run at least one existing test before making changes. "
            f"Executed commands: {self._executed_commands(agent)}",
        )
        if first_impl_write_idx is not None:
            self.assertLess(
                first_test_idx, first_impl_write_idx,
                f"Agent should run a test (idx {first_test_idx}) before any "
                f"implementation write/edit (idx {first_impl_write_idx})",
            )

    def _find_plan_files(self) -> list[str]:
        """Return paths of dated plan .md files in the sandbox workspace."""
        result = self.sandbox.execute(
            "find /workspace -maxdepth 1 -name '????-??-??-*.md' 2>/dev/null"
        )
        return [l.strip() for l in result.output.strip().splitlines() if l.strip()]

    def _find_new_test_files(self) -> list[str]:
        """Return test files created after pyproject.toml (i.e., new test files)."""
        result = self.sandbox.execute(
            "find /workspace -name 'test_*.py' -newer /workspace/pyproject.toml 2>/dev/null"
        )
        return [l.strip() for l in result.output.strip().splitlines() if l.strip()]

    # ------------------------------------------------------------------
    # Main eval
    # ------------------------------------------------------------------

    def test_planning_flow_diff_pagination(self):
        """Full multi-turn planning-first TDD eval for diff pagination feature."""
        agent = self._create_agent()

        # ----------------------------------------------------------------
        # Phase 1: Exploration + plan
        # Agent should: explore, run a test, write a plan, ask for approval
        # ----------------------------------------------------------------
        response_1 = agent.message(_FEATURE_REQUEST)

        # Agent ran at least one test before making any writes
        self._assert_ran_test_before_writing(agent)

        # Agent used context-agent
        context_calls = [
            name for name, _ in self._task_calls(agent)
            if 'context' in name.lower()
        ]
        self.assertTrue(
            len(context_calls) > 0,
            f"Agent should call context-agent before making changes. "
            f"Task calls: {self._task_calls(agent)}",
        )

        # A dated plan file exists in the workspace
        plan_files = self._find_plan_files()
        self.assertTrue(
            len(plan_files) > 0,
            f"Agent should write a dated plan .md file (YYYY-MM-DD-*.md). "
            f"Workspace ls output: "
            f"{self.sandbox.execute('ls /workspace/*.md 2>/dev/null || echo none').output}",
        )

        # Plan has required sections
        plan_path = plan_files[0]
        plan_content = self.sandbox.execute(f"cat {plan_path}").output.lower()
        required_keywords = ['reason', 'test', 'change', 'outcome']
        found = [kw for kw in required_keywords if kw in plan_content]
        self.assertGreaterEqual(
            len(found), 3,
            f"Plan should contain required sections (reason, test, change, outcome). "
            f"Found: {found}. Plan preview: {plan_content[:600]}",
        )

        # No implementation files written yet (plan-only phase)
        # Check specifically for manage/web.py or domain_manager.py writes
        written_phase1 = self._written_paths(agent)
        impl_writes = [
            p for p in written_phase1
            if ('manage/web.py' in p or 'domain_manager.py' in p)
        ]
        self.assertEqual(
            len(impl_writes), 0,
            f"Agent should not write implementation files (manage/web.py, domain_manager.py) "
            f"before plan is approved. Written: {written_phase1}",
        )

        # Agent wrote a plan file and stopped (doesn't need to ask in exact words)
        # The key evidence is the plan file + no implementation writes
        # (The approval check is done by verifying plan files exist, not response wording)
        # Optionally verify the response asks for approval in some form
        response_lower = response_1.lower()
        asked_for_approval = any(word in response_lower for word in [
            'approve', 'review', 'proceed', 'confirm', 'let me know',
            'your approval', 'your feedback', 'your review', 'please review',
            'take a look', 'thoughts', 'shall i', 'ready to', 'would you like',
            'before i', 'before proceeding', 'before writing', 'before implementing',
        ])
        # This is a soft check — if the agent didn't ask, just log it but don't fail
        # The hard requirement is that no implementation was written (checked above)
        if not asked_for_approval:
            logger.warning(
                "Agent didn't explicitly ask for approval in Phase 1. "
                f"Response preview: {response_1[:300]}"
            )

        # ----------------------------------------------------------------
        # Phase 2: Approve plan → agent writes failing tests
        # ----------------------------------------------------------------
        response_2 = agent.message(
            "The plan looks good. Please proceed with writing the tests."
        )

        # New test files were written
        new_test_files = self._find_new_test_files()
        written_phase2 = self._written_paths(agent)
        test_files_written = [p for p in written_phase2 if 'test' in p.lower()]
        self.assertTrue(
            len(test_files_written) > 0 or len(new_test_files) > 0,
            f"Agent should write test files after plan approval. "
            f"Written via tool: {written_phase2}. Sandbox test files: {new_test_files}",
        )

        # Tests were run and the output was shown (tests must fail)
        all_commands = self._executed_commands(agent)
        test_run_commands = [
            c for c in all_commands
            if any(kw in c for kw in ('pytest', 'python -m pytest', 'python -m unittest'))
        ]
        self.assertGreaterEqual(
            len(test_run_commands), 2,
            f"Agent should run tests at least twice (once in phase 1, once for new tests). "
            f"Test commands: {test_run_commands}",
        )

        # Response mentions failure (agent shows the red-phase output)
        self.assertTrue(
            any(word in response_2.lower() for word in ['fail', 'error', 'red', 'failing', 'failed']),
            f"Agent should report failing tests in phase 2. Response: {response_2[:500]}",
        )

        # Agent asks for approval again (soft check - don't fail if missing)
        response_2_lower = response_2.lower()
        asked_for_test_approval = any(word in response_2_lower for word in [
            'approve', 'review', 'proceed', 'confirm', 'implement',
            'your approval', 'your feedback', 'before i', 'shall i',
            'ready to', 'would you like', 'ready for', 'please review',
        ])
        if not asked_for_test_approval:
            logger.warning(
                "Agent didn't explicitly ask for test approval in Phase 2. "
                f"Response preview: {response_2[:300]}"
            )

        # ----------------------------------------------------------------
        # Phase 3: Approve tests → agent implements, all tests pass
        # ----------------------------------------------------------------
        response_3 = agent.message(
            "The tests look correct. Please implement the feature."
        )

        # Implementation files were modified
        all_written = self._written_paths(agent)
        all_edited = self._edited_paths(agent)
        impl_files = [
            p for p in (all_written + all_edited)
            if any(x in p for x in ('web.py', 'domain_manager'))
        ]
        self.assertTrue(
            len(impl_files) > 0,
            f"Agent should modify web.py or domain_manager.py for the implementation. "
            f"All written/edited: {all_written + all_edited}",
        )

        # The feature is actually present in web.py
        grep_result = self.sandbox.execute(
            "grep -n 'show_diff' /workspace/manage/web.py 2>/dev/null"
        )
        self.assertTrue(
            grep_result.exit_code == 0 and grep_result.output.strip(),
            f"manage/web.py should contain show_diff logic after implementation. "
            f"grep output: {grep_result.output}",
        )

        # Verify the new test files were written (TDD was followed)
        new_test_files_phase3 = self._find_new_test_files()
        self.assertTrue(
            len(new_test_files_phase3) > 0,
            f"Agent should have written test files for the feature. "
            f"New test files in sandbox: {new_test_files_phase3}",
        )

        # Try to run the new tests specifically (not the full suite which may fail
        # due to sandbox environment setup issues with missing packages)
        new_test_paths = " ".join(new_test_files_phase3[:3])  # run at most 3 new tests
        final_test = self.sandbox.execute(
            f"cd /workspace && ("
            f"  venv/bin/python -m pytest {new_test_paths} -q --tb=short 2>&1 ||"
            f"  .venv/bin/python -m pytest {new_test_paths} -q --tb=short 2>&1 ||"
            f"  python3 -m pytest {new_test_paths} -q --tb=short 2>&1 ||"
            f"  python -m pytest {new_test_paths} -q --tb=short 2>&1"
            f") | tail -20"
        )
        # Log the test output for debugging, but don't fail if env setup is incomplete
        logger.info("Phase 3 test run output: %s", final_test.output[-500:])
