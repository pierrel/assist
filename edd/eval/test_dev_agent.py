"""Evals for the software development agent.

Tests a dev agent running inside a Docker sandbox against this codebase.
The project is rsync'd to a temp directory and mounted into the container.

Expected behaviors:
1. Writes tests (or validation scripts) for all code change requests
2. Runs tests in the sandbox to verify behavior
3. Discovers how to install dependencies and does so
4. Handles basic code improvements correctly
5. Updates documentation when asked
6. Explains the codebase thoroughly
"""
import logging
import os
import subprocess
import tempfile
import shutil
import uuid
from unittest import TestCase

from langchain_core.messages import AIMessage, ToolMessage

from assist.model_manager import select_chat_model
from assist.agent import create_dev_agent, AgentHarness
from assist.sandbox_manager import SandboxManager


logger = logging.getLogger(__name__)


def _project_root() -> str:
    """Return the root of the assist project."""
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
        '--exclude', 'playground',
        '--exclude', 'comparisons',
        '--exclude', 'generated',
        '--exclude', 'eval_results',
        '--exclude', '*.log',
        '--exclude', 'logs',
        _project_root() + '/',
        dest + '/',
    ], check=True)


def _cleanup_workspace(path: str) -> None:
    """Remove workspace directory, using Docker to delete root-owned files.

    When the sandbox runs commands (pip install, etc.) it creates files owned
    by root inside the workspace volume.  shutil.rmtree fails on those.  We
    use a throwaway alpine container to chmod and remove them first.
    """
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


class TestDevAgent(TestCase):
    """Evals for the software development agent in a Docker sandbox.

    Each test gets its own workspace and sandbox container to ensure isolation.
    Agent actions in one test (e.g. creating a venv) cannot affect subsequent tests.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = select_chat_model("gpt-oss-20b", 0.1)

    def setUp(self):
        self.workspace = tempfile.mkdtemp(prefix="dev_agent_eval_")
        _rsync_project(self.workspace)

        self.sandbox = SandboxManager.get_sandbox_backend(self.workspace)
        if self.sandbox is None:
            self.skipTest("Docker sandbox unavailable — is Docker running and assist-sandbox built?")

    def tearDown(self):
        SandboxManager.cleanup(self.workspace)
        _cleanup_workspace(self.workspace)

    def _create_agent(self):
        return AgentHarness(create_dev_agent(
            self.model,
            self.workspace,
            sandbox_backend=self.sandbox,
        ))

    def _invoke(self, agent, text: str) -> str:
        """Invoke the agent and return the final response."""
        return agent.message(text)

    # ------------------------------------------------------------------
    # Helpers to inspect agent message history
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tool_calls(agent) -> list[tuple[str, dict]]:
        """Return [(tool_name, args_dict), ...] from the agent's history."""
        calls = []
        for m in agent.all_messages():
            if isinstance(m, AIMessage):
                for tc in (getattr(m, 'tool_calls', None) or []):
                    calls.append((tc.get('name', ''), tc.get('args', {})))
        return calls

    def _executed_commands(self, agent) -> list[str]:
        """Return command strings from all ``execute`` tool calls."""
        return [
            args.get('command', '')
            for name, args in self._get_tool_calls(agent)
            if name == 'execute'
        ]

    def _written_paths(self, agent) -> list[str]:
        """Return file paths from all ``write_file`` / ``write`` calls."""
        return [
            args.get('file_path', '') or args.get('path', '')
            for name, args in self._get_tool_calls(agent)
            if name in ('write_file', 'write')
        ]

    def _edited_paths(self, agent) -> list[str]:
        """Return file paths from all ``edit_file`` / ``edit`` calls."""
        return [
            args.get('file_path', '') or args.get('path', '')
            for name, args in self._get_tool_calls(agent)
            if name in ('edit_file', 'edit')
        ]

    def _task_calls(self, agent) -> list[tuple[str, str]]:
        """Return [(agent_name, prompt), ...] from all ``task`` tool calls.

        The deepagents task tool uses ``subagent_type`` as the agent name
        parameter.  Older keys (``agent``, ``name``) are kept as fallbacks
        for backward compatibility.
        """
        return [
            (
                args.get('subagent_type', args.get('agent', args.get('name', ''))),
                args.get('description', args.get('prompt', '')),
            )
            for name, args in self._get_tool_calls(agent)
            if name == 'task'
        ]

    def _tool_call_order(self, agent) -> list[str]:
        """Return tool names in order of invocation."""
        return [name for name, _ in self._get_tool_calls(agent)]

    # ------------------------------------------------------------------
    # Evals — ordered from simplest to most complex
    # ------------------------------------------------------------------

    def test_explains_codebase(self):
        """The dev agent should provide thorough codebase explanations."""
        agent = self._create_agent()
        response = self._invoke(agent,
            "Explain the architecture of this codebase. What are the main "
            "components, how do they interact, and what design patterns "
            "are used?"
        )

        response_lower = response.lower()
        key_terms = ['agent', 'thread', 'sandbox', 'domain', 'middleware']
        found = [t for t in key_terms if t in response_lower]
        self.assertGreaterEqual(
            len(found), 3,
            f"Response should reference key components. Found: {found}. "
            f"Preview: {response[:500]}",
        )
        self.assertGreater(
            len(response), 200,
            "Explanation should be thorough (>200 chars)",
        )

    def test_updates_documentation(self):
        """The dev agent should update documentation correctly."""
        agent = self._create_agent()
        self._invoke(agent,
            "Update the README.md to add a section explaining the "
            "SandboxManager class — what it does and how it differs "
            "from DomainManager."
        )

        # Check 1: direct write/edit tool calls on README
        all_modified = self._edited_paths(agent) + self._written_paths(agent)
        readme_edits = [p for p in all_modified if 'readme' in p.lower()]

        # Check 2: execute commands that modify README (echo, cat, sed, etc.)
        commands = self._executed_commands(agent)
        readme_cmds = [c for c in commands if 'readme' in c.lower() or 'README' in c]

        # Check 3: verify the file in the sandbox actually contains SandboxManager
        result = self.sandbox.execute("grep -i sandboxmanager /workspace/README.md")
        sandbox_has_content = result.exit_code == 0 and result.output.strip()

        self.assertTrue(
            len(readme_edits) > 0 or len(readme_cmds) > 0 or sandbox_has_content,
            f"Agent should update README with SandboxManager content. "
            f"Edit/write calls: {all_modified}, "
            f"Execute cmds mentioning README: {readme_cmds}, "
            f"Sandbox grep result: {result.output[:200] if result else 'N/A'}. "
            f"All tool calls: {[n for n, _ in self._get_tool_calls(agent)]}",
        )

    def test_discovers_and_installs_dependencies(self):
        """The dev agent should figure out how to install project deps."""
        agent = self._create_agent()
        self._invoke(agent,
            "Look at this Python project's dependency files and install "
            "the dependencies so you can run code."
        )

        commands = self._executed_commands(agent)
        install_cmds = [
            c for c in commands
            if 'pip install' in c or 'pip3 install' in c
        ]
        self.assertTrue(
            len(install_cmds) > 0,
            f"Agent should install dependencies. Executed: {commands}",
        )

    def test_writes_tests_for_feature_request(self):
        """The dev agent should write tests when asked to implement a feature."""
        agent = self._create_agent()
        self._invoke(agent,
            "Add a function `is_palindrome(s: str) -> bool` to a new file "
            "`assist/utils.py` that returns True if the string reads the same "
            "forwards and backwards (case-insensitive, ignoring spaces)."
        )

        # Check 1: test file created via write tool
        written = self._written_paths(agent)
        test_files_written = [p for p in written if 'test' in p.lower()]

        # Check 2: test file exists in sandbox
        result = self.sandbox.execute(
            "find /workspace -name 'test_*.py' -newer /workspace/pyproject.toml -o "
            "-name '*_test.py' -newer /workspace/pyproject.toml 2>/dev/null"
        )
        sandbox_test_files = [l for l in result.output.strip().splitlines() if l.strip()]

        self.assertTrue(
            len(test_files_written) > 0 or len(sandbox_test_files) > 0,
            f"Agent should write at least one test file. "
            f"Written via tool: {written}, Sandbox test files: {sandbox_test_files}",
        )

    def test_runs_tests(self):
        """The dev agent should execute tests after writing code."""
        agent = self._create_agent()
        self._invoke(agent,
            "Write a small test for the `is_git_repo` function in "
            "`assist/domain_manager.py` and run it to verify it works."
        )

        commands = self._executed_commands(agent)
        test_runs = [
            c for c in commands
            if any(kw in c for kw in ('pytest', 'python -m pytest', 'python -m unittest', 'python test_'))
        ]
        self.assertTrue(
            len(test_runs) > 0,
            f"Agent should run tests via execute. Executed: {commands}. "
            f"All tool calls: {[n for n, _ in self._get_tool_calls(agent)]}",
        )

    def test_uses_context_agent_before_changes(self):
        """The dev agent should call context-agent BEFORE making any writes or edits."""
        agent = self._create_agent()
        self._invoke(agent,
            "Add a logging statement to the top of every public method in "
            "`assist/thread.py` that logs the method name when called."
        )

        call_order = self._tool_call_order(agent)
        task_indices = [i for i, name in enumerate(call_order) if name == 'task']
        write_indices = [i for i, name in enumerate(call_order)
                         if name in ('write', 'write_file', 'edit', 'edit_file')]

        # context-agent should be called (via task) at least once
        task_calls = self._task_calls(agent)
        context_calls = [
            (name, prompt) for name, prompt in task_calls
            if 'context' in name.lower()
        ]
        self.assertTrue(
            len(context_calls) > 0,
            f"Agent should call context-agent before making changes. "
            f"Task calls: {task_calls}. All tool calls: {call_order}",
        )

        # The first task call should come before the first write/edit
        if task_indices and write_indices:
            self.assertLess(
                task_indices[0], write_indices[0],
                f"context-agent (task) should be called before any write/edit. "
                f"First task at index {task_indices[0]}, "
                f"first write/edit at index {write_indices[0]}",
            )

    def test_follows_existing_patterns(self):
        """The dev agent should follow existing patterns when creating new code.

        Uses two turns to test the planning-first TDD workflow:
          Turn 1: agent explores, writes plan, asks for approval
          Turn 2: user approves → agent implements following existing patterns
        """
        agent = self._create_agent()
        response_1 = self._invoke(agent,
            "Add a new middleware class called `RequestTimingMiddleware` in "
            "`assist/middleware/request_timing.py` that logs how long each "
            "model call takes. Follow the same patterns as the existing "
            "middleware classes in the project."
        )

        # Phase 1: agent should explore and write a plan
        task_calls = self._task_calls(agent)
        context_calls = [(n, p) for n, p in task_calls if 'context' in n.lower()]
        self.assertTrue(
            len(context_calls) > 0,
            f"Agent should call context-agent in phase 1. Task calls: {task_calls}",
        )

        # Phase 2: approve the plan → agent implements
        self._invoke(agent, "The plan looks good, please proceed with the implementation.")

        # Should have created the middleware file
        written = self._written_paths(agent) + self._edited_paths(agent)
        middleware_files = [p for p in written if 'timing' in p.lower() or 'middleware' in p.lower()]
        self.assertTrue(
            len(middleware_files) > 0,
            f"Agent should create the middleware file after approval. Modified: {written}",
        )

        # Verify the file exists in the sandbox and follows patterns
        result = self.sandbox.execute(
            "cat /workspace/assist/middleware/request_timing.py 2>/dev/null"
        )
        if result.exit_code == 0 and result.output.strip():
            content = result.output
            # Should use a class-based structure like other middleware
            self.assertIn('class', content,
                          "Middleware should be class-based like existing middleware")
            # Should import from the middleware ecosystem
            self.assertTrue(
                'middleware' in content.lower() or 'Middleware' in content,
                f"Should reference middleware patterns. Content preview: {content[:500]}",
            )

    def test_uses_research_agent(self):
        """The dev agent should use research-agent for unfamiliar topics."""
        agent = self._create_agent()
        self._invoke(agent,
            "Add rate limiting to the sandbox execute function using the "
            "token bucket algorithm. Research how the token bucket algorithm "
            "works and the best way to implement it in Python before writing code."
        )

        task_calls = self._task_calls(agent)
        research_calls = [
            (name, prompt) for name, prompt in task_calls
            if 'research' in name.lower()
        ]
        self.assertTrue(
            len(research_calls) > 0,
            f"Agent should call research-agent for unfamiliar topics. "
            f"Task calls: {task_calls}. "
            f"All tool calls: {self._tool_call_order(agent)}",
        )

    def test_leverages_existing_utilities(self):
        """The dev agent should reuse existing code instead of rewriting."""
        agent = self._create_agent()
        response = self._invoke(agent,
            "Add a function that takes a Jinja2 template name and a dict "
            "of variables, and returns the rendered string. Check the "
            "existing codebase first — there may already be something you "
            "can reuse."
        )

        # The agent should discover and reference promptable.py
        response_lower = response.lower()
        all_calls = self._get_tool_calls(agent)
        all_content = response_lower + ' '.join(
            str(args) for _, args in all_calls
        ).lower()

        self.assertTrue(
            'promptable' in all_content or 'base_prompt_for' in all_content
            or 'already exist' in response_lower or 'existing' in response_lower,
            f"Agent should discover and reference existing promptable.py utility. "
            f"Response preview: {response[:500]}",
        )

    def test_handles_basic_improvement(self):
        """The dev agent should make code improvements and write tests.

        Uses three turns to test the full planning-first TDD workflow:
          Turn 1: agent explores, writes plan, asks for approval
          Turn 2: user approves plan → agent writes failing tests, asks for approval
          Turn 3: user approves tests → agent implements and all tests pass
        """
        agent = self._create_agent()
        self._invoke(agent,
            "The `create_timestamped_branch` function in "
            "`assist/domain_manager.py` doesn't handle the case where the "
            "'main' branch doesn't exist. Add error handling that raises a "
            "clear ValueError if 'main' is missing. Write a test for this."
        )

        # Phase 1: agent should explore and write a plan (no code yet)
        task_calls_phase1 = self._task_calls(agent)
        context_calls = [(n, p) for n, p in task_calls_phase1 if 'context' in n.lower()]
        self.assertTrue(
            len(context_calls) > 0,
            f"Agent should call context-agent in phase 1. Task calls: {task_calls_phase1}",
        )

        # Phase 2: approve plan → agent writes failing tests
        self._invoke(agent, "The plan looks good. Please write the tests.")

        # Phase 3: approve tests → agent implements
        self._invoke(agent, "Tests look correct. Please implement the fix.")

        # Should have modified domain_manager.py
        all_edited = self._edited_paths(agent)
        all_written = self._written_paths(agent)
        impl_files = [
            p for p in (all_edited + all_written)
            if 'domain_manager' in p.lower()
        ]
        self.assertTrue(
            len(impl_files) > 0,
            f"Agent should modify domain_manager.py. "
            f"Edited: {all_edited}, Written: {all_written}",
        )

        # Should have written a test file
        test_files = [p for p in all_written if 'test' in p.lower()]
        self.assertTrue(
            len(test_files) > 0,
            f"Agent should write a test file. Written: {all_written}",
        )
