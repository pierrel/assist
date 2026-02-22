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
from langgraph.errors import GraphRecursionError

from assist.model_manager import select_chat_model
from assist.agent import create_dev_agent, AgentHarness
from assist.sandbox_manager import SandboxManager


logger = logging.getLogger(__name__)

RECURSION_LIMIT = 500


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
        _project_root() + '/',
        dest + '/',
    ], check=True)


class TestDevAgent(TestCase):
    """Evals for the software development agent in a Docker sandbox."""

    @classmethod
    def setUpClass(cls):
        cls.model = select_chat_model("gpt-oss-20b", 0.1)
        cls.workspace = tempfile.mkdtemp(prefix="dev_agent_eval_")
        _rsync_project(cls.workspace)

        cls.sandbox = SandboxManager.get_sandbox_backend(cls.workspace)
        if cls.sandbox is None:
            raise RuntimeError("Docker sandbox unavailable — is Docker running and assist-sandbox built?")

    @classmethod
    def tearDownClass(cls):
        SandboxManager.cleanup(cls.workspace)
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def _create_agent(self):
        return AgentHarness(create_dev_agent(
            self.model,
            self.workspace,
            sandbox_backend=self.sandbox,
        ))

    def _invoke(self, agent, text: str) -> str:
        """Invoke the agent with a recursion limit.

        If the agent hits the recursion limit, log it but don't fail —
        we can still inspect the partial tool-call history.
        """
        try:
            resp = agent.agent.invoke(
                {"messages": [{"role": "user", "content": text}]},
                {"configurable": {"thread_id": agent.thread_id},
                 "recursion_limit": RECURSION_LIMIT},
            )
            return resp["messages"][-1].content
        except GraphRecursionError:
            logger.warning("Agent hit recursion limit (%d) — checking partial history", RECURSION_LIMIT)
            return ""

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

    def test_handles_basic_improvement(self):
        """The dev agent should make code improvements and write tests."""
        agent = self._create_agent()
        self._invoke(agent,
            "The `create_timestamped_branch` function in "
            "`assist/domain_manager.py` doesn't handle the case where the "
            "'main' branch doesn't exist. Add error handling that raises a "
            "clear ValueError if 'main' is missing. Write a test for this."
        )

        # Should have modified code
        modified = self._edited_paths(agent) + self._written_paths(agent)
        self.assertTrue(
            len(modified) > 0,
            f"Agent should modify or create files. Tool calls: "
            f"{[n for n, _ in self._get_tool_calls(agent)]}",
        )

        # Should have written a test
        test_files = [p for p in self._written_paths(agent) if 'test' in p.lower()]
        self.assertTrue(
            len(test_files) > 0,
            f"Agent should write a test file. Written: {self._written_paths(agent)}",
        )
