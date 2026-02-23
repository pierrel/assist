"""Eval: dev-agent can run an inner eval inside its sandbox.

This proves the sandbox has access to ASSIST_* env vars (model URL, API key)
and can therefore invoke LLM-backed agents end-to-end.

Dependencies are pre-installed during setup so the agent only needs to
run the pytest command — dep installation is tested separately in
test_dev_agent.py::test_discovers_and_installs_dependencies.
"""
import logging
import os
import subprocess
import tempfile
import shutil
from unittest import TestCase

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from assist.model_manager import select_chat_model
from assist.agent import create_dev_agent, AgentHarness
from assist.sandbox_manager import SandboxManager


logger = logging.getLogger(__name__)

RECURSION_LIMIT = 500

PYTEST_CMD = (
    "python -m pytest edd/eval/test_context_agent.py"
    "::TestContextAgent::test_surfaces_todo_files_for_task_request -v"
)


def _project_root() -> str:
    """Return the root of the assist project."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _rsync_project(dest: str) -> None:
    """Rsync the assist project to *dest*, including edd/ for inner evals."""
    subprocess.run([
        'rsync', '-a',
        '--exclude', '.git',
        '--exclude', '.venv',
        '--exclude', '.venv_old',
        '--exclude', '__pycache__',
        '--exclude', '/reference',
        '--exclude', '.dev.env',
        '--exclude', '.deploy.env',
        '--exclude', 'improvements',
        _project_root() + '/',
        dest + '/',
    ], check=True)


class TestDevAgentRunsEval(TestCase):
    """The dev-agent runs an inner eval in the sandbox (deps pre-installed)."""

    @classmethod
    def setUpClass(cls):
        cls.model = select_chat_model("gpt-oss-20b", 0.1)
        cls.workspace = tempfile.mkdtemp(prefix="dev_agent_eval_runs_eval_")
        _rsync_project(cls.workspace)

        cls.sandbox = SandboxManager.get_sandbox_backend(cls.workspace)
        if cls.sandbox is None:
            raise RuntimeError(
                "Docker sandbox unavailable — is Docker running and assist-sandbox built?"
            )

        # Pre-install dependencies into system Python so that `pytest`,
        # `python -m pytest`, etc. all work without venv activation.
        # --break-system-packages bypasses Arch's externally-managed guard.
        result = cls.sandbox.execute(
            "pip install --break-system-packages -r requirements.txt -e ."
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Dep install failed:\n{result.output[:500]}"
            )

        # Verify the installation works end-to-end
        result = cls.sandbox.execute(PYTEST_CMD)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Verification pytest run failed:\n{result.output[:3000]}"
            )
        logger.info("Pre-installed dependencies and verified inner eval passes")

    @classmethod
    def tearDownClass(cls):
        SandboxManager.cleanup(cls.workspace)
        shutil.rmtree(cls.workspace, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helpers (same pattern as test_dev_agent.py)
    # ------------------------------------------------------------------

    def _create_agent(self):
        return AgentHarness(create_dev_agent(
            self.model,
            self.workspace,
            sandbox_backend=self.sandbox,
        ))

    def _invoke(self, agent, text: str) -> str:
        try:
            resp = agent.agent.invoke(
                {"messages": [{"role": "user", "content": text}]},
                {"configurable": {"thread_id": agent.thread_id},
                 "recursion_limit": RECURSION_LIMIT},
            )
            return resp["messages"][-1].content
        except GraphRecursionError:
            logger.warning(
                "Agent hit recursion limit (%d) — checking partial history",
                RECURSION_LIMIT,
            )
            return ""

    @staticmethod
    def _get_tool_calls(agent) -> list[tuple[str, dict]]:
        calls = []
        for m in agent.all_messages():
            if isinstance(m, AIMessage):
                for tc in (getattr(m, 'tool_calls', None) or []):
                    calls.append((tc.get('name', ''), tc.get('args', {})))
        return calls

    def _executed_commands(self, agent) -> list[str]:
        return [
            args.get('command', '')
            for name, args in self._get_tool_calls(agent)
            if name == 'execute'
        ]

    def _tool_results(self, agent) -> list[str]:
        """Return the text content of all ToolMessage results."""
        results = []
        for m in agent.all_messages():
            if isinstance(m, ToolMessage):
                content = m.content if isinstance(m.content, str) else str(m.content)
                results.append(content)
        return results

    def _ai_messages(self, agent) -> list[str]:
        """Return the text content of all AIMessage responses."""
        msgs = []
        for m in agent.all_messages():
            if isinstance(m, AIMessage) and m.content:
                msgs.append(m.content if isinstance(m.content, str) else str(m.content))
        return msgs

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def test_runs_context_agent_eval(self):
        """Dev-agent runs context agent eval, eval passes."""
        agent = self._create_agent()

        response = self._invoke(agent, (
            "Dependencies are already installed. "
            "Run this exact command using the execute tool:\n\n"
            f"  {PYTEST_CMD}\n\n"
            "Report whether the test passed or failed."
        ))

        # Check 1: agent used execute to run pytest
        commands = self._executed_commands(agent)
        pytest_cmds = [c for c in commands if 'pytest' in c]

        # Check 2: tool results contain "passed"
        results = self._tool_results(agent)
        tool_passed = any('passed' in r for r in results)

        # Check 3: agent's response mentions passing
        all_ai = self._ai_messages(agent)
        response_passed = any('passed' in msg.lower() for msg in all_ai)

        self.assertTrue(
            len(pytest_cmds) > 0,
            f"Agent should run pytest via execute. Executed: {commands}. "
            f"All tool calls: {[n for n, _ in self._get_tool_calls(agent)]}",
        )

        self.assertTrue(
            tool_passed or response_passed,
            f"Inner eval should pass. "
            f"Tool results (last 3): {[r[:300] for r in results[-3:]]}. "
            f"Agent response: {response[:500]}",
        )
