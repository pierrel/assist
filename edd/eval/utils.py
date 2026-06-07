import os
import shutil
import subprocess
from unittest import TestCase
from langchain_core.messages import ToolMessage, AIMessage


class AgentTestMixin:
    """
    Mixin for TestCase classes that adds agent-specific assertions.

    Usage:
        class MyTest(AgentTestMixin, TestCase):
            def test_something(self):
                agent, root = self.create_agent({...})
                agent.message("Write a file")
                self.assertToolCall(agent, "write_file", "Should have written")
    """

    def assertToolCall(self, agent, tool_name: str, msg: str = None):
        """
        Assert that a specific tool was called by the agent.

        Args:
            agent: The AgentHarness instance
            tool_name: The name of the tool to check for
            msg: Optional custom assertion message
        """
        tool_calls = [m.name for m in agent.all_messages() if isinstance(m, ToolMessage)]

        if msg is None:
            msg = f"Tool '{tool_name}' should have been called. Called tools: {tool_calls}"

        self.assertIn(tool_name, tool_calls, msg)

    def subagent_calls(self, agent) -> list[str]:
        """Names of every subagent dispatched via the `task` tool, in order.

        Reads AIMessage tool_calls (outgoing) rather than ToolMessages
        (results), so a dispatch counts even if the subagent errored.
        Returns a list (not a set) so callers can assert on *how many*
        times a subagent was dispatched, not just whether it was.
        """
        calls = []
        for m in agent.all_messages():
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    if tc.get("name") == "task":
                        # deepagents' task tool names the target via
                        # `subagent_type`, but the small model sometimes
                        # emits it under `agent`/`name` instead — the
                        # dev-agent evals (test_dev_agent.py:167,
                        # test_dev_agent_planning_flow.py:157) carry the
                        # same fallback against that observed shape, so
                        # match it here for consistent counting.  The
                        # `or` chain also recovers an empty `subagent_type`
                        # (which SubagentTypeInferenceMiddleware would
                        # otherwise default to general-purpose).
                        args = tc.get("args") or {}
                        sa = (args.get("subagent_type")
                              or args.get("agent")
                              or args.get("name") or "")
                        if sa:
                            calls.append(sa)
        return calls

    def assertSubAgentCall(self, agent, subagent_name: str, msg: str = None):
        """
        Assert that a specific subagent was called by the agent via the task tool.

        Checks AIMessage tool_calls (outgoing calls) rather than ToolMessages (results),
        so this passes even if the subagent itself errors out.

        Args:
            agent: The AgentHarness instance
            subagent_name: The subagent_type value to look for (e.g. "dev-agent")
            msg: Optional custom assertion message
        """
        calls = self.subagent_calls(agent)

        if msg is None:
            msg = f"Subagent '{subagent_name}' should have been called via task tool. Called subagents: {calls}"

        self.assertIn(subagent_name, calls, msg)


def assertToolCall(test_case, agent, tool_name: str, msg: str = None):
    """
    Assert that a specific tool was called by the agent.

    This function can be used directly or as a helper to add to TestCase classes.

    Args:
        test_case: The TestCase instance (pass self from the test)
        agent: The AgentHarness instance
        tool_name: The name of the tool to check for
        msg: Optional custom assertion message

    Usage (direct):
        from tests.integration.validation.utils import assertToolCall

        agent, root = self.create_agent({...})
        agent.message("Do something")
        assertToolCall(self, agent, "write_file", "Should have written a file")

    Usage (as method - add to TestCase setUp):
        from tests.integration.validation.utils import assertToolCall

        class MyTest(TestCase):
            def setUp(self):
                # Add as instance method
                self.assertToolCall = lambda agent, tool, msg=None: assertToolCall(self, agent, tool, msg)

            def test_something(self):
                agent, root = self.create_agent({...})
                agent.message("Write a file")
                self.assertToolCall(agent, "write_file", "Should have written")
    """
    tool_calls = [m.name for m in agent.all_messages() if isinstance(m, ToolMessage)]

    if msg is None:
        msg = f"Tool '{tool_name}' should have been called. Called tools: {tool_calls}"

    test_case.assertIn(tool_name, tool_calls, msg)


def read_file(path: str) -> str:
    """Returns the full contents of file at path"""
    with open(path, 'r') as f:
        return f.read()

def files_in_directory(path: str) -> list[str]:
    """Returns the files in path as a list"""
    return os.listdir(path)

def skill_was_loaded(agent, skill_name: str) -> bool:
    """True iff a tool call loaded the named skill's body.

    Recognizes both routes the SkillsMiddleware exposes:

    - ``load_skill(name=skill_name)`` — the small-model tool registered by
      ``SmallModelSkillsMiddleware``.
    - ``read_file`` / ``read`` with a path containing ``/skills/<name>/`` —
      the upstream deepagents path.

    Shared by the skill-loading evals; grep tool-call args rather than results,
    since the model proves intent the moment it issues the call.
    """
    path_needle = f"/skills/{skill_name}/"
    for m in agent.all_messages():
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            args = tc.get("args") or {}
            if tc.get("name") == "load_skill" and args.get("name") == skill_name:
                return True
            for v in args.values():
                if isinstance(v, str) and path_needle in v:
                    return True
    return False


def executed_commands(agent) -> list[str]:
    """Command strings from every ``execute`` tool call, in order."""
    cmds = []
    for m in agent.all_messages():
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            if tc.get("name") == "execute":
                cmd = (tc.get("args") or {}).get("command", "")
                if cmd:
                    cmds.append(cmd)
    return cmds


def cleanup_workspace(path: str) -> None:
    """Remove a sandbox workspace, using Docker to delete root-owned files.

    Sandbox commands (pip, pytest, emacs, ...) write files as root inside the
    bind mount; a plain ``shutil.rmtree`` fails on those without an
    intermediate chmod, so run one in a throwaway alpine container first.
    """
    try:
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{path}:/cleanup', 'alpine',
             'sh', '-c', 'chmod -R 777 /cleanup 2>/dev/null; rm -rf /cleanup/*'],
            check=False, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    shutil.rmtree(path, ignore_errors=True)


def create_filesystem(root_dir: str,
                      structure: dict):
    """Creates a directory structure and files according to `structure`. For example:
    {"README.org": "This is the readme file",
    "gtd": {"inbox.org": "This is the inbox file"},
           {"projects": {"project1.org": "This is a project file"}}}

    Creates:
    a README.org file with content "This is the readme file"
    a gtd directory
    a gtd/inbox.org file with content "This is the inbox file"
    ..."""
    for name, content in structure.items():
        path = os.path.join(root_dir, name)

        if isinstance(content, str):
            # Create a file with the given content
            with open(path, 'w') as f:
                f.write(content)
        elif isinstance(content, dict):
            # Create a directory and recursively process its contents
            os.makedirs(path, exist_ok=True)
            create_filesystem(path, content)
