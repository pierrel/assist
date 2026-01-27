import os
from unittest import TestCase
from langchain_core.messages import ToolMessage


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
