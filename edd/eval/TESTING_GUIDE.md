# Testing Guide for Agent Tests

## Using assertToolCall

When testing agent behavior, you often want to verify that specific tools were called. The `assertToolCall` helper makes this easy.

### Method 1: Using AgentTestMixin (Recommended)

Inherit from `AgentTestMixin` to add `self.assertToolCall()` to your test class:

```python
from unittest import TestCase
from tests.integration.validation.utils import AgentTestMixin, create_filesystem

class MyAgentTest(AgentTestMixin, TestCase):
    def test_agent_writes_file(self):
        agent, root = self.create_agent({"README.md": "Initial content"})

        agent.message("Create a new file called output.txt")

        # Assert that write_file tool was called
        self.assertToolCall(agent, "write_file", "Should have written a file")
```

### Method 2: Using Standalone Function

If you prefer not to use the mixin, use the standalone function:

```python
from unittest import TestCase
from tests.integration.validation.utils import assertToolCall, create_filesystem

class MyAgentTest(TestCase):
    def test_agent_reads_file(self):
        agent, root = self.create_agent({"data.txt": "Some data"})

        agent.message("What's in data.txt?")

        # Pass self as first argument
        assertToolCall(self, agent, "read_file", "Should have read the file")
```

## Common Tool Names

- `read_file` - Reading files
- `write_file` - Writing files
- `ls` - Listing directories
- `glob` - Finding files by pattern
- `bash` - Running shell commands
- `task` - Invoking subagents

## Debugging Tool Calls

When a test fails, the assertion message will show which tools were actually called:

```
AssertionError: Tool 'write_file' should have been called. Called tools: ['read_file', 'ls', 'glob']
```

This helps you understand what the agent actually did vs. what you expected.

## Example: Testing File Operations

```python
from unittest import TestCase
from tests.integration.validation.utils import (
    AgentTestMixin,
    create_filesystem,
    read_file
)

class TestFileOperations(AgentTestMixin, TestCase):
    def setUp(self):
        self.model = select_chat_model("gpt-oss-20b", 0.1)

    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)
        return AgentHarness(create_agent(self.model, root)), root

    def test_agent_modifies_file(self):
        agent, root = self.create_agent({
            "todo.txt": "- Buy milk\n- Call dentist"
        })

        # Ask agent to add a todo
        res = agent.message("Add 'Fix bug' to my todo list")

        # Verify the tool was called
        self.assertToolCall(agent, "write_file", "Should have written to todo.txt")

        # Verify the content changed
        contents = read_file(f"{root}/todo.txt")
        self.assertIn("Fix bug", contents)
```
