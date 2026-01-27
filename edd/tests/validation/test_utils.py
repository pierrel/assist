"""
Tests for test utility functions.
"""
import os
import tempfile
from textwrap import dedent
from unittest import TestCase

from assist.model_manager import select_chat_model
from assist.agent import create_agent, AgentHarness

from .utils import read_file, create_filesystem, assertToolCall, AgentTestMixin


class TestAssertToolCallStandalone(TestCase):
    """Test the standalone assertToolCall function"""
    def setUp(self):
        self.model = select_chat_model("gpt-oss-20b", 0.1)

    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)
        return AgentHarness(create_agent(self.model, root)), root

    def test_assert_tool_call_success(self):
        """Test that assertToolCall passes when tool was called"""
        agent, root = self.create_agent({
            "README.md": "Test file"
        })

        # Make agent read a file
        agent.message("What's in README.md?")

        # This should pass - agent should have called read_file
        assertToolCall(self, agent, "read_file", "Should have read the file")

    def test_assert_tool_call_with_write(self):
        """Test that assertToolCall detects write operations"""
        agent, root = self.create_agent({
            "test.txt": "Initial content"
        })

        # Make agent write a file
        agent.message("Create a new file called output.txt with the text 'Hello World'")

        # This should pass - agent should have called write_file
        assertToolCall(self, agent, "write_file", "Should have written a file")

    def test_assert_tool_call_failure(self):
        """Test that assertToolCall fails when tool was not called"""
        agent, root = self.create_agent({
            "test.txt": "Test content"
        })

        # Make agent do something that doesn't involve a specific tool
        agent.message("Hello")

        # This should fail - agent likely didn't call bash
        with self.assertRaises(AssertionError):
            assertToolCall(self, agent, "definitely_not_called_tool")


class TestAssertToolCallMixin(AgentTestMixin, TestCase):
    """Test the AgentTestMixin that adds assertToolCall as a method"""

    def setUp(self):
        self.model = select_chat_model("gpt-oss-20b", 0.1)

    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)
        return AgentHarness(create_agent(self.model, root)), root

    def test_mixin_assert_tool_call(self):
        """Test that mixin assertToolCall works like self.assertX methods"""
        agent, root = self.create_agent({
            "README.md": "Test file"
        })

        # Make agent read a file
        agent.message("What's in README.md?")

        # Use as a method on self (like self.assertEqual, self.assertIn, etc.)
        self.assertToolCall(agent, "read_file", "Should have read the file")

    def test_mixin_assert_tool_call_failure(self):
        """Test that mixin assertToolCall fails appropriately"""
        agent, root = self.create_agent({
            "test.txt": "Test content"
        })

        agent.message("Hello")

        # This should fail
        with self.assertRaises(AssertionError) as cm:
            self.assertToolCall(agent, "nonexistent_tool")

        # Check that the error message includes the list of actual tools called
        self.assertIn("nonexistent_tool", str(cm.exception))
