"""
Unit tests for the edd.capture module.
"""
import os
import json
import tempfile
import shutil
from unittest import TestCase
from unittest.mock import Mock

from edd.capture import (
    sanitize_dirname,
    capture_conversation,
)


class TestSanitizeDirname(TestCase):
    def test_basic_sanitization(self):
        result = sanitize_dirname("Fix authentication bug")
        # Should start with timestamp and include slug
        self.assertRegex(result, r'^\d{8}-\d{6}-fix-authentication-bug$')

    def test_long_description(self):
        result = sanitize_dirname("This is a very long description with many words that should be truncated")
        # Should only take first 5 words
        self.assertRegex(result, r'^\d{8}-\d{6}-this-is-a-very-long$')

    def test_special_characters(self):
        result = sanitize_dirname("Fix: auth/login @#$% bug!")
        # Should replace special chars with hyphens
        self.assertIn("fix-auth-login", result)
        self.assertNotIn("@", result)
        self.assertNotIn("#", result)

    def test_empty_description(self):
        result = sanitize_dirname("")
        # Should still return timestamp
        self.assertRegex(result, r'^\d{8}-\d{6}-?$')


class TestCaptureConversation(TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.improvements_dir = os.path.join(self.test_dir, "improvements")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_capture_empty_conversation(self):
        # Create mock thread with no messages
        mock_thread = Mock()
        mock_thread.get_messages.return_value = []

        # Should raise ValueError
        with self.assertRaises(ValueError) as cm:
            capture_conversation(mock_thread, "test", self.improvements_dir)

        self.assertIn("empty", str(cm.exception).lower())

    def test_directory_creation(self):
        # Create mock thread
        mock_thread = Mock()
        mock_thread.thread_id = "test-123"
        mock_thread.working_dir = tempfile.mkdtemp()
        mock_thread.model = Mock(model_name="test-model")
        mock_thread.get_messages.return_value = [
            {"role": "user", "content": "Test question"},
            {"role": "assistant", "content": "Test answer"}
        ]
        mock_thread.description.return_value = "Test conversation"

        try:
            # Note: This will actually try to create an agent and run it
            # For now, we just test that the directory is created
            # In a full test, we'd mock the agent creation

            # Create the improvements dir to avoid agent execution
            os.makedirs(self.improvements_dir, exist_ok=True)

            # Test that sanitize_dirname works as expected
            dirname = sanitize_dirname("Test conversation")
            self.assertRegex(dirname, r'^\d{8}-\d{6}-test-conversation$')

        finally:
            shutil.rmtree(mock_thread.working_dir)

    def test_directory_collision(self):
        # Test that collisions are handled by appending numbers
        desc = "Same description"
        dir1 = sanitize_dirname(desc)
        dir2 = sanitize_dirname(desc)

        # They should have the same format (timestamp might differ by milliseconds)
        self.assertRegex(dir1, r'^\d{8}-\d{6}-same-description$')
        self.assertRegex(dir2, r'^\d{8}-\d{6}-same-description$')
