import os
import tempfile
import shutil

from unittest import TestCase
from langchain_core.messages import HumanMessage, AIMessage

from assist.thread import ThreadManager, Thread


class TestThreadE2E(TestCase):
    """End-to-end tests for Thread functionality (no domain management)."""

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.thread_manager = ThreadManager(self.working_dir)

    def tearDown(self):
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)

    def test_thread_creation(self):
        """Test that threads can be created."""
        thread = self.thread_manager.new()
        self.assertIsNotNone(thread.thread_id)
        self.assertTrue(os.path.isdir(thread.working_dir))

    def test_message_and_response(self):
        """Test basic message sending and receiving."""
        thread = self.thread_manager.new()
        resp = thread.message("What is 2+2?")
        self.assertIsInstance(resp, str)
        self.assertTrue(len(resp) > 0)

    def test_message_history(self):
        """Test that messages are persisted in thread history."""
        thread = self.thread_manager.new()

        # Send a message
        thread.message("Hello, my name is Alice.")

        # Send another message
        thread.message("What is my name?")

        # Check message history
        messages = thread.get_messages()
        self.assertGreaterEqual(len(messages), 2)

        # Should have at least one user message
        user_messages = [m for m in messages if m.get("role") == "user"]
        self.assertGreaterEqual(len(user_messages), 2)

        # Should have at least one assistant message
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        self.assertGreaterEqual(len(assistant_messages), 1)

    def test_thread_persistence(self):
        """Test that threads can be retrieved after creation."""
        thread1 = self.thread_manager.new()
        tid = thread1.thread_id

        thread1.message("Remember this: banana")

        # Retrieve the same thread
        thread2 = self.thread_manager.get(tid)
        self.assertEqual(thread1.thread_id, thread2.thread_id)

        # Message history should be preserved
        messages = thread2.get_messages()
        user_messages = [m for m in messages if m.get("role") == "user"]
        self.assertGreaterEqual(len(user_messages), 1)

    def test_multiple_threads(self):
        """Test that multiple threads can coexist independently."""
        thread1 = self.thread_manager.new()
        thread2 = self.thread_manager.new()

        # Different thread IDs
        self.assertNotEqual(thread1.thread_id, thread2.thread_id)

        # Send different messages to each
        thread1.message("My favorite color is blue")
        thread2.message("My favorite color is red")

        # Check they have independent histories
        msgs1 = thread1.get_messages()
        msgs2 = thread2.get_messages()

        # Find the user messages
        user_msgs1 = [m.get("content", "") for m in msgs1 if m.get("role") == "user"]
        user_msgs2 = [m.get("content", "") for m in msgs2 if m.get("role") == "user"]

        # Check that the messages are different
        self.assertTrue(any("blue" in msg for msg in user_msgs1))
        self.assertTrue(any("red" in msg for msg in user_msgs2))

    def test_list_threads(self):
        """Test listing all threads."""
        # Create a few threads
        thread1 = self.thread_manager.new()
        thread2 = self.thread_manager.new()

        # List should include both
        thread_ids = self.thread_manager.list()
        self.assertIn(thread1.thread_id, thread_ids)
        self.assertIn(thread2.thread_id, thread_ids)

    def test_description_generation(self):
        """Test that thread descriptions can be generated."""
        thread = self.thread_manager.new()
        thread.message("I want to learn about quantum physics")

        # Should be able to generate a description
        description = thread.description()
        self.assertIsInstance(description, str)
        self.assertTrue(len(description) > 0)
