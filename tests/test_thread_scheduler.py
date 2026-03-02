# Copyright (c) 2026, Assist
# Licensed under the MIT License.

"""
Test cases for thread scheduler functionality.
"""

import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from pathlib import Path

# Import the thread scheduler module
from assist.thread_scheduler import (
    save_thread_config,
    load_thread_configs,
    trigger_starting_prompt,
    trigger_scheduled_prompt,
    schedule_thread_prompt,
    create_thread,
    ensure_threads_dir,
)


class TestThreadScheduler(unittest.TestCase):
    """Test cases for thread scheduler functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.old_threads_dir = os.getenv("THREADS_DIR")
        os.environ["THREADS_DIR"] = self.test_dir

    def tearDown(self):
        """Clean up test environment."""
        if os.getenv("THREADS_DIR") == self.test_dir:
            del os.environ["THREADS_DIR"]
        if self.old_threads_dir:
            os.environ["THREADS_DIR"] = self.old_threads_dir
        
        # Clean up test directory
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_ensure_threads_dir(self):
        """Test that the threads directory is created."""
        ensure_threads_dir()
        self.assertTrue(os.path.exists(Path(self.test_dir)))

    def test_save_thread_config(self):
        """Test saving a thread configuration."""
        thread_id = "test-thread-1"
        thread_config = {
            "thread_id": thread_id,
            "name": "Test Thread",
            "description": "Test description",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active",
            "scheduled_prompt": "Test scheduled prompt",
            "cron_schedule": "0 0 * * *",
        }
        
        result = save_thread_config(thread_id, thread_config)
        self.assertTrue(result)
        
        thread_file = Path(self.test_dir) / f"{thread_id}.json"
        self.assertTrue(thread_file.exists())
        
        with open(thread_file, "r") as f:
            saved_config = json.load(f)
            self.assertEqual(saved_config["name"], thread_config["name"])

    def test_load_thread_configs(self):
        """Test loading thread configurations."""
        thread_id = "test-thread-2"
        thread_config = {
            "thread_id": thread_id,
            "name": "Loaded Thread",
            "description": "Loaded description",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active",
            "scheduled_prompt": "Loaded scheduled prompt",
            "cron_schedule": "0 0 * * *",
        }
        
        save_thread_config(thread_id, thread_config)
        
        configs = load_thread_configs()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["name"], thread_config["name"])

    @patch("assist.thread_scheduler.trigger_starting_prompt")
    @patch("assist.thread_scheduler.schedule_thread_prompt")
    def test_create_thread(self, mock_schedule, mock_trigger):
        """Test creating a thread with scheduler."""
        mock_schedule.return_value = True
        mock_trigger.return_value = True
        
        thread_id = create_thread(
            thread_name="Test Thread",
            description="Test description",
            scheduled_prompt="Test scheduled prompt",
            cron_schedule="0 0 * * *",
        )
        
        self.assertIsNotNone(thread_id)
        mock_trigger.assert_called_once()
        mock_schedule.assert_called_once()

    @patch("assist.thread_scheduler.trigger_scheduled_prompt")
    def test_trigger_scheduled_prompt(self, mock_trigger):
        """Test triggering a scheduled prompt."""
        thread_id = "test-thread-3"
        scheduled_prompt = "Scheduled prompt"
        
        result = trigger_scheduled_prompt(thread_id, scheduled_prompt)
        self.assertTrue(result)
        mock_trigger.assert_called_once_with(thread_id, scheduled_prompt)


if __name__ == "__main__":
    unittest.main()