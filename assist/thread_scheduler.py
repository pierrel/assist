# Copyright (c) 2026, Assist
# Licensed under the MIT License.

"""
Thread Scheduler Module
-----------------------

This module handles:
- Saving thread configurations to JSON files in the `threads` directory.
- Loading thread configurations on startup.
- Triggering starting prompts when a thread is created.
- Triggering scheduled prompts based on cron schedules.
- Persisting thread state between restarts.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
THREADS_DIR = Path("threads")
THREAD_TEMPLATE = {
    "thread_id": str(uuid.uuid4()),
    "name": "New Thread",
    "description": "",
    "created_at": datetime.now().isoformat(),
    "updated_at": datetime.now().isoformat(),
    "status": "active",
    "scheduled_prompt": None,
    "cron_schedule": "0 0 * * *",  # Default: daily at midnight
    "last_scheduled_run": None,
}


def ensure_threads_dir():
    """Ensure the threads directory exists."""
    THREADS_DIR.mkdir(exist_ok=True)


def save_thread_config(thread_id: str, thread_config: Dict[str, Any]) -> bool:
    """
    Save a thread configuration to a JSON file in the threads directory.
    
    Args:
        thread_id: Unique identifier for the thread.
        thread_config: Dictionary containing thread configuration.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        thread_file = THREADS_DIR / f"{thread_id}.json"
        with open(thread_file, "w") as f:
            json.dump(thread_config, f, indent=4)
        logger.info(f"Thread {thread_id} saved to {thread_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save thread {thread_id}: {e}")
        return False


def load_thread_configs() -> List[Dict[str, Any]]:
    """
    Load all thread configurations from the threads directory.
    
    Returns:
        List[Dict[str, Any]]: List of thread configurations.
    """
    thread_configs = []
    try:
        for file in THREADS_DIR.glob("*.json"):
            with open(file, "r") as f:
                try:
                    thread_config = json.load(f)
                    thread_configs.append(thread_config)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON in {file}: {e}")
                    continue
        logger.info(f"Loaded {len(thread_configs)} thread configurations")
        return thread_configs
    except Exception as e:
        logger.error(f"Failed to load thread configurations: {e}")
        return []


from apscheduler.schedulers.background import BackgroundScheduler

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()


def trigger_starting_prompt(thread_id: str, starting_prompt: str) -> bool:
    """
    Trigger the starting prompt for a thread.
    
    Args:
        thread_id: Unique identifier for the thread.
        starting_prompt: The prompt to execute for the thread.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    logger.info(f"Triggering starting prompt for thread {thread_id}: {starting_prompt}")
    print(f"Executing starting prompt: {starting_prompt}")
    return True


def trigger_scheduled_prompt(thread_id: str, scheduled_prompt: str) -> bool:
    """
    Trigger the scheduled prompt for a thread.
    
    Args:
        thread_id: Unique identifier for the thread.
        scheduled_prompt: The prompt to execute on schedule.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    logger.info(f"Triggering scheduled prompt for thread {thread_id} at {datetime.now()}: {scheduled_prompt}")
    print(f"Executing scheduled prompt: {scheduled_prompt}")
    return True


def schedule_thread(thread_id: str, cron_schedule: str, scheduled_prompt: str) -> bool:
    """
    Schedule the thread's scheduled prompt using apscheduler.
    
    Args:
        thread_id: Unique identifier for the thread.
        cron_schedule: Cron schedule string for the prompt.
        scheduled_prompt: The prompt to execute on schedule.
        
    Returns:
        bool: True if scheduling is successful, False otherwise.
    """
    def scheduled_job():
        trigger_scheduled_prompt(thread_id, scheduled_prompt)
    
    try:
        cron_parts = cron_schedule.split()
        
        # Parse cron schedule
        minute = cron_parts[0]
        hour = cron_parts[1]
        day_of_month = cron_parts[2]
        month = cron_parts[3]
        day_of_week = cron_parts[4] if len(cron_parts) > 4 else None
        
        scheduler.add_job(
            scheduled_job,
            "cron",
            minute=minute,
            hour=hour,
            day=day_of_month,
            month=month,
            day_of_week=day_of_week
        )
        logger.info(f"Scheduled thread {thread_id} with cron: {cron_schedule}")
        return True
    except Exception as e:
        logger.error(f"Failed to schedule thread {thread_id}: {e}")
        return False


def create_thread(thread_name: str, description: str = "", scheduled_prompt: Optional[str] = None, cron_schedule: str = "0 0 * * *") -> str:
    """
    Create a new thread with the given configuration.
    
    Args:
        thread_name: Name of the thread.
        description: Description of the thread.
        scheduled_prompt: Optional scheduled prompt.
        cron_schedule: Cron schedule for the scheduled prompt.
        
    Returns:
        str: Thread ID of the newly created thread.
    """
    thread_id = str(uuid.uuid4())
    thread_config = {
        **THREAD_TEMPLATE,
        "thread_id": thread_id,
        "name": thread_name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "scheduled_prompt": scheduled_prompt,
        "cron_schedule": cron_schedule,
    }
    
    if save_thread_config(thread_id, thread_config):
        if scheduled_prompt:
            schedule_thread(thread_id, cron_schedule, scheduled_prompt)
        trigger_starting_prompt(thread_id, f"Starting new thread: {thread_name}")
        logger.info(f"Thread {thread_id} created successfully")
        return thread_id
    else:
        logger.error(f"Failed to create thread {thread_id}")
        return ""


if __name__ == "__main__":
    ensure_threads_dir()
    # Example usage
    thread_id = create_thread(
        thread_name="Example Thread",
        description="An example thread for testing.",
        scheduled_prompt="Run a daily check on this thread.",
        cron_schedule="0 0 * * *",
    )
    logger.info(f"Created thread with ID: {thread_id}")