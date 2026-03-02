#!/usr/bin/env python3

import json
import logging
import os
import uuid
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Thread configurations directory
THREADS_DIR = "/workspace/threads"


def save_thread_config(thread_config):
    """Save thread configuration to a JSON file in the threads directory."""
    thread_id = thread_config["thread_id"]
    file_path = os.path.join(THREADS_DIR, f"{thread_id}.json")
    
    with open(file_path, "w") as f:
        json.dump(thread_config, f, indent=4)
    
    logger.info(f"Thread {thread_id} saved successfully.")


def load_thread_configs():
    """Load all thread configurations from the threads directory."""
    thread_configs = []
    
    for filename in os.listdir(THREADS_DIR):
        if filename.endswith(".json"):
            try:
                file_path = os.path.join(THREADS_DIR, filename)
                with open(file_path, "r") as f:
                    thread_config = json.load(f)
                    thread_configs.append(thread_config)
                    logger.info(f"Loaded thread configuration: {thread_config['thread_id']}")
            except Exception as e:
                logger.error(f"Error loading thread configuration {filename}: {e}")
    
    return thread_configs


def trigger_starting_prompt(thread_config):
    """Trigger the starting prompt for a thread."""
    logger.info(f"Triggering starting prompt for thread {thread_config['thread_id']}: {thread_config['starting_prompt']}")
    # Add logic to execute the starting prompt here
    print(f"Executing starting prompt: {thread_config['starting_prompt']}")


def trigger_scheduled_prompt(thread_config):
    """Trigger the scheduled prompt for a thread."""
    logger.info(f"Triggering scheduled prompt for thread {thread_config['thread_id']} at {datetime.now()}: {thread_config['scheduled_prompt']}")
    # Add logic to execute the scheduled prompt here
    print(f"Executing scheduled prompt: {thread_config['scheduled_prompt']}")


def schedule_thread(thread_config):
    """Schedule the thread's scheduled prompt using apscheduler."""
    def scheduled_job():
        trigger_scheduled_prompt(thread_config)
    
    cron_schedule = thread_config["cron_schedule"]
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
    
    logger.info(f"Scheduled thread {thread_config['thread_id']} with cron: {cron_schedule}")


def main():
    """Main function to load and schedule threads."""
    # Load thread configurations
    thread_configs = load_thread_configs()
    
    # Trigger starting prompts
    for thread in thread_configs:
        trigger_starting_prompt(thread)
        schedule_thread(thread)
    
    # Keep the scheduler running
    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler shutdown.")


if __name__ == "__main__":
    main()