"""Evaluation test suite for agent behavior.

This module automatically loads environment variables from .dev.env
when imported, allowing evals to run directly without make targets.
"""
import os
from pathlib import Path


def _load_dev_env():
    """Load environment variables from .dev.env if it exists.

    Searches upward from this file to find .dev.env in the project root.
    Only loads if the file exists. Skips comments and empty lines.
    """
    # Start from this file's directory and search upward
    current = Path(__file__).resolve().parent

    while current != current.parent:  # Stop at filesystem root
        env_file = current / '.dev.env'
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    # Parse KEY=VALUE format
                    if '=' in line:
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip()
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        os.environ[key] = value
            return True
        current = current.parent

    return False


# Auto-load .dev.env when this module is imported
_load_dev_env()
