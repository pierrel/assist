"""Web tool for changing the current thread's displayed description."""
from __future__ import annotations

from langgraph.config import get_config


def thread_description_tools(set_description):
    """Build the current-thread tool around the web app's description writer."""

    def update_thread_description(description: str) -> str:
        """Set this thread's description/title when the user asks to rename it.

        Pass the requested one-line title. This changes only this thread.
        """
        tid = ((get_config() or {}).get("configurable") or {}).get("thread_id")
        if not tid:
            return "This tool is only available in a thread."

        description = " ".join(description.split())[:120]
        if not description:
            return "The description cannot be empty; the current description was kept."

        try:
            set_description(tid, description)
        except FileNotFoundError:
            return "This thread no longer exists."
        return "Updated this thread's description."

    return [update_thread_description]
