"""Normal web tool for a routine turn's ordinary new-status choice."""

from __future__ import annotations

from langgraph.config import get_config


def quiet_tools(request_quiet) -> list:
    """Build the visible web tool with a host-owned exact-Run callback."""

    def quiet() -> str:
        """Suppress this turn's ordinary new badge when nothing needs attention."""
        config = (get_config() or {}).get("configurable") or {}
        tid = config.get("thread_id")
        run_id = config.get("web_run_id")
        if not tid or not run_id:
            return "Quiet is available only during a visible web turn."
        try:
            requested = request_quiet(tid, run_id)
        except Exception:
            requested = False
        if not requested:
            return "Couldn't mark this turn quiet; its reply will appear as new."
        return "This turn's ordinary completion will not add a new badge."

    return [quiet]
