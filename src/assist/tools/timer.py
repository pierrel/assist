from __future__ import annotations

import time
from typing import Any, Dict
from langchain_core.tools import BaseTool


class TimerTool(BaseTool):
    """Start, check, and cancel named timers."""

    name: str = "timer"
    description: str = (
        "Manage timers. Args: action ('start', 'status', 'cancel'), "
        "name (str), seconds (float, optional for start)."
    )

    def __init__(self) -> None:
        super().__init__()
        self._timers: Dict[str, float] = {}

    def _run(self, action: str, name: str, seconds: float | None = None) -> str:
        now = time.time()
        if action == "start":
            if seconds is None:
                return "Missing 'seconds' for start"
            self._timers[name] = now + seconds
            return f"Timer '{name}' started for {seconds} seconds"
        if action == "status":
            end = self._timers.get(name)
            if end is None:
                return f"Timer '{name}' not found"
            remaining = end - now
            if remaining <= 0:
                del self._timers[name]
                return f"Timer '{name}' completed"
            return f"{remaining:.1f} seconds remaining"
        if action == "cancel":
            if self._timers.pop(name, None) is None:
                return f"Timer '{name}' not found"
            return f"Timer '{name}' cancelled"
        return "Unknown action. Use 'start', 'status', or 'cancel'"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - sync tool
        raise NotImplementedError


__all__ = ["TimerTool"]
