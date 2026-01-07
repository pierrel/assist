import os
from typing import Any, Dict
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime
from loguru import logger

# Simple middleware-style hooks to log LLM prompts, tool calls, and responses
# to a per-thread logs.log file. Inspired by console callback, but file-based.

class LoggingMiddleware(AgentMiddleware):
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "logs.log")
        try:
            logger.add(log_path, rotation="10 MB", retention="10 days", enqueue=True, backtrace=False)
        except Exception:
            pass

    def beforeAgent(self, state: AgentState, runtime: Runtime) -> dict[str,Any] | None:
        # Log incoming prompts if present
        msgs = state.get("messages") or []
        logger.info("===== Agent Start =====")
        logger.info("Messages: {}", msgs)
        return None
