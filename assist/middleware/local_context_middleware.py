import os
from typing import Any, Dict
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime


class UserContextMiddleware(AgentMiddleware):
    """Inserts context about the local filesystem and any memories into the system prompt"""
    def __init__(self, working_dir: str):
        self.working_dir = working_dir
    
