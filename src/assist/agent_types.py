from __future__ import annotations

from typing import List

from pydantic import BaseModel
from langchain_core.messages import BaseMessage


class AgentInvokeResult(BaseModel):
    """Standard structure returned by agents."""

    messages: List[BaseMessage]
