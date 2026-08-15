"""Inject the per-turn context rider into the model call.

The client attaches a ``ContextRider`` (when the message was sent) to the turn's
``configurable``; this middleware renders its time line into an EPHEMERAL system
message for the current model call only — via ``request.override``, so it is NOT
written to the checkpoint. Browser location is intentionally excluded from the
rider prose and is available only through the private current-location tool.
Installed on main and delegate Assist graphs, but never on specialists. The web
delegate currently receives no rider, making this middleware inert there. See
assist/context_rider.py + docs/2026-08-15-location-context-tool.org.
"""
from __future__ import annotations

import logging
from typing import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.config import get_config

from assist.context_rider import CONTEXT_RIDER_KEY

logger = logging.getLogger(__name__)


class ContextRiderMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | AIMessage],
    ) -> ModelResponse | AIMessage:
        line = None
        try:
            # The run config (incl. `configurable`) is NOT on request.runtime in
            # this langchain — read it via langgraph's get_config() (the supported
            # accessor; runtime.config does not exist).
            cfg = get_config() or {}
            rider = (cfg.get("configurable") or {}).get(CONTEXT_RIDER_KEY)
            if rider is not None:
                line = rider.prose_line()
        except Exception as e:  # never break a turn over context injection
            logger.debug("ContextRiderMiddleware: skipped (%s)", e)
            line = None
        if line:
            # Fold the context into the SYSTEM message, not a trailing message —
            # the Qwen chat template rejects a system message anywhere but the
            # start ("System message must be at the beginning").
            base = request.system_message
            merged = f"{base.content}\n\n{line}" if base is not None else line
            request = request.override(system_message=SystemMessage(content=merged))
        return handler(request)
