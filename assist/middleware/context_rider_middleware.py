"""Inject the per-turn context rider into the model call.

The client attaches a ``ContextRider`` (when/where the message was sent) to the
turn's ``configurable``; this middleware renders its prose line into an EPHEMERAL
system message for the current model call only — via ``request.override``, so it is
NOT written to the checkpoint (location isn't persisted, and a later turn without a
rider carries no stale context).  Installed on the MAIN agent only — never on the
research/context sub-agents that have web egress, so a location line can't ride out
in an outbound query.  See assist/context_rider.py + docs/2026-06-29-context-rider.org.
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
            request = request.override(
                messages=list(request.messages) + [SystemMessage(content=line)])
        return handler(request)
