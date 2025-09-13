"""Utilities for studying large files by summarizing them chunk by chunk."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from assist.model_manager import select_chat_model, get_context_limit

# Leave room for prompts, user context and running summary
CONTEXT_FRACTION = 0.8


def _summarize_chunk(
    llm: BaseChatModel,
    chunk: str,
    task: str,
    request: str,
    prior: str,
) -> str:
    """Return a summary for ``chunk`` building on ``prior``."""

    messages = [
        SystemMessage(
            content=(
                "You are a study agent. Summarize file chunks so the user can "
                "complete their task."
            )
        ),
        HumanMessage(
            content=(
                f"Original request:\n{request}\n\n"
                f"Current task:\n{task}\n\n"
                f"Previous summary:\n{prior}\n\n"
                f"File chunk:\n{chunk}\n\n"
                "Provide an updated concise summary relevant to the task."
            )
        ),
    ]
    resp = llm.invoke(messages)
    return getattr(resp, "content", str(resp))


def study_file(
    path: Path | str,
    task: str = "",
    request: str = "",
    llm: Optional[BaseChatModel] = None,
) -> str:
    """Return the contents of ``path`` or a summary if it's too large.

    The file is read in chunks sized to the language model's context window. A
    running summary is updated after each chunk so that the final result fits
    within the model's limits and remains focused on the ``task`` and
    ``request``.
    """

    p = Path(path)
    if llm is None:
        model = os.getenv("STUDY_AGENT_MODEL", "gpt-4o-mini")
        llm = select_chat_model(model, temperature=0)

    limit = get_context_limit(llm)
    chunk_size = int(limit * CONTEXT_FRACTION)

    text = p.read_text(encoding="utf-8", errors="ignore")
    if len(text) <= chunk_size:
        return text

    summary = ""
    start = 0
    end = len(text)
    while start < end:
        chunk = text[start : start + chunk_size]
        summary = _summarize_chunk(llm, chunk, task, request, summary)
        start += chunk_size
    return summary


__all__ = ["study_file"]

