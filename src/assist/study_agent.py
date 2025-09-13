from __future__ import annotations

"""Agent for studying resources and collecting notes relevant to an objective."""

from pathlib import Path
from typing import Dict, List, TypedDict, Callable

import requests
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END

from assist.promptable import base_prompt_for


class StudyState(TypedDict):
    """State for the study graph."""

    objective: str
    resources: List[str]
    index: int
    notes: List[str]
    output_path: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_resource(path: str) -> str:
    """Return the text content of ``path`` which may be a file or URL."""

    if path.startswith("http://") or path.startswith("https://"):
        resp = requests.get(path, timeout=10)
        resp.raise_for_status()
        return resp.text
    return Path(path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------

def build_study_node(llm: Runnable) -> Callable[[StudyState], Dict[str, object]]:
    def study_node(state: StudyState) -> Dict[str, object]:
        idx = state["index"]
        source = state["resources"][idx]
        content = _read_resource(source)
        messages: List[BaseMessage] = [
            SystemMessage(content=base_prompt_for("study_agent/extract_system.txt")),
            HumanMessage(
                content=base_prompt_for(
                    "study_agent/extract_user.txt",
                    objective=state["objective"],
                    content=content,
                    source=source,
                )
            ),
        ]
        result = llm.invoke(messages)
        if isinstance(result, AIMessage):
            note = result.content
        else:  # pragma: no cover - defensive
            note = str(result)
        state["notes"].append(f"# {source}\n{note}")
        state["index"] = idx + 1
        return state

    return study_node


def build_save_node() -> Callable[[StudyState], Dict[str, object]]:
    def save_node(state: StudyState) -> Dict[str, object]:
        out = Path(state["output_path"])
        out.write_text("\n\n".join(state["notes"]), encoding="utf-8")
        return state

    return save_node


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_study_graph(llm: Runnable) -> Runnable:
    """Return a graph that studies resources and writes notes to a file."""

    graph = StateGraph(StudyState)
    graph.add_node("study", build_study_node(llm))
    graph.add_node("save", build_save_node())

    def cond(state: StudyState) -> str:
        return "study" if state["index"] < len(state["resources"]) else "save"

    graph.add_conditional_edges("study", cond)
    graph.set_entry_point("study")
    graph.add_edge("save", END)
    return graph.compile()


__all__ = ["StudyState", "build_study_graph"]
