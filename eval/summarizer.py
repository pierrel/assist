import re
from langgraph.graph import StateGraph, END
from assist.reflexion_agent import _default_llm, build_summarize_node, StepResolution
from .types import Validation

_llm = _default_llm()
_graph = StateGraph(dict)
_graph.add_node("sum", build_summarize_node(_llm, []))
_graph.set_entry_point("sum")
_graph.add_edge("sum", END)
GRAPH = _graph.compile()

STATE = {
    "messages": [],
    "history": [
        StepResolution(action="Greet", objective="Say hi", resolution="Hi there!"),
        StepResolution(action="Share fact", objective="Inform user", resolution="Tea originated in China."),
    ],
}

VALIDATIONS = [
    Validation(
        input=STATE,
        check=re.compile("messages"),
    )
]
