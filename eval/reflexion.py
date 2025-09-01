import re
from langchain_core.messages import HumanMessage
from assist.reflexion_agent import reflexion_graph_v1
from .types import Validation

GRAPH = reflexion_graph_v1()

VALIDATIONS = [
    Validation(
        input={"messages": [HumanMessage(content="Identify the capital of France and provide one fact about it.")]},
        check=re.compile("needs_replan"),
    )
]
