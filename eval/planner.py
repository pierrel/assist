import re
from langchain_core.messages import HumanMessage
from assist.reflexion_agent import planner_graph_v1
from .types import Validation

GRAPH = planner_graph_v1()

VALIDATIONS = [
    Validation(
        input={"messages": [HumanMessage(content="Make a short plan for brewing a cup of tea.")]},
        check=re.compile("steps"),
    )
]
