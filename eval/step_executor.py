import re
from langchain_core.messages import HumanMessage
from assist.reflexion_agent import step_executor_graph_v1, Plan, Step
from .types import Validation

GRAPH = step_executor_graph_v1()

PLAN = Plan(
    goal="Greet user",
    steps=[Step(action="Greet the user politely", objective="Provide a friendly greeting")],
    assumptions=[],
    risks=[],
)

STATE = {
    "messages": [HumanMessage(content="Execute the plan")],
    "plan": PLAN,
    "step_index": 0,
    "history": [],
    "needs_replan": False,
    "learnings": [],
}

VALIDATIONS = [
    Validation(
        input=STATE,
        check=re.compile("resolution"),
    )
]
