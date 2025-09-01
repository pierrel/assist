import re
from assist.reflexion_agent import plan_checker_graph_v1, Plan, Step, StepResolution
from .types import Validation

GRAPH = plan_checker_graph_v1()

PLAN = Plan(
    goal="Prepare tea",
    steps=[
        Step(action="Boil water", objective="Boil water"),
        Step(action="Steep tea", objective="Steep tea"),
    ],
    assumptions=[],
    risks=[],
)

STATE = {
    "messages": [],
    "plan": PLAN,
    "step_index": 1,
    "history": [StepResolution(action="Boil water", objective="Boil water", resolution="done")],
    "needs_replan": False,
    "learnings": [],
}

VALIDATIONS = [
    Validation(
        input=STATE,
        check=re.compile("needs_replan"),
    )
]
