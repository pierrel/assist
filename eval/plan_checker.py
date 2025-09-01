from assist.reflexion_agent import build_plan_check_node, Plan, Step, StepResolution, ReflexionState
from eval.types import Validation
from langchain_openai import ChatOpenAI

def check_needs_replan(state: ReflexionState) -> bool:
    return state["needs_replan"] is False

GRAPH = build_plan_check_node(ChatOpenAI(model="gpt-4o-mini"),
                              [])

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
