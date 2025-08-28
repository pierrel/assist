from langchain_core.runnables import RunnableLambda
from assist.reflexion_agent import build_plan_check_node, _default_llm


def plan_checker_runnable_v1():
    """Runnable wrapper around the plan check node."""
    node = build_plan_check_node(_default_llm(), [])
    return RunnableLambda(node)
