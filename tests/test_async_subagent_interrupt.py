"""Parent-child suspension is gone; user-to-main interjection remains separate."""
import inspect

from assist import async_subagents
from assist.middleware.interjection import InterjectionMiddleware


def test_async_task_tools_do_not_suspend_the_parent_graph():
    source = inspect.getsource(async_subagents)
    assert "langgraph.types" not in source
    assert "interrupt(" not in source
    assert "Command(resume" not in source


def test_main_interjection_middleware_still_exists():
    assert InterjectionMiddleware.__name__ == "InterjectionMiddleware"
