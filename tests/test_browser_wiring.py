"""The typed browser surface belongs only to the visible web main agent."""
from unittest.mock import patch

from assist.thread_manager import ThreadManager


def test_browser_tools_and_skill_route_are_web_main_only(tmp_path):
    manager = ThreadManager(str(tmp_path))
    manager._model = object()
    (tmp_path / "thread" / "domain").mkdir(parents=True)

    def browser_open(url: str):
        return url

    with patch("assist.thread_manager.Thread",
               side_effect=lambda *_args, **kwargs: kwargs["spec"]):
        main = manager.get("thread", browser_tools=(browser_open,))
        triage = manager.get("thread", triage=True,
                            browser_tools=(browser_open,))
        delegate = manager.get("thread", assistant_id="delegate-agent",
                              browser_tools=(browser_open,))

    assert browser_open in main.tools
    assert "/browser-skill/" in main.skill_sources
    assert browser_open not in triage.tools
    assert "/browser-skill/" not in triage.skill_sources
    assert browser_open not in delegate.tools
    assert "/browser-skill/" not in delegate.skill_sources
