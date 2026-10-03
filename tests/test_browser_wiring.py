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


def test_managed_main_and_child_sandboxes_bind_thread_authority(tmp_path, monkeypatch):
    from manage.web import state
    from assist.sandbox_manager import SandboxManager

    monkeypatch.setattr(state.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(state, "_get_domain_manager", lambda _tid: None)
    calls = []
    monkeypatch.setattr(SandboxManager, "get_sandbox_backend",
                        lambda work_dir, **kwargs: calls.append((work_dir, kwargs)))
    for include_agent in (True, False):
        state._get_sandbox_backend("thread", include_agent=include_agent)
    assert len(calls) == 2
    for work_dir, kwargs in calls:
        assert work_dir == str(tmp_path / "thread" / "domain")
        assert kwargs["thread_scope"] == (str(tmp_path), "thread")
