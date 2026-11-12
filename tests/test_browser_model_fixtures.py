"""Offline browser observations must expose the retained form's real state."""
from edd.eval.test_browser_agent import _BrowserSite
import pytest


def test_form_observation_keeps_values_targets_and_page_boundary(tmp_path):
    site = _BrowserSite("v1-form", str(tmp_path))
    site.command("open", url="https://v1-form.fern.example/")
    site.command("act", page_id="page-1", snapshot_id="form-1", action="fill",
                 target={"ref": "destination", "text": "Portland"})
    observed = site.command("observe", page_id="page-1")["result"]
    assert observed["page_id"] == "page-1" and observed["snapshot_id"] == "form-1"
    assert "Portland" in observed["snapshot"]
    assert any(target["ref"] == "review" for target in observed["targets"])
    failed = site.command("act", page_id="missing-page", snapshot_id="form-1",
                          action="click", target={"ref": "review"})
    assert "error" in failed
    assert site.form_page["result"]["snapshot_id"] == "form-1"
    reviewed = site.command("act", page_id="page-1", snapshot_id="form-1",
                            action="click", target={"ref": "review"})
    assert site.command("observe", page_id="page-1") == reviewed
    site.end_turn()
    assert "error" in site.command("observe", page_id="page-1")


@pytest.mark.parametrize("page_id", ["page-1", "missing-page"])
def test_continuity_eval_requires_a_successful_review(monkeypatch, page_id):
    import time
    from threading import Event
    from langchain_core.messages import AIMessage
    from tests.middleware.test_skills_kernel import _RecordingModel
    from edd.eval.test_browser_agent import TestBrowserAgent
    from assist.thread_queue import active_handle

    def call(name, args, index):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(index)}])
    model = _RecordingModel(responses=[
        call("load_skill", {"name": "browse-website"}, 1),
        call("browser_open", {"url": "https://v1-form.fern.example/"}, 2),
        call("browser_act", {"page_id": "page-1", "snapshot_id": "form-1", "action": "fill",
                             "target": {"ref": "destination", "text": "Portland"}}, 3),
        call("browser_act", {"page_id": page_id, "snapshot_id": "form-1", "action": "click",
                             "target": {"ref": "review"}}, 4),
        AIMessage(content="Portland draft reviewed.")])
    original = _RecordingModel._generate
    calls = 0
    def generate(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            handle = active_handle()
            deadline = time.monotonic() + 2
            while not handle.pause_requested and time.monotonic() < deadline:
                Event().wait(0.001)
            assert handle.pause_requested
        return original(self, *args, **kwargs)
    monkeypatch.setattr(_RecordingModel, "_generate", generate)
    monkeypatch.setattr(TestBrowserAgent, "model", model, raising=False)
    case = TestBrowserAgent("test_v1_form_continues_after_fair_yield")
    if page_id == "missing-page":
        with pytest.raises(AssertionError, match="old page is closed"):
            case.test_v1_form_continues_after_fair_yield()
    else:
        case.test_v1_form_continues_after_fair_yield()
