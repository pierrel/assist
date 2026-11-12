"""Offline browser observations expose current pages and turn boundaries."""
from edd.eval.test_browser_agent import _BrowserSite
import pytest


@pytest.mark.parametrize("scenario,role,name", [
    ("v1-form", "button", "Review"), ("approval", "link", "Latest report")])
def test_selected_wait_returns_current_visible_control(tmp_path, scenario, role, name):
    site = _BrowserSite(scenario, str(tmp_path))
    opened = site.command("open", url=f"https://{scenario}.fern.example/")
    page_id = opened["result"]["page_id"]
    assert site.command("wait", page_id=page_id, role=role, name=name) == opened
    assert "error" in site.command("wait", page_id=page_id, role=role, name="Missing")
    site.command("close", page_id=page_id)
    assert "error" in site.command("wait", page_id=page_id, role=role, name=name)


def test_approval_allowed_preflight_requires_current_page_but_denial_does_not(tmp_path):
    site = _BrowserSite("approval", str(tmp_path))
    partner = "https://partner.fern.example/report"
    denied = site.command("preflight", url=partner, page_id="missing-page")
    assert not denied["result"]["top_level"]["allowed"]
    site.approved = True
    assert "error" in site.command("preflight", url=partner, page_id="missing-page")
    assert site.command("preflight", url=partner)["result"]["top_level"]["allowed"]
    opened = site.command("open", url=partner)
    page_id = opened["result"]["page_id"]
    assert site.command("preflight", url=partner, page_id=page_id)["result"]["top_level"]["allowed"]
    site.command("close", page_id=page_id)
    assert "error" in site.command("preflight", url=partner, page_id=page_id)


@pytest.mark.parametrize("scenario", ["v1-form", "approval", "v1-auth"])
def test_selected_close_closes_only_current_page_and_preserves_auth(tmp_path, scenario):
    site = _BrowserSite(scenario, str(tmp_path))
    site.auth_cookie = True
    url = f"https://{scenario}.fern.example/"
    opened = site.command("open", url=url)
    page_id = opened["result"]["page_id"]
    if scenario == "v1-form":
        opened = site.command("act", page_id=page_id, snapshot_id="form-1",
                              action="fill", target={"ref": "destination", "text": "Portland"})
    assert "error" in site.command("close", page_id="missing-page")
    assert site.command("observe", page_id=page_id) == opened
    turn = site.turn
    closed = site.command("close", page_id=page_id)
    assert closed == {"result": {"closed_page_id": page_id, "pages": [],
                                 "active_page_id": None}}
    assert site.turn == turn and site.auth_cookie
    assert "error" in site.command("observe", page_id=page_id)
    assert "error" in site.command("preflight", url=url, page_id=page_id)
    assert "error" in site.command("act", page_id=page_id, snapshot_id="form-1",
                                   action="click", target={"ref": "review"})
    assert "error" in site.command("close", page_id=page_id)
    assert "error" in site.command("open", url=url, reuse_page_id=page_id)
    reopened = site.command("open", url=url)
    assert reopened["result"]["page_id"] != page_id
    assert "error" in site.command("observe", page_id=page_id)
    if scenario == "v1-form":
        assert "Destination: blank" in reopened["result"]["snapshot"]
        site.command("act", page_id=reopened["result"]["page_id"], snapshot_id="form-1",
                     action="fill", target={"ref": "destination", "text": "Portland"})
    reused = site.command("open", url=url, reuse_page_id=reopened["result"]["page_id"])
    assert reused["result"]["page_id"] == reopened["result"]["page_id"]
    if scenario == "v1-form":
        assert "Destination: blank" in reused["result"]["snapshot"]


def test_private_report_observation_keeps_current_page_and_turn_boundary(tmp_path):
    site = _BrowserSite("v1-auth", str(tmp_path))
    site.auth_cookie = True
    opened = site.command("open", url="https://v1-auth.fern.example/")
    assert site.command("observe", page_id="page-1") == opened
    assert "October Reliability Review" in opened["result"]["snapshot"]
    assert "error" in site.command("observe", page_id="missing-page")
    site.end_turn()
    assert "error" in site.command("observe", page_id="page-1")
    reopened = site.command("open", url="https://v1-auth.fern.example/")
    assert site.command("observe", page_id="page-2") == reopened
    assert "error" in site.command("observe", page_id="page-1")


def test_approval_observation_keeps_current_report_and_turn_boundary(tmp_path):
    site = _BrowserSite("approval", str(tmp_path))
    first = site.command("open", url="https://approval.fern.example/")
    page_id = first["result"]["page_id"]
    assert site.command("observe", page_id=page_id) == first
    site.end_turn()
    assert "error" in site.command("observe", page_id=page_id)
    assert "error" in site.command("act", page_id=page_id, snapshot_id="first",
                                   action="click", target={"ref": "partner-report"})
    site.approved = True
    report_url = "https://partner.fern.example/report"
    report = site.command("open", url=report_url)
    assert report["result"]["page_id"] != page_id
    assert report["result"]["url"] == report_url
    assert "October Reliability Review" in report["result"]["snapshot"]
    assert site.command("observe", page_id=report["result"]["page_id"]) == report
    assert "error" in site.command("observe", page_id=page_id)
    assert "error" in site.command("act", page_id=page_id, snapshot_id="approved",
                                   action="click", target={"ref": "partner-report"})
    home = site.command("open", url="https://approval.fern.example/")
    clicked = site.command("act", page_id=home["result"]["page_id"],
                           snapshot_id=home["result"]["snapshot_id"],
                           action="click", target={"ref": "partner-report"})
    assert clicked["result"]["url"] == report_url
    assert "October Reliability Review" in clicked["result"]["snapshot"]
    assert site.command("observe", page_id=clicked["result"]["page_id"]) == clicked


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
    assert site.current_page["result"]["snapshot_id"] == "form-1"
    reviewed = site.command("act", page_id="page-1", snapshot_id="form-1",
                            action="click", target={"ref": "review"})
    assert site.command("observe", page_id="page-1") == reviewed
    site.end_turn()
    assert "error" in site.command("observe", page_id="page-1")


@pytest.mark.parametrize("page_id,close_after_review", [
    ("page-1", False), ("missing-page", False), ("page-1", True)])
def test_continuity_eval_requires_a_successful_review(monkeypatch, page_id, close_after_review):
    import time
    from threading import Event
    from langchain_core.messages import AIMessage
    from tests.middleware.test_skills_kernel import _RecordingModel
    from edd.eval.test_browser_agent import TestBrowserAgent
    from assist.thread_queue import active_handle

    def call(name, args, index):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(index)}])
    responses = [
        call("load_skill", {"name": "browse-website"}, 1),
        call("browser_open", {"url": "https://v1-form.fern.example/"}, 2),
        call("browser_act", {"page_id": "page-1", "snapshot_id": "form-1", "action": "fill",
                             "target": {"ref": "destination", "text": "Portland"}}, 3),
        call("browser_act", {"page_id": page_id, "snapshot_id": "form-1", "action": "click",
                             "target": {"ref": "review"}}, 4)]
    if close_after_review:
        responses.append(call("browser_close", {"page_id": page_id}, 5))
    responses.append(AIMessage(content="Portland draft reviewed."))
    model = _RecordingModel(responses=responses)
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
