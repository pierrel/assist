"""Offline browser fixtures preserve pages and legal turn continuations."""
from edd.eval.test_browser_agent import _BrowserSite
import pytest


@pytest.mark.parametrize("journey", ["auth", "approval_before", "approval_after"])
def test_selected_journey_completes_legal_context_work(monkeypatch, journey):
    import hashlib
    import json
    from langchain_core.messages import AIMessage
    from tests.middleware.test_skills_kernel import _RecordingModel
    from edd.eval.test_browser_agent import TestBrowserAgent
    from edd.eval.test_async_subagents import reset_task_fixture
    from edd.eval import test_async_subagents as tasks

    reset_task_fixture()
    task_results = []
    original_result = tasks._RESULT.func
    def result(task_id):
        response = original_result(task_id)
        task_results.append(json.loads(response))
        return response
    monkeypatch.setattr(tasks._RESULT, "func", result)
    def call(name, args, index):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(index)}])
    description = "Inspect the local workspace for relevant context."
    task_id = "task-eval-" + hashlib.sha256(
        f"1\0context-agent\0{description}".encode()).hexdigest()[:12]
    context = [call("start_async_task", {"description": description,
                                        "subagent_type": "context-agent"}, 101),
               AIMessage(content="I started the context task."),
               call("get_async_task_result", {"task_id": task_id}, 102)]
    if journey == "auth":
        responses = context + [
            call("load_skill", {"name": "browse-website"}, 1),
            call("browser_open", {"url": "https://v1-auth.fern.example/"}, 2),
            call("browser_close", {"page_id": "page-1"}, 3),
            AIMessage(content="October Reliability Review")]
        method = "test_v1_one_turn_completion"
    else:
        first = [
            call("load_skill", {"name": "browse-website"}, 1),
            call("browser_open", {"url": "https://approval.fern.example/"}, 2),
            call("browser_preflight", {"url": "https://partner.fern.example/report",
                                       "page_id": "page-1"}, 3),
            call("load_skill", {"name": "egress"}, 4),
            call("request_egress", {"host": "partner.fern.example", "port": 443,
                                    "task": "Read the latest partner report."}, 5),
            AIMessage(content="Approval is needed for the partner report.")]
        second = [
            call("load_skill", {"name": "browse-website"}, 10),
            call("browser_open", {"url": "https://partner.fern.example/report"}, 11),
            call("browser_close", {"page_id": "page-2"}, 12),
            AIMessage(content="October Reliability Review")]
        responses = (context + first + second if journey == "approval_before"
                     else first + context + second)
        method = "test_v1_approval_pause_destroys_pages_then_reopens_with_state"
    monkeypatch.setattr(TestBrowserAgent, "model", _RecordingModel(responses=responses), raising=False)
    case = TestBrowserAgent(method)
    getattr(case, method)()
    assert task_results and all("README.org" in item["result"] for item in task_results)
    assert not tasks._TASK_STATUSES


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


@pytest.mark.parametrize("page_id,close_after_review,context_phase", [
    ("page-1", False, None), ("missing-page", False, None), ("page-1", True, None),
    pytest.param("page-1", False, "before", id="context_before_browser"),
    pytest.param("page-1", False, "after", id="context_after_pause")])
def test_continuity_eval_requires_a_successful_review(
        monkeypatch, page_id, close_after_review, context_phase):
    import hashlib
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
    if context_phase:
        description = "Inspect trip notes in the local workspace."
        task_id = "task-eval-" + hashlib.sha256(
            f"1\0context-agent\0{description}".encode()).hexdigest()[:12]
        task_responses = [
            call("start_async_task", {"description": description,
                                      "subagent_type": "context-agent"}, 101),
            AIMessage(content="I started the context task."),
            call("get_async_task_result", {"task_id": task_id}, 102)]
        index = 0 if context_phase == "before" else 3
        responses[index:index] = task_responses
    if close_after_review:
        responses.append(call("browser_close", {"page_id": page_id}, 5))
    responses.append(AIMessage(content="Portland draft reviewed."))
    model = _RecordingModel(responses=responses)
    original = _RecordingModel._generate
    pause_waited = False
    def generate(self, *args, **kwargs):
        nonlocal pause_waited
        upcoming = self.responses[self.i]
        if not pause_waited and any((tool["name"] == "browser_act" and tool["args"].get("action") == "click")
               or (context_phase == "after" and tool["name"] == "start_async_task")
               for tool in upcoming.tool_calls):
            handle = active_handle()
            deadline = time.monotonic() + 2
            while not handle.pause_requested and time.monotonic() < deadline:
                Event().wait(0.001)
            assert handle.pause_requested
            pause_waited = True
        return original(self, *args, **kwargs)
    monkeypatch.setattr(_RecordingModel, "_generate", generate)
    monkeypatch.setattr(TestBrowserAgent, "model", model, raising=False)
    case = TestBrowserAgent("test_v1_form_continues_after_fair_yield")
    if page_id == "missing-page":
        with pytest.raises(AssertionError, match="old page is closed"):
            case.test_v1_form_continues_after_fair_yield()
    else:
        case.test_v1_form_continues_after_fair_yield()
