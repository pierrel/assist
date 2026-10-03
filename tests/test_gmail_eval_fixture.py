"""The synthetic Gmail lifecycle must not fabricate a delegated document result."""
import json

import pytest


def test_nested_completion_uses_honest_empty_workspace_and_keeps_spies(tmp_path,monkeypatch):
    import edd.eval.test_gmail_skill as fixture
    from edd.eval.test_async_subagents import _result, _start, reset_task_fixture
    from assist import gmail
    import requests
    results=[]
    class Harness:
        def __init__(self,agent):
            self.task_id=None
        def message(self,prompt):
            if self.task_id is None:
                self.task_id=_start("Look for a local document","delegate-agent").split("task_id: ")[1].split(".")[0]
                return "Task started"
            # Exercise the actual lifecycle result fixture at each completion.
            result=json.loads(_result(self.task_id))["result"]
            results.append(result)
            assert "No matching workspace document" in result
            assert "ZX-42" in gmail.GmailClient().read("abc123")["body"]
            with pytest.raises(AssertionError,match="must not mutate"):
                gmail._mutate(["abc123"],"archive")
            with pytest.raises(AssertionError,match="no external requests"):
                requests.get("https://example.test")
            if len(results)==1:
                self.task_id=_start("Try another local document","delegate-agent").split("task_id: ")[1].split(".")[0]
                return "Another task started"
            return "No local document found"
    monkeypatch.setattr(fixture,"AgentHarness",Harness)
    monkeypatch.setattr(fixture,"create_agent",lambda *a,**k:object())
    case=fixture.TestGmailSkill()
    case.model=object()
    case.root=str(tmp_path)
    reset_task_fixture()
    try:
        _,response=case._run("Find the letter")
        assert response=="No local document found" and len(results)==2
    finally:
        reset_task_fixture()
