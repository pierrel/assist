"""The terminal verifier is a strict, read-only research leaf."""

from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage

from assist.agent import create_research_agent
from assist.async_subagents import SUBAGENTS
from assist.middleware.read_only_enforcer import ReadOnlyEnforcerMiddleware
from assist.middleware.url_provenance import UrlProvenanceMiddleware
from assist.promptable import base_prompt_for


def test_terminal_verifier_factory_keeps_leaf_and_source_guards():
    with TemporaryDirectory() as work, patch("assist.agent.create_deep_agent") as create:
        create.return_value = MagicMock()
        create_research_agent(
            MagicMock(), work, leaf=True, terminal_fact_check=True,
            trust_human_messages=False,
        )

    agent = create.call_args.kwargs
    assert agent["subagents"] == []
    tool_names = {getattr(tool, "name", None) or getattr(tool, "__name__", None)
                  for tool in agent["tools"]}
    assert {"search_internet", "read_url"} <= tool_names
    assert "caller names the report file" in agent["system_prompt"]
    assert "a provenance refusal makes no http request" in agent["system_prompt"].lower()
    middleware = agent["middleware"]
    assert any(isinstance(item, ReadOnlyEnforcerMiddleware) for item in middleware)
    assert any(isinstance(item, UrlProvenanceMiddleware) for item in middleware)
    assert "fact-check-agent" in SUBAGENTS

    guard = next(item for item in middleware
                 if isinstance(item, UrlProvenanceMiddleware))
    report_url = "https://source.example/report"
    file_read = [
        AIMessage(content="", tool_calls=[{
            "name": "read_file", "args": {"file_path": "report.md"}, "id": "file-1"}]),
        ToolMessage(content=f"Citation: {report_url}", name="read_file",
                    tool_call_id="file-1"),
    ]

    def request(messages):
        return ToolCallRequest(
            tool_call={"name": "read_url", "args": {"url": report_url},
                       "id": "fetch-1"},
            tool=None, state={"messages": messages}, runtime=None,
        )

    with patch("assist.middleware.url_provenance.get_config", return_value={}):
        denied = guard.wrap_tool_call(
            request(file_read), lambda _request: pytest.fail("draft URL fetched"))
        assert denied.status == "error"
        echoed = file_read + [ToolMessage(
            content=f"Verify {report_url}", name="write_todos",
            tool_call_id="todo-1")]
        denied_echo = guard.wrap_tool_call(
            request(echoed), lambda _request: pytest.fail("todo echo fetched"))
        assert denied_echo.status == "error"
        searched = file_read + [ToolMessage(
            content=f"[{{'url': '{report_url}', 'content': 'source passage'}}]",
            name="search_internet", tool_call_id="search-1")]
        allowed = guard.wrap_tool_call(
            request(searched), lambda _request: ToolMessage(
                content="FETCHED", name="read_url", tool_call_id="fetch-1"))
        assert allowed.content == "FETCHED"


@pytest.mark.parametrize("tool", ["write_file", "edit_file", "execute"])
def test_terminal_verifier_rejects_report_mutation(tool):
    request = SimpleNamespace(tool_call={"name": tool, "id": "call-1"})
    result = ReadOnlyEnforcerMiddleware().wrap_tool_call(
        request, lambda _request: pytest.fail("mutating handler ran"),
    )
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.tool_call_id == "call-1"


def test_terminal_verifier_does_not_change_legacy_checker_prompt():
    legacy = base_prompt_for("deepagents/fact_checker.md.j2")
    terminal = base_prompt_for("deepagents/fact_checker.md.j2",
                               terminal_fact_check=True)
    assert "final_report.md" in legacy and "question.txt" in legacy
    assert "caller names the report file" not in legacy
    assert "caller names the report file" in terminal
    assert "final_report.md" not in terminal


def test_terminal_verifier_rejects_trusted_or_recursive_profile():
    with pytest.raises(ValueError, match="untrusted leaf"):
        create_research_agent(MagicMock(), "/tmp/work", leaf=True,
                              terminal_fact_check=True)
    with pytest.raises(ValueError, match="untrusted leaf"):
        create_research_agent(MagicMock(), "/tmp/work", leaf=False,
                              terminal_fact_check=True,
                              trust_human_messages=False)
