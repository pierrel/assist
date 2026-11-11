"""Natural small-model acceptance for exact source references."""

from __future__ import annotations

import functools
import json
import re
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from assist import tools
from assist.agent import create_research_agent
from assist.model_manager import select_assistant_model
from assist.research_cleanup import ReferencesCleanupRunnable
from assist.thread_datastore import ThreadDataStore


SOURCE_URL = "https://decider.example/2026/10/12/the-10-best-horror-movies-now-streaming-on-shudder"


@pytest.mark.parametrize("follow_up", [None, "Could you send me that source link again?"])
def test_natural_horror_research_uses_exact_source_reference(tmp_path, follow_up):
    rows = [{
        "title": "Decider: horror movies on Shudder",
        "url": SOURCE_URL,
        "content": "The article recommends The Wailing, Host, and When Evil Lurks on Shudder.",
    }]

    @functools.wraps(tools.search_internet)
    def search_internet(query: str, max_results: int = 5) -> str:
        return str(rows[:max_results])

    @functools.wraps(tools.read_url)
    def read_url(url: str) -> str:
        return rows[0]["content"] if url == SOURCE_URL else "Error fetching URL: unavailable"

    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    config = {"configurable": {"thread_id": "horror-research",
                               "thread_datastore_path": store.path},
              "recursion_limit": 30}
    with (patch("assist.agent.search_internet", search_internet),
          patch("assist.agent.read_url", read_url)):
        agent = create_research_agent(select_assistant_model(0.1), str(tmp_path),
                                      leaf=True, thread_data_available=True)
        result = agent.invoke(
            {"messages": [{"role": "user", "content":
                           "What horror movies can I stream on Shudder this weekend? Please include a source."}]},
            config)
        if follow_up:
            second = agent.invoke(
                {"messages": result["messages"] + [HumanMessage(content=follow_up)]},
                config)

    assert store.list(kind="source")[0]["value"] == SOURCE_URL
    output = result["messages"][-1].content
    assert f"<{SOURCE_URL}>" in output
    assert "ref:" not in output
    if follow_up:
        assert SOURCE_URL in second["messages"][-1].content
        assert "ref:" not in second["messages"][-1].content


def test_report_file_reread_update_capability_keeps_exact_source(tmp_path):
    """Exercise literal saved-file reread and update around validation."""
    rows = [{
        "title": "Decider: horror movies on Shudder",
        "url": SOURCE_URL,
        "content": "The article recommends The Wailing, Host, and When Evil Lurks on Shudder.",
    }]

    @functools.wraps(tools.search_internet)
    def search_internet(query: str, max_results: int = 5) -> str:
        return str(rows[:max_results])

    @functools.wraps(tools.read_url)
    def read_url(url: str) -> str:
        return rows[0]["content"] if url == SOURCE_URL else "Error fetching URL: unavailable"

    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    report = tmp_path / "references" / "shudder-watchlist.md"
    config = {"configurable": {"thread_id": "watchlist-update",
                               "thread_datastore_path": store.path},
              "recursion_limit": 40}
    prompts = (
        "Please make a short sourced Shudder watchlist at references/shudder-watchlist.md "
        "from a current roundup. Write it, read back what you saved, then put Host "
        "first and polish the list.",
        "Please update that watchlist for a friend: add a short 'For a friend' "
        "heading, keep the three films, and keep the source link intact.",
    )
    before_validation = []
    validate = ReferencesCleanupRunnable._validate_report

    def capture_validation(self, basename, source_store):
        before_validation.append({"basename": basename,
                                  "content": (report.parent / basename).read_text()})
        return validate(self, basename, source_store)

    with (patch("assist.agent.search_internet", search_internet),
          patch("assist.agent.read_url", read_url),
          patch.object(ReferencesCleanupRunnable, "_validate_report",
                       capture_validation)):
        agent = create_research_agent(select_assistant_model(0.1), str(tmp_path),
                                      leaf=True, thread_data_available=True)
        first = agent.invoke({"messages": [{"role": "user", "content": prompts[0]}]}, config)
        first_validation_count = len(before_validation)
        first_text = report.read_text()
        second = agent.invoke(
            {"messages": first["messages"] + [HumanMessage(content=prompts[1])]}, config)

    (tmp_path / "messages.json").write_text(json.dumps({
        "prompts": prompts,
        "before_validation": before_validation,
        "first_validation_count": first_validation_count,
        "first_report": first_text,
        "final_report": report.read_text(),
        "first": [message.model_dump(mode="json") for message in first["messages"]],
        "second": [message.model_dump(mode="json") for message in second["messages"]],
    }, indent=2))

    def report_calls(messages):
        return [call for message in messages if isinstance(message, AIMessage)
                for call in message.tool_calls
                if str(call.get("args", {}).get("file_path", "")).lstrip("/")
                in {"shudder-watchlist.md", "references/shudder-watchlist.md"}]

    calls = report_calls(first["messages"])
    later_calls = report_calls(second["messages"][len(first["messages"]):])
    tool_results = {message.tool_call_id: message for message in second["messages"]
                    if isinstance(message, ToolMessage)}
    successful = lambda call: (tool_results.get(call["id"]) is not None
                               and tool_results[call["id"]].status == "success")
    successful_calls = [call for call in calls if successful(call)]
    successful_later_calls = [call for call in later_calls if successful(call)]
    writes = [index for index, call in enumerate(successful_calls)
              if call["name"] in {"write_file", "edit_file"}]
    reads = [index for index, call in enumerate(successful_calls)
             if call["name"] == "read_file"]
    assert writes and reads and any(writes[0] < read < writes[-1] for read in reads)
    assert any(call["name"] == "read_file" for call in successful_later_calls)
    assert any(call["name"] in {"write_file", "edit_file"}
               for call in successful_later_calls)
    source = store.list(kind="source")[0]
    assert source["value"] == SOURCE_URL
    assert any(SOURCE_URL in entry["content"] and "ref:" not in entry["content"]
               for entry in before_validation[:first_validation_count]
               if entry["basename"] == report.name)
    assert any(f"ref:{source['id']}" in str(call.get("args", {}).get("content", ""))
               for call in calls if call["name"] == "write_file")
    first_read_ids = {call["id"] for call in calls if call["name"] == "read_file"}
    later_read_ids = {call["id"] for call in later_calls if call["name"] == "read_file"}
    first_reads = [message.content for message in first["messages"]
                   if isinstance(message, ToolMessage)
                   and message.tool_call_id in first_read_ids]
    later_reads = [message.content for message in second["messages"]
                   if isinstance(message, ToolMessage)
                   and message.tool_call_id in later_read_ids]
    assert any(SOURCE_URL in text and "ref:" not in text for text in first_reads)
    assert any(SOURCE_URL in text and "ref:" not in text for text in later_reads)
    assert "for a friend" not in first_text.lower()
    assert re.search(r"(?im)^\s*#{1,6}\s+For a friend\b[^\n]*$",
                     report.read_text())
    entries = [match.group(1).lstrip("*_[") for line in first_text.splitlines()
               if (match := re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.+)$", line))]
    entries = [next((title for title in ("Host", "The Wailing", "When Evil Lurks")
                     if entry.startswith(title)), None) for entry in entries]
    entries = [entry for entry in entries if entry is not None]
    assert entries[:3] == ["Host", "The Wailing", "When Evil Lurks"]
    for text in (first_text, report.read_text()):
        assert SOURCE_URL in text
        assert "ref:" not in text
        assert all(title in text for title in ("The Wailing", "Host", "When Evil Lurks"))
