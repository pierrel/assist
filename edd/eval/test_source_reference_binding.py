"""Natural small-model acceptance for exact source references."""

from __future__ import annotations

import functools
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from assist import tools
from assist.agent import create_research_agent
from assist.model_manager import select_assistant_model
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
        agent = create_research_agent(select_assistant_model(0.1), str(tmp_path), leaf=True)
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
