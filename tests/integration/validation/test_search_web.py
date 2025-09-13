from assist.tools.web_search import SearchWeb
from langchain_tavily import TavilySearch


def test_search_web_accepts_config(monkeypatch):
    def fake_run(self, query, **kwargs):
        return {"results": [{"title": "t", "url": "u", "content": "c"}]}

    monkeypatch.setattr(TavilySearch, "_run", fake_run)

    tool = SearchWeb(max_results=1)
    result = tool.invoke("test", config={})
    assert result["brief_summary"] == "1 result"
