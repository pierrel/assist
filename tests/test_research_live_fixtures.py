"""Frozen retrieval controls for later model graph trials; no network access."""

import pytest
import requests

from edd.eval.research_live_fixtures import FrozenRetrieval, install_frozen_retrieval


def _fixture(tmp_path, name):
    return FrozenRetrieval(tmp_path, "http://searxng.test", name)


def test_music_search_is_query_sensitive_and_unknown_is_empty(tmp_path):
    fixture = _fixture(tmp_path, "music")
    kexp = fixture.get("http://searxng.test/search",
                       params={"q": "Deafheaven KEXP Sessions"}).json()["results"]
    phoenix = fixture.get("http://searxng.test/search",
                          params={"q": "Phoenix United Record Store Day"}).json()["results"]
    missing = fixture.get("http://searxng.test/search",
                          params={"q": "unrelated telescope"}).json()["results"]
    assert kexp[0]["url"].endswith("/SpecialRelease/19982")
    assert phoenix[0]["url"].endswith("/SpecialRelease/19972")
    assert missing == []
    assert (tmp_path / "fixture.jsonl").is_file()


def test_partial_page_is_http_failure_without_fabricated_answer(tmp_path):
    fixture = _fixture(tmp_path, "helio-partial")
    results = fixture.get("http://searxng.test/search",
                          params={"q": "Helio Borealis specification"}).json()["results"]
    assert all("TIDELOCK" not in row["content"] for row in results)
    page = fixture.get("https://helio-specs.test/tier/borealis")
    assert page.status_code == 503
    with pytest.raises(requests.HTTPError):
        page.raise_for_status()
    assert "TIDELOCK" not in page.text


def test_helio_citation_frontier_is_background_only(tmp_path):
    fixture = _fixture(tmp_path, "helio")
    page = fixture.get("https://helio-specs.test/cite/aurora/0")
    assert "no tier hardware specification" in page.text
    assert "QUASAR-9" not in page.text
    deep = fixture.get("https://helio-specs.test/cite/aurora/0/0/0")
    assert deep.status_code == 200
    for url in (
        "https://unrecorded.test/page",
        "https://helio-specs.test/cite/unrecorded-tier/0",
        "https://helio-specs.test/cite/aurora/12",
        "https://helio-specs.test/cite/aurora/0/999",
        "https://helio-specs.test/cite/aurora/0/0/8",
        "https://helio-specs.test/cite/aurora/0?unrecorded=1",
        "https://helio-specs.test//cite/aurora/0/",
        "https://helio-specs.test/" + "/" * 21 + "cite/aurora/0" + "/" * 20,
    ):
        with pytest.raises(requests.ConnectionError, match="no recorded response"):
            fixture.get(url)


def test_ballot_pdf_source_uses_exact_frozen_response(tmp_path):
    fixture = _fixture(tmp_path, "ballot")
    text = fixture.pdf("https://elections.cdn.sos.ca.gov/sov/2024-general/sov/65-official-dec-vote-results-bm.pdf")
    assert "Proposition 2 approved" in text
    with pytest.raises(requests.ConnectionError):
        fixture.pdf("https://elections.cdn.sos.ca.gov/other.pdf")


def test_actual_research_tools_consume_frozen_search_and_page(tmp_path, monkeypatch):
    from assist import tools
    from assist.browser.manager import BrowserSession, BrowserUnavailable

    monkeypatch.setenv("ASSIST_SEARCH_URL", "http://searxng.test")
    monkeypatch.setattr(tools, "_host_throttle", lambda _host: None)
    monkeypatch.setattr(tools.requests, "get", tools.requests.get)
    monkeypatch.setattr(tools.requests, "post", tools.requests.post)
    monkeypatch.setattr(tools, "_read_pdf_url", tools._read_pdf_url)
    monkeypatch.setattr(BrowserSession, "command", BrowserSession.command)
    install_frozen_retrieval(tmp_path, "http://searxng.test", "helio")
    result = tools.search_internet("Helio Borealis specification")
    page = tools.read_url("https://helio-specs.test/tier/borealis")
    assert "https://helio-specs.test/tier/borealis" in result
    assert "TIDELOCK power cell" in page
    assert "Error fetching URL" in tools.read_url("https://unrecorded.test/page")
    with pytest.raises(BrowserUnavailable, match="frozen evaluation corpus"):
        BrowserSession.command(None, "open", url="https://helio-specs.test/tier/borealis")
    assert "frozen_browser_unavailable" in (tmp_path / "fixture.jsonl").read_text()
