import pytest
import os
from dotenv import load_dotenv, find_dotenv


@pytest.fixture(scope="session", autouse=True)
def configure_env():
    print("Loading dev dotenv")
    if load_dotenv(find_dotenv(".dev.env")):
        print("Loaded dev dotenv")
    else:
        print("Could not find dev dotenv")
    # SEARCH DEFAULTS TO MOCKED. Strip the live Tavily key from the default eval env so an
    # un-mocked eval can't silently fall back to real search: under eval load SearXNG
    # throttles, and with no Tavily backup the turn then fails LOUD — reminding you to mock
    # (stub_research_subagent) rather than hammer the shared SearXNG / burn the dev-tier
    # Tavily budget. The ONE eval that genuinely exercises live search (test_research_agent)
    # re-sets the key itself via real_search_env(). ASSIST_SEARCH_URL is left as-is on
    # purpose — without it, search GRINDS for hours instead of failing fast.
    os.environ.pop("TAVILY_API_KEY", None)
    yield
