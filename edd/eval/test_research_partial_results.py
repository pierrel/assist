"""Research resilience: honest partial-results / best-effort on rate-limiting.

The 2026-07-02 failure: SearXNG's upstream engines rate-limit under a burst, so the
research turn GAVE UP ("couldn't look this up") and DISCARDED a real, on-point result it
already had. The fix (docs/2026-07-08-research-audit.org): search_internet surfaces a
distinct PARTIAL signal; the searcher/orchestrator USE partial/earlier results and are
HONEST about the rate-limiting; the SearchUnavailableBreaker finalizes-not-kills.

Two mocked shapes (NO real search — per docs/2026-07-06-research-eval-mocking.org):
  * PARTIAL — every search returns real results prefixed with the rate-limited note.
  * MIXED-BURST — the first search fully succeeds, the rest come back fully unavailable
    (the real coffee-shop shape). The good first result must still reach the report.

Asserts the SYMPTOM on the real create_research_agent: a non-empty report that USES the
real result and is HONEST about the rate-limiting — not a total-failure give-up.
"""
import os
import tempfile
from unittest.mock import patch

from assist.agent import create_research_agent, AgentHarness
from assist.model_manager import select_assistant_model

os.environ.setdefault("ASSIST_MODEL_URL", "http://127.0.0.1:8000/v1")
_SEARCH_HOST = "http://searxng.test"

# A distinctive fact that only appears in the (mocked) real result, so the report can only
# contain it if the agent actually USED the result instead of giving up. Synthetic persona.
# The URL/content must look like a REAL search hit — an obviously-fake host (example.test)
# makes the small model distrust the result as "placeholder data" and give up for the wrong
# reason, conflating fixture-distrust with the real rate-limit give-up we're testing.
_RESULT = {"title": "Blackbird Roastery — best SF cafes for remote work",
           "url": "https://remoteworkcafes.com/san-francisco/blackbird-roastery",
           "content": ("Blackbird Roastery on Fell Street has fast wifi, ample outlets, and "
                       "quiet mornings — a top pick for remote work in San Francisco.")}
_PAGE = ("<article><h1>Blackbird Roastery</h1><p>Fast wifi, ample outlets, quiet mornings; "
         "a top remote-work pick on Fell Street.</p></article>")
_ENGINES = [["brave", "too many requests"], ["duckduckgo", "CAPTCHA"]]


class _Resp:
    def __init__(self, *, json_payload=None, text=""):
        self._json = json_payload
        self.text = text
        self.status_code = 200

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


def _run(mock_get, prompt):
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "references"), exist_ok=True)
    agent = AgentHarness(create_research_agent(select_assistant_model(0.1), root))
    with patch.dict(os.environ, {"ASSIST_SEARCH_URL": _SEARCH_HOST}), \
         patch("assist.tools.requests.get", mock_get), \
         patch("assist.tools._host_throttle", lambda host: None):
        res = str(agent.message(prompt))
    return res, root


def _is_search(url):
    return url.startswith(_SEARCH_HOST)


def _honest(res):
    # A rate-limit / search-unavailability disclosure — but NOT a coffee-shop's own
    # "unavailable on weekends" / "limited seating". Accept the rate-limit-specific phrases,
    # OR "unavailable"/"couldn't reach"/"limited" ONLY when it's about the SEARCH/BACKEND.
    r = res.lower()
    if any(k in r for k in ("rate-limit", "rate limit", "rate-limited", "rate limited",
                            "couldn't complete", "could not complete", "some searches",
                            "some sources couldn't", "some sources could not")):
        return True
    return (("search" in r or "backend" in r)
            and any(k in r for k in ("unavailable", "couldn't reach", "could not reach",
                                     "couldn't look", "could not look", "rate", "limited")))


def _report_text(root, res):
    """The real deliverable = the report FILE the general agent reads, not the
    orchestrator's terse ``FINAL_REPORT: <file>`` handoff message. Combine any
    report file written under ``root`` with ``res`` (the give-up path writes no
    file, so its text lives only in ``res``) and assert on the union."""
    parts = [res]
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith((".org", ".md")):
                with open(os.path.join(dirpath, f)) as fh:
                    parts.append(fh.read())
    return "\n".join(parts).lower()


_PROMPT = "Where's a good coffee shop for remote work in San Francisco? Write a short report."


class TestResearchPartialResults:
    def test_mixed_burst_keeps_the_first_good_result_and_is_honest(self):
        """THE real failure (2026-07-02): the first search succeeds, later searches come back
        fully rate-limited. The good first result must reach the report (not be discarded),
        and the report must honestly note that some searches couldn't complete."""
        calls = [0]

        def mock_get(url, **kw):
            if _is_search(url):
                calls[0] += 1
                if calls[0] == 1:  # first search fully succeeds
                    return _Resp(json_payload={"results": [_RESULT], "unresponsive_engines": []})
                return _Resp(json_payload={"results": [], "unresponsive_engines": _ENGINES})  # then unavailable
            return _Resp(text=_PAGE)
        res, root = _run(mock_get, _PROMPT)
        report = _report_text(root, res)
        print(f"\n[mixed-burst] searches={calls[0]} blackbird? {'blackbird' in report} "
              f"honest? {_honest(report)}\n  {res[:400]!r}\n")
        assert "blackbird" in report, "the first search's good result must reach the report, not be discarded"
        assert _honest(report), "must honestly note that some searches couldn't complete"

    def test_all_unavailable_no_results_relays_and_does_not_fabricate(self):
        """The narrow give-up case must still work: EVERY search is fully unavailable and no
        result is ever returned -> relay 'couldn't look this up', and do NOT invent an answer
        from memory (the good-result string must NOT appear — there was no good result)."""
        def mock_get(url, **kw):
            if _is_search(url):
                return _Resp(json_payload={"results": [], "unresponsive_engines": _ENGINES})
            return _Resp(text=_PAGE)
        res, root = _run(mock_get, _PROMPT)
        r = _report_text(root, res)
        print(f"\n[all-unavailable] {res[:400]!r}\n")
        assert any(k in r for k in ("couldn't look", "could not look", "couldn't complete",
                                    "unavailable", "couldn't find", "could not find")), \
            "must relay that it couldn't look this up"
        assert "blackbird" not in r, "must NOT fabricate — there was no result to use"
