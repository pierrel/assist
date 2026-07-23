"""Receptionist tool logic — resolution, ambiguity, domain matching, callback
isolation, and the by-construction tool-set guarantee (assist/receptionist.py;
docs/2026-07-21-voice-call-tech-design.org §4, P0.5)."""
import json
import os

from assist.catalog import ThreadCatalog
from assist.receptionist import _match_domain, _resolve, receptionist_tools
from assist.thread_manager import ThreadManager


def _mk(root, tid, *, description, stage="ready", urgent=False):
    d = os.path.join(root, tid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "description.txt"), "w") as f:
        f.write(description)
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump({"stage": stage}, f)
    if urgent:
        open(os.path.join(d, "urgent_response"), "w").close()
    return d


def _tools(tmp_path, domains, *, opened=None, created=None):
    """Build a tool set over a real catalog; record callback invocations."""
    cat = ThreadCatalog(ThreadManager(str(tmp_path)))
    on_open = (lambda tid: opened.append(tid)) if opened is not None else (lambda tid: None)
    on_new = ((lambda d, m: created.append((d, m)))
              if created is not None else (lambda d, m: None))
    by_name = {t.__name__: t for t in receptionist_tools(cat, domains, on_open, on_new)}
    return by_name


# ── _resolve ────────────────────────────────────────────────────────────────

def _entries(tmp_path):
    import time
    _mk(tmp_path, "a", description="trip planning")
    time.sleep(0.01)
    _mk(tmp_path, "b", description="garden beds")
    return ThreadCatalog(ThreadManager(str(tmp_path))).entries()   # [b, a] mtime-desc


def test_resolve_by_number(tmp_path):
    entries = _entries(tmp_path)
    assert _resolve("1", entries) == (entries[0], [])
    assert _resolve("2", entries) == (entries[1], [])


def test_resolve_number_out_of_range_is_miss(tmp_path):
    entries = _entries(tmp_path)
    assert _resolve("9", entries) == (None, [])
    assert _resolve("0", entries) == (None, [])


def test_resolve_by_exact_id(tmp_path):
    entries = _entries(tmp_path)
    entry, amb = _resolve("a", entries)
    assert entry.id == "a" and amb == []


def test_resolve_by_substring(tmp_path):
    entries = _entries(tmp_path)
    entry, amb = _resolve("garden", entries)
    assert entry.id == "b" and amb == []


def test_resolve_ambiguous_returns_matches(tmp_path):
    import time
    _mk(tmp_path, "a", description="trip to Rome planning")
    time.sleep(0.01)
    _mk(tmp_path, "b", description="trip to Oslo planning")
    entries = ThreadCatalog(ThreadManager(str(tmp_path))).entries()
    entry, amb = _resolve("trip", entries)
    assert entry is None
    assert {e.id for e in amb} == {"a", "b"}


def test_resolve_empty_or_miss(tmp_path):
    entries = _entries(tmp_path)
    assert _resolve("", entries) == (None, [])
    assert _resolve("nonexistent topic", entries) == (None, [])


# ── _match_domain ─────────────────────────────────────────────────────────────

def test_match_domain_exact_and_substring():
    domains = ["Home", "Garden Project"]
    assert _match_domain("home", domains) == ("Home", [])
    assert _match_domain("garden", domains) == ("Garden Project", [])
    assert _match_domain("the garden project thread", domains) == ("Garden Project", [])
    assert _match_domain("", domains) == (None, [])
    assert _match_domain("finance", domains) == (None, [])


def test_match_domain_overlapping_labels_are_ambiguous_not_misrouted():
    # "home" is a substring of "homework" — first-match-wins would silently
    # misroute; instead both match ⇒ ambiguous ⇒ new_thread asks which.
    matched, amb = _match_domain("I need homework help", ["Home", "Homework"])
    assert matched is None
    assert set(amb) == {"Home", "Homework"}
    matched, amb = _match_domain("the network one", ["Work", "Network"])
    assert matched is None and set(amb) == {"Work", "Network"}


# ── list_threads ──────────────────────────────────────────────────────────────

def test_list_threads_numbers_and_tags(tmp_path):
    import time
    _mk(tmp_path, "a", description="trip planning", stage="processing")
    time.sleep(0.01)
    _mk(tmp_path, "b", description="garden", urgent=True)
    out = _tools(tmp_path, ["Home"])["list_threads"]()
    lines = out.splitlines()
    assert lines[0] == "1. garden (urgent)"          # mtime-desc, urgent tag
    assert lines[1] == "2. trip planning (processing)"


def test_list_threads_empty(tmp_path):
    out = _tools(tmp_path, ["Home"])["list_threads"]()
    assert "no threads" in out.lower()


# ── open_thread ───────────────────────────────────────────────────────────────

def test_open_thread_by_number_uses_last_listing(tmp_path):
    opened = []
    tools = _tools(tmp_path, ["Home"], opened=opened)
    _mk(tmp_path, "a", description="trip planning")
    tools["list_threads"]()                 # seeds _last
    reply = tools["open_thread"]("1")
    assert opened == ["a"]
    assert "trip planning" in reply


def test_open_thread_without_listing_scans_catalog(tmp_path):
    opened = []
    _mk(tmp_path, "a", description="trip planning")
    tools = _tools(tmp_path, ["Home"], opened=opened)
    reply = tools["open_thread"]("trip")   # no list_threads first
    assert opened == ["a"] and "trip planning" in reply


def test_open_thread_is_idempotent_within_session(tmp_path):
    # The small model sometimes fires open_thread twice for one intent; the second
    # open of the thread we're already on must NOT bind (or speak "connecting") again.
    opened = []
    _mk(tmp_path, "a", description="trip planning")
    tools = _tools(tmp_path, ["Home"], opened=opened)
    first = tools["open_thread"]("trip")
    second = tools["open_thread"]("trip")
    assert opened == ["a"]                          # on_open fired once, not twice
    assert "connecting" in first.lower()
    assert "already connected" in second.lower()


def test_open_thread_topic_resolves_full_catalog_not_stale_listing(tmp_path):
    # A topic/id must reach ANY thread even after the caller listed — resolve those
    # against the full catalog, not the (capped, now-stale) _last listing.
    opened = []
    _mk(tmp_path, "a", description="apple thread")
    tools = _tools(tmp_path, ["Home"], opened=opened)
    tools["list_threads"]()                         # _last = [apple] (the heard list)
    _mk(tmp_path, "b", description="banana thread")  # created AFTER the listing
    reply = tools["open_thread"]("banana")           # not in _last, but in the catalog
    assert opened == ["b"]                           # resolved against the full catalog
    assert "banana" in reply


def test_open_thread_ambiguous_asks(tmp_path):
    opened = []
    _mk(tmp_path, "a", description="trip to Rome")
    _mk(tmp_path, "b", description="trip to Oslo")
    tools = _tools(tmp_path, ["Home"], opened=opened)
    reply = tools["open_thread"]("trip")
    assert opened == []                              # nothing opened
    assert "Rome" in reply and "Oslo" in reply


def test_open_thread_miss(tmp_path):
    opened = []
    _mk(tmp_path, "a", description="trip planning")
    tools = _tools(tmp_path, ["Home"], opened=opened)
    reply = tools["open_thread"]("nonexistent")
    assert opened == [] and "couldn't tell" in reply.lower()


def test_open_thread_callback_failure_is_spoken(tmp_path):
    _mk(tmp_path, "a", description="trip planning")
    cat = ThreadCatalog(ThreadManager(str(tmp_path)))

    def boom(tid):
        raise RuntimeError("session gone")

    tools = {t.__name__: t for t in
             receptionist_tools(cat, ["Home"], boom, lambda d, m: None)}
    reply = tools["open_thread"]("trip")
    assert "couldn't connect" in reply.lower()       # not raised


# ── new_thread ────────────────────────────────────────────────────────────────

def test_new_thread_matches_domain(tmp_path):
    created = []
    tools = _tools(tmp_path, ["Home", "Garden"], created=created)
    reply = tools["new_thread"]("garden", "plant tomatoes")
    assert created == [("Garden", "plant tomatoes")]
    assert "Garden" in reply


def test_new_thread_single_domain_auto(tmp_path):
    created = []
    tools = _tools(tmp_path, ["Home"], created=created)
    reply = tools["new_thread"]("whatever", "hi")
    assert created == [("Home", "hi")]               # sole domain used despite no match


def test_new_thread_unknown_domain_asks(tmp_path):
    created = []
    tools = _tools(tmp_path, ["Home", "Garden"], created=created)
    reply = tools["new_thread"]("finance", "x")
    assert created == []
    assert "Home" in reply and "Garden" in reply     # lists the options


def test_new_thread_ambiguous_domain_asks_which(tmp_path):
    created = []
    tools = _tools(tmp_path, ["Home", "Homework"], created=created)
    reply = tools["new_thread"]("I need homework help", "algebra problem set")
    assert created == []                             # doesn't silently pick "Home"
    assert "Home" in reply and "Homework" in reply   # asks which


def test_new_thread_no_domains_configured(tmp_path):
    created = []
    tools = _tools(tmp_path, [], created=created)
    reply = tools["new_thread"]("x", "y")
    assert created == [] and "aren't any projects" in reply.lower()


def test_new_thread_callback_failure_is_spoken(tmp_path):
    cat = ThreadCatalog(ThreadManager(str(tmp_path)))

    def boom(d, m):
        raise RuntimeError("create failed")

    tools = {t.__name__: t for t in
             receptionist_tools(cat, ["Home"], lambda tid: None, boom)}
    reply = tools["new_thread"]("home", "x")
    assert "couldn't start" in reply.lower()


# ── by-construction tool set ─────────────────────────────────────────────────

def test_tool_set_is_exactly_the_three(tmp_path):
    cat = ThreadCatalog(ThreadManager(str(tmp_path)))
    names = {t.__name__ for t in receptionist_tools(cat, ["Home"], lambda t: None, lambda d, m: None)}
    assert names == {"list_threads", "open_thread", "new_thread"}
