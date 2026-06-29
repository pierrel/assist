"""The context rider: the per-turn ContextRider contract, its prose rendering, the
sandbox-TZ seam, the model-injection middleware, and the web _build_rider boundary.
"""
from datetime import datetime, timezone

import pytest

from assist.context_rider import ContextRider, CONTEXT_RIDER_KEY


# --- the contract / validation -------------------------------------------------

def test_empty_rider_is_inert():
    assert ContextRider().prose_line() is None


def test_bad_timezone_rejected_at_construction():
    with pytest.raises(Exception):
        ContextRider(tz="Not/AZone")


def test_naive_sent_at_rejected():
    with pytest.raises(ValueError):
        ContextRider(sent_at=datetime(2026, 6, 29, 14, 0))  # no tzinfo


def test_out_of_range_coords_rejected():
    with pytest.raises(ValueError):
        ContextRider(lat=91.0, lon=0.0)
    with pytest.raises(ValueError):
        ContextRider(lat=0.0, lon=-181.0)


def test_frozen():
    r = ContextRider(tz="America/Los_Angeles")
    with pytest.raises(Exception):
        r.tz = "UTC"


# --- prose rendering -----------------------------------------------------------

def test_prose_renders_time_in_the_riders_zone():
    # 21:05 UTC == 14:05 PDT
    r = ContextRider(sent_at=datetime(2026, 6, 29, 21, 5, tzinfo=timezone.utc),
                     tz="America/Los_Angeles")
    line = r.prose_line()
    assert line == ("[Message context: sent Monday, June 29, 2026 at 2:05 PM "
                    "(America/Los_Angeles).]")


def test_prose_includes_coarse_location_when_present():
    r = ContextRider(sent_at=datetime(2026, 6, 29, 21, 5, tzinfo=timezone.utc),
                     tz="America/Los_Angeles", lat=37.7749, lon=-122.4194)
    line = r.prose_line()
    assert "from ~37.77, -122.42" in line       # coarse — not full GPS precision
    assert "37.7749" not in line and "-122.4194" not in line


def test_prose_prefers_place_label_over_coords():
    r = ContextRider(lat=37.7749, lon=-122.4194, place_label="downtown SF")
    assert r.prose_line() == "[Message context: from downtown SF.]"


# --- the sandbox-TZ seam -------------------------------------------------------

def test_sandbox_timezone_override_wins():
    from assist.sandbox_manager import _sandbox_timezone
    assert _sandbox_timezone("America/New_York") == "America/New_York"


def test_sandbox_timezone_falls_back_without_override():
    from assist.sandbox_manager import _sandbox_timezone
    assert _sandbox_timezone(None)  # the host/UTC chain — never empty


# --- the model-injection middleware -------------------------------------------
# The middleware reads the run config via langgraph's get_config() (NOT a
# request.runtime attribute, which doesn't exist). We drive it through that real
# accessor; the full chain (get_config populated by a live run) is smoke-tested on
# the deployed web app.

class _FakeRequest:
    def __init__(self, messages):
        self.messages = messages

    def override(self, messages):
        return _FakeRequest(messages)


def _run_mw(configurable, monkeypatch):
    from assist.middleware import context_rider_middleware as mod
    monkeypatch.setattr(mod, "get_config",
                        lambda: ({"configurable": configurable}
                                 if configurable is not None else None))
    req = _FakeRequest(messages=["base"])
    seen = {}

    def handler(r):
        seen["messages"] = r.messages
        return "resp"

    out = mod.ContextRiderMiddleware().wrap_model_call(req, handler)
    return out, seen["messages"]


def test_middleware_injects_the_rider_line(monkeypatch):
    from langchain_core.messages import SystemMessage
    rider = ContextRider(sent_at=datetime(2026, 6, 29, 21, 5, tzinfo=timezone.utc),
                         tz="America/Los_Angeles")
    out, messages = _run_mw({CONTEXT_RIDER_KEY: rider}, monkeypatch)
    assert out == "resp"
    assert len(messages) == 2 and isinstance(messages[-1], SystemMessage)
    assert "June 29, 2026 at 2:05 PM" in messages[-1].content


def test_middleware_noop_without_a_rider(monkeypatch):
    _, messages = _run_mw({}, monkeypatch)
    assert messages == ["base"]


def test_middleware_noop_for_empty_rider(monkeypatch):
    _, messages = _run_mw({CONTEXT_RIDER_KEY: ContextRider()}, monkeypatch)
    assert messages == ["base"]


def test_middleware_noop_when_no_run_config(monkeypatch):
    # Outside a run get_config() returns None — must not crash, must not inject.
    _, messages = _run_mw(None, monkeypatch)
    assert messages == ["base"]


# --- the web boundary: _build_rider -------------------------------------------

def test_build_rider_from_iso_and_tz():
    from manage.web.threads import _build_rider
    r = _build_rider("2026-06-29T21:05:00+00:00", "America/Los_Angeles")
    assert r is not None and r.tz == "America/Los_Angeles"
    assert "2:05 PM" in r.prose_line()


def test_build_rider_none_when_absent():
    from manage.web.threads import _build_rider
    assert _build_rider(None, None) is None
    assert _build_rider("", "") is None


def test_build_rider_swallows_bad_input():
    from manage.web.threads import _build_rider
    assert _build_rider("not-a-date", "America/Los_Angeles") is None  # never blocks a message
    assert _build_rider("2026-06-29T21:05:00+00:00", "Not/AZone") is None
