"""End-to-end regression test for `_process_message`.

The 2026-05-30 sandbox-defer-until-queue PR (#117) initially shipped
with a `THREAD_QUEUE` NameError — the symbol was referenced in
`_process_message` but never imported in `manage/web/threads.py`.  The
existing test suite missed it because `tests/test_web_review.py` stubs
`_process_message` wholesale, and no other test actually runs the
function end-to-end.

This file plugs that gap.  It POSTs to the real
``/thread/{tid}/message`` route via FastAPI's `TestClient`, lets the
background task run, and asserts the thread reaches ``status="ready"``
without crashing inside ``_process_message``.

What this test runs FOR REAL (catches regressions in):
  - The FastAPI route handler and `BackgroundTasks` scheduling
  - `_process_message`'s top-level flow (imports, status writes,
    exception handling)
  - The `THREAD_QUEUE.acquire(...)` block (catches missing-import
    and contextvar-handling regressions)
  - The post-acquire status sequence

What is STUBBED (NOT exercised here — would need an integration test
with real Docker + a real LLM):
  - `_get_sandbox_backend` — stubbed to None (the same shape it
    returns when Docker is unavailable)
  - `MANAGER.get` — returns a `_FakeChat` (Thread / agent / LLM stack
    is not exercised)
  - The domain-manager sync and description-generation paths
"""
import time

import pytest
from fastapi.testclient import TestClient

from manage import web


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Repoint the singleton ThreadManager at a tmp dir + create one
    thread directory.  Match the pattern from test_web_review.py.
    """
    tdir = tmp_path / "thread-e2e"
    tdir.mkdir()
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(
        web.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid)
    )
    # _process_message also calls MANAGER.thread_default_working_dir
    # (in the SandboxContainerLostError branch); point it at the same dir.
    monkeypatch.setattr(
        web.MANAGER, "thread_default_working_dir", lambda tid: str(tmp_path / tid),
    )
    return TestClient(web.app)


def _wait_for_terminal_status(tid: str, deadline_s: float = 5.0) -> dict:
    """Poll status.json until stage in {ready, error}, or fail."""
    from manage.web.state import _get_status
    end = time.time() + deadline_s
    while time.time() < end:
        st = _get_status(tid)
        if st.get("stage") in ("ready", "error"):
            return st
        time.sleep(0.05)
    return _get_status(tid)


def test_post_message_runs_process_message_without_crashing(
    client, monkeypatch
):
    """The full POST → BackgroundTask → _process_message → THREAD_QUEUE
    acquire → sandbox-backend lookup → chat.message → status="ready"
    path runs without raising.  Regression: PR #117's initial commit
    broke this with a `THREAD_QUEUE` NameError that no existing test
    caught."""
    # Stub the sandbox backend lookup so the test doesn't require Docker.
    # Returning None is the same shape `_get_sandbox_backend` returns when
    # Docker is unavailable on the host.  Patch both the source binding
    # AND the threads-module's already-imported reference; _process_message
    # calls the latter.
    monkeypatch.setattr(
        "manage.web.state._get_sandbox_backend", lambda tid: None,
    )
    monkeypatch.setattr(
        "manage.web.threads._get_sandbox_backend", lambda tid: None,
    )

    # Stub MANAGER.get to return a minimal fake chat whose `.message()`
    # returns a canned response without touching the LLM or langgraph.
    class _FakeChat:
        thread_id = "thread-e2e"
        agent = None
        on_queue_state = None
        def message(self, text):
            return "ok"
        def description(self):
            return "desc"
        def get_messages(self):
            return [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"}]

    monkeypatch.setattr(
        web.MANAGER, "get",
        lambda tid, sandbox_backend=None, on_queue_state=None: _FakeChat(),
    )
    monkeypatch.setattr(web.MANAGER, "touch", lambda tid: None)

    # Stub the domain-manager hook so the post-message sync block is a no-op.
    monkeypatch.setattr(
        "manage.web.threads._get_domain_manager", lambda tid: None,
    )

    # Stub description generation so it doesn't try to network.
    # IMPORTANT: `_process_message` does `from manage.web.state import
    # ... get_cached_description`, so the live binding is the one in
    # `manage.web.threads`'s module namespace — patching only the source
    # module (`manage.web.state.get_cached_description`) would leave the
    # imported reference unchanged.
    monkeypatch.setattr(
        "manage.web.threads.get_cached_description", lambda tid: "stub",
    )

    r = client.post(
        "/thread/thread-e2e/message",
        data={"text": "hello"},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text

    # The BackgroundTask runs in the same TestClient request cycle once
    # the response is consumed (anyio.to_thread.run_sync semantics under
    # the test client are synchronous for our purposes).  Give it a
    # short poll window in case of timing.
    status = _wait_for_terminal_status("thread-e2e", deadline_s=5.0)
    assert status.get("stage") == "ready", (
        f"_process_message did not reach 'ready' — final status: {status!r}. "
        f"This usually means an exception in _process_message itself, e.g. "
        f"a missing import (NameError) or a stub that didn't match the "
        f"call shape."
    )
