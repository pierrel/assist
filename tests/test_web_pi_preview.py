"""The visible Pi choice is host-gated before a thread can be reserved."""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from manage.web import threads
from assist.thread_engine import read_thread_engine


class _Preview:
    def __init__(self, admits: bool) -> None:
        self._admits = admits

    def claim_admits(self, engine: str) -> bool:
        assert engine == "pi"
        return self._admits

    def admits(self, engine: str) -> bool:
        assert engine == "pi"
        return self._admits


@pytest.fixture
def pi_threads_root(tmp_path, monkeypatch):
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    threads._RUN_SERVICES_BY_ROOT.pop(str(tmp_path), None)
    yield tmp_path
    threads._RUN_SERVICES_BY_ROOT.pop(str(tmp_path), None)


def test_new_thread_engine_defaults_to_deep_and_rejects_unknown(monkeypatch) -> None:
    assert threads._require_new_thread_engine("deepagents") == "deepagents"
    with pytest.raises(HTTPException) as error:
        threads._require_new_thread_engine("anything")
    assert error.value.status_code == 400


def test_new_pi_thread_requires_fresh_host_admission(monkeypatch) -> None:
    monkeypatch.setattr(threads, "PI_PREVIEW", _Preview(False))
    with pytest.raises(HTTPException) as error:
        threads._require_new_thread_engine("pi")
    assert error.value.status_code == 503
    monkeypatch.setattr(threads, "PI_PREVIEW", _Preview(True))
    assert threads._require_new_thread_engine("pi") == "pi"


def test_merge_refuses_pi_before_constructing_a_deep_thread(monkeypatch) -> None:
    monkeypatch.setattr(threads, "_is_pi_thread", lambda tid: True)
    monkeypatch.setattr(
        threads.MANAGER, "get",
        lambda *args, **kwargs: pytest.fail("Pi must not construct a Deep thread"))

    with pytest.raises(HTTPException) as error:
        threads.merge_thread("pi-thread")

    assert error.value.status_code == 409


def test_pi_page_offers_deep_continuation_when_preview_is_disabled(
    pi_threads_root, monkeypatch,
) -> None:
    threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    monkeypatch.setattr(threads, "PI_PREVIEW", _Preview(False))

    page = threads.render_thread("pi-source", None, pi_messages=[])

    assert 'action="/thread/pi-source/continue-deep"' in page
    assert "Continue in Deep Agents" in page
    assert "Pi transcript and workspace are not" in page
    assert 'name="summary"' in page
    assert 'name="text" required' in page
    assert 'name="text" required placeholder="Type your message..." disabled' in page


def test_continue_pi_in_deep_creates_independent_deep_thread(
    pi_threads_root, monkeypatch,
) -> None:
    source_dir = threads.MANAGER.thread_dir("pi-source")
    threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    threads._PI_CONVERSATIONS.append(source_dir, "pi-run", "user", "Pi-only history")
    initialized: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(
        threads,
        "_initialize_thread",
        lambda tid, run_id, domain, rider=None: initialized.append((tid, run_id, domain)),
    )

    response = TestClient(threads.app).post(
        "/thread/pi-source/continue-deep",
        data={"summary": "Carry this visible context forward."},
        follow_redirects=False,
    )

    assert response.status_code == 303
    destination = response.headers["location"].removeprefix("/thread/")
    assert destination != "pi-source"
    assert read_thread_engine(threads.MANAGER.thread_dir(destination)).name == "deepagents"
    runs = threads._runs().list(destination)
    assert len(runs) == 1
    assert runs[0].text == (
        "Continue this work from Pi preview thread pi-source.\n\n"
        "The Pi transcript, workspace, tools, credentials, approvals, and agent state "
        "were not transferred.\n\n"
        "User-provided summary:\nCarry this visible context forward."
    )
    assert not (pi_threads_root / destination / "pi-conversation.jsonl").exists()
    assert initialized == [(destination, runs[0].id, None)]


def test_empty_pi_handoff_summary_copies_no_transcript() -> None:
    assert threads._pi_continuation_message("pi-source", "   ") == (
        "Continue this work from Pi preview thread pi-source.\n\n"
        "The Pi transcript, workspace, tools, credentials, approvals, and agent state "
        "were not transferred."
    )


def test_continue_deep_refuses_non_pi_source_before_creating_a_thread(
    pi_threads_root,
) -> None:
    threads.MANAGER.reserve_visible("deepagents", thread_id="deep-source")

    response = TestClient(threads.app, raise_server_exceptions=False).post(
        "/thread/deep-source/continue-deep",
        data={"summary": "nope"},
        follow_redirects=False,
    )

    assert response.status_code == 409
    assert sorted(path.name for path in pi_threads_root.iterdir()) == ["deep-source"]


def test_continue_deep_rejects_an_overlong_visible_summary(
    pi_threads_root,
) -> None:
    threads.MANAGER.reserve_visible("pi", thread_id="pi-source")

    response = TestClient(threads.app, raise_server_exceptions=False).post(
        "/thread/pi-source/continue-deep",
        data={"summary": "x" * (threads._PI_CONTINUATION_SUMMARY_LIMIT + 1)},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert sorted(path.name for path in pi_threads_root.iterdir()) == ["pi-source"]


def test_continue_deep_admission_escapes_an_exhausted_shared_worker_pool(
    pi_threads_root, monkeypatch,
) -> None:
    """The handoff is durably admitted even while every default worker is occupied."""
    import threading

    from manage import web
    from manage.web import state as state_module

    threads.MANAGER.reserve_visible("pi", thread_id="pi-source")

    async def no_initialize(*args) -> None:
        return None

    monkeypatch.setattr(threads, "_initialize_thread", no_initialize)
    monkeypatch.setattr(state_module, "_recover_interrupted_threads", lambda: None)
    monkeypatch.setattr(threads, "start_scheduler", lambda: None)
    monkeypatch.setattr(threads, "stop_scheduler", lambda: None)
    monkeypatch.setattr(web.MANAGER, "close", lambda: None)

    async def default_limiter():
        return threads.anyio.to_thread.current_default_thread_limiter()

    occupied = threading.Event()
    release = threading.Event()

    def block_default_worker() -> None:
        occupied.set()
        release.wait()

    async def occupy_default_worker() -> None:
        await threads.anyio.to_thread.run_sync(block_default_worker)

    with TestClient(threads.app) as client:
        limiter = client.portal.call(default_limiter)
        original_tokens = limiter.total_tokens
        limiter.total_tokens = 1
        blocker = client.portal.start_task_soon(occupy_default_worker)
        assert occupied.wait(timeout=2)
        reply: list[object] = []
        request = threading.Thread(
            target=lambda: reply.append(client.post(
                "/thread/pi-source/continue-deep",
                data={"summary": "carry on"},
                follow_redirects=False,
            )),
        )
        try:
            request.start()
            request.join(timeout=2)
            assert not request.is_alive(), "handoff waited on the exhausted shared pool"
            assert reply[0].status_code == 303
        finally:
            release.set()
            blocker.result(timeout=2)
            limiter.total_tokens = original_tokens
