"""The visible Pi choice is host-gated before a thread can be reserved."""
from __future__ import annotations

import asyncio
import shutil
from contextlib import contextmanager

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from manage.web import threads
from manage.web import state
from assist.pi_runtime import PiRuntimeError, PiRuntimeResult
from assist.pi_trace import PiTraceEvent
from assist.thread_engine import read_thread_engine
from assist.web_main_prompt import (WebMainPromptError, WebMainPromptUnavailable,
                                    render_pi_web_main_prompt)


class _Preview:
    def __init__(self, admits: bool) -> None:
        self._admits = admits

    def claim_admits(self, engine: str) -> bool:
        assert engine == "pi"
        return self._admits

    def admits(self, engine: str) -> bool:
        assert engine == "pi"
        return self._admits


class _Queue:
    @contextmanager
    def acquire(self, tid: str, *, user_priority: bool):
        assert tid == "pi-source"
        assert user_priority is False
        yield


@pytest.fixture
def pi_threads_root(tmp_path, monkeypatch):
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    threads._RUN_SERVICES_BY_ROOT.pop(str(tmp_path), None)
    state.DESCRIPTION_CACHE.clear()
    yield tmp_path
    threads._RUN_SERVICES_BY_ROOT.pop(str(tmp_path), None)
    state.DESCRIPTION_CACHE.clear()


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


def test_pi_system_prompt_translates_a_renderer_failure(monkeypatch) -> None:
    def broken_renderer():
        raise WebMainPromptUnavailable("web-main prompt is unavailable")

    monkeypatch.setattr(threads, "render_pi_web_main_prompt", broken_renderer)

    with pytest.raises(PiRuntimeError, match="unavailable"):
        threads._pi_system_prompt()


def test_pi_system_prompt_fails_closed_on_an_unknown_renderer_error(monkeypatch) -> None:
    def broken_renderer():
        raise WebMainPromptError("web-main prompt is unavailable")

    monkeypatch.setattr(threads, "render_pi_web_main_prompt", broken_renderer)

    with pytest.raises(PiRuntimeError, match="invalid"):
        threads._pi_system_prompt()


def test_pi_system_prompt_is_the_host_rendered_shared_prompt() -> None:
    assert threads._pi_system_prompt() == render_pi_web_main_prompt().text


def test_pi_run_passes_the_host_rendered_prompt_to_the_runtime(
    pi_threads_root, monkeypatch,
) -> None:
    threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    threads._create_empty_pi_workspace("pi-source")
    run = threads._create_run("pi-source", "Inspect this workspace")
    received = {}

    class Runtime:
        def run(self, **kwargs):
            received.update(kwargs)
            return PiRuntimeResult("Done", 1)

    monkeypatch.setattr(threads, "PI_PREVIEW", _Preview(True))
    monkeypatch.setattr(threads, "THREAD_QUEUE", _Queue())
    monkeypatch.setattr(threads, "_PI_RUNTIME", Runtime())
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_args: None)

    threads._execute_pi_run(run, user_priority=False)

    assert received["system_prompt"] == render_pi_web_main_prompt().text
    assert received["prompt"] == "Inspect this workspace"
    assert received["history"] == []


def test_pi_prompt_render_failure_does_not_start_the_runtime(
    pi_threads_root, monkeypatch,
) -> None:
    threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    threads._create_empty_pi_workspace("pi-source")
    run = threads._create_run("pi-source", "Inspect this workspace")
    started = False

    class Runtime:
        def run(self, **_kwargs):
            nonlocal started
            started = True
            raise AssertionError("Pi runtime must not start")

    def broken_renderer():
        raise WebMainPromptError("web-main prompt is invalid")

    monkeypatch.setattr(threads, "PI_PREVIEW", _Preview(True))
    monkeypatch.setattr(threads, "THREAD_QUEUE", _Queue())
    monkeypatch.setattr(threads, "_PI_RUNTIME", Runtime())
    monkeypatch.setattr(threads, "render_pi_web_main_prompt", broken_renderer)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_args: None)

    threads._execute_pi_run(run, user_priority=False)

    assert not started


def test_merge_refuses_pi_before_constructing_a_deep_thread(monkeypatch) -> None:
    monkeypatch.setattr(threads, "_is_pi_thread", lambda tid: True)
    monkeypatch.setattr(
        threads.MANAGER, "get",
        lambda *args, **kwargs: pytest.fail("Pi must not construct a Deep thread"))

    with pytest.raises(HTTPException) as error:
        threads.merge_thread("pi-thread")

    assert error.value.status_code == 409


def test_invalid_engine_marker_returns_controlled_web_errors(pi_threads_root) -> None:
    tid = threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    (pi_threads_root / tid / "engine.json").write_text("{}")
    client = TestClient(threads.app)

    page = client.get(f"/thread/{tid}")
    message = client.post(f"/thread/{tid}/message", data={"text": "hello"})

    assert page.status_code == 409
    assert message.status_code == 409


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


def test_first_pi_message_creation_creates_its_empty_workspace(
    pi_threads_root, monkeypatch,
) -> None:
    monkeypatch.setattr(threads, "PI_PREVIEW", _Preview(True))
    monkeypatch.setattr(threads, "DOMAINS", [])

    tid, run_id, selected = threads.create_thread_with_message_core(
        "hello", None, engine="pi")

    assert run_id
    assert selected is None
    assert (pi_threads_root / tid / "domain").is_dir()
    assert (pi_threads_root / tid / "description.txt").read_text() == "hello"


def test_empty_pi_thread_creation_creates_its_workspace(
    pi_threads_root, monkeypatch,
) -> None:
    monkeypatch.setattr(threads, "PI_PREVIEW", _Preview(True))
    monkeypatch.setattr(threads, "DOMAINS", [])

    response = asyncio.run(threads.create_thread(None, "pi"))

    tid = response.headers["location"].removeprefix("/thread/")
    assert (pi_threads_root / tid / "domain").is_dir()


def test_empty_pi_thread_never_generates_a_deep_title(
    pi_threads_root, monkeypatch,
) -> None:
    tid = threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    monkeypatch.setattr(
        threads.MANAGER, "get",
        lambda *args, **kwargs: pytest.fail("Pi title lookup must not construct Deep"),
    )

    assert state.get_cached_description(tid) == "Pi thread"
    assert not (pi_threads_root / tid / "description.txt").exists()


def test_first_message_on_an_empty_pi_thread_persists_its_title(
    pi_threads_root,
) -> None:
    tid = threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    threads._create_empty_pi_workspace(tid)

    threads._accept_message_run(tid, "  First useful line\nmore detail")

    assert (pi_threads_root / tid / "description.txt").read_text() == "First useful line"


def test_pi_title_uses_its_first_durable_user_message(pi_threads_root) -> None:
    tid = threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    threads._PI_CONVERSATIONS.append(
        threads.MANAGER.thread_dir(tid), "first-run", "user", "Original first message")

    threads._accept_message_run(tid, "A later submission")

    assert (pi_threads_root / tid / "description.txt").read_text() == "Original first message"


def test_opening_a_pre_title_pi_thread_backfills_its_first_user_title(
    pi_threads_root,
) -> None:
    tid = threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    threads._PI_CONVERSATIONS.append(
        threads.MANAGER.thread_dir(tid), "pi-run", "user", "Repair this title")
    threads._PI_CONVERSATIONS.append(
        threads.MANAGER.thread_dir(tid), "pi-run", "assistant", "Done")

    page = asyncio.run(threads.get_thread(tid))

    assert "Repair this title" in page
    assert (pi_threads_root / tid / "description.txt").read_text() == "Repair this title"


def test_pi_activity_renders_between_a_user_turn_and_its_reply(pi_threads_root) -> None:
    tid = threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    page = threads.render_thread(
        tid, None,
        pi_messages=[
            {"role": "user", "content": "Check the workspace", "run_id": "run-1"},
            {"role": "assistant", "content": "Done", "run_id": "run-1"},
        ],
        pi_traces=[
            PiTraceEvent("run-1", 1, 1, "tool", "read", "started"),
            PiTraceEvent("run-1", 2, 1, "tool", "read", "completed"),
        ],
    )

    assert "Pi activity: read" in page
    assert page.index("Done") < page.index("Pi activity: read") < page.index("Check the workspace")


def test_terminal_pi_turn_without_a_trace_is_unavailable(pi_threads_root) -> None:
    tid = threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    run = threads._runs().create(tid, "main", "Check the workspace")
    threads._runs().claim(tid, run.id)
    threads._runs().transition(tid, run.id, "success")

    page = threads.render_thread(
        tid, None,
        pi_messages=[{"role": "user", "content": "Check the workspace", "run_id": run.id}],
        pi_traces=[],
    )

    assert "Activity unavailable" in page


def test_corrupt_pi_trace_renders_one_unavailable_notice(pi_threads_root) -> None:
    tid = threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    run = threads._runs().create(tid, "main", "Check the workspace")
    threads._runs().claim(tid, run.id)
    threads._runs().transition(tid, run.id, "success")

    page = threads.render_thread(
        tid, None,
        pi_messages=[{"role": "user", "content": "Check the workspace", "run_id": run.id}],
        pi_traces=[], pi_trace_unavailable=True,
    )

    assert page.count("Activity unavailable") == 1


def test_pi_title_backfill_preserves_a_user_rename(pi_threads_root) -> None:
    tid = threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    state.set_description(tid, "A name I chose")
    threads._PI_CONVERSATIONS.append(
        threads.MANAGER.thread_dir(tid), "pi-run", "user", "Do not replace my name")

    threads._backfill_pi_description(tid)

    assert (pi_threads_root / tid / "description.txt").read_text() == "A name I chose"


def test_first_pi_turn_failure_still_reports_setup_failure(pi_threads_root) -> None:
    tid = threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    threads._ensure_pi_description(tid, "A first message")
    threads._create_run(tid, "A first message")
    threads._set_status(tid, "error", error="Pi preview is unavailable")

    page = threads.render_thread(tid, None, pi_messages=[])

    assert "Setup failed:" in page


def test_empty_workspace_setup_cannot_recreate_a_deleted_pi_thread(
    pi_threads_root,
) -> None:
    tid = threads.MANAGER.reserve_visible("pi", thread_id="pi-source")
    shutil.rmtree(pi_threads_root / tid)

    threads._create_empty_pi_workspace(tid)

    assert not (pi_threads_root / tid).exists()


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
