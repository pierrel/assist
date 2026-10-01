"""Intentional restart waits the actual producers and durable Run boundary."""

import asyncio
import json
import os
import select
import signal
import subprocess
import sys
import threading

import pytest
import uvicorn
from fastapi import HTTPException
from starlette.background import BackgroundTask
from starlette.responses import Response

from assist.geo.provisioner import Provisioner
from assist.geo.proposals import Proposal, ProposalStore
from assist.geo.registry import RegionRegistry
from assist.schedule.scheduler import Scheduler
from manage.web.__main__ import TurnSafeServer
from manage.web import protocol_service, state, threads
from manage.web.drain import DrainClosed, RUN_GATE


def _thread_root(tmp_path, monkeypatch):
    monkeypatch.setattr(state.MANAGER, "root_dir", str(tmp_path))
    (tmp_path / "turn").mkdir()


def test_startup_term_enters_normal_uvicorn_shutdown(monkeypatch):
    server = TurnSafeServer(uvicorn.Config("manage.web:app"))
    shutdown_called = []

    async def startup(*, sockets=None):
        server.handle_exit(signal.SIGTERM, None)
        server.handle_exit(signal.SIGINT, None)
        server.started = True

    async def shutdown(*, sockets=None):
        shutdown_called.append(True)

    monkeypatch.setattr(server, "startup", startup)
    monkeypatch.setattr(server, "shutdown", shutdown)
    asyncio.run(server._serve())

    assert shutdown_called == [True]
    assert server.should_exit and not server.force_exit


def test_full_asgi_background_turn_blocks_shutdown_until_terminal_run(
        tmp_path, monkeypatch):
    """A sent response is not completion: Uvicorn still owns its BackgroundTask."""
    _thread_root(tmp_path, monkeypatch)
    run = threads._create_run("turn", "accepted")
    service = threads._runs()
    finalizing = threading.Event()
    finish = threading.Event()
    original_transition = service.transition

    def delayed_transition(tid, run_id, status, **fields):
        if status == "success":
            finalizing.set()
            assert finish.wait(5)
        return original_transition(tid, run_id, status, **fields)

    def run_model(tid, text, *, _run, **kwargs):
        service.claim(tid, _run.id)
        state._set_status(tid, "ready")

    monkeypatch.setattr(service, "transition", delayed_transition)
    monkeypatch.setattr(threads, "_process_message", run_model)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *args: None)

    async def scenario():
        server = TurnSafeServer(uvicorn.Config("manage.web:app"))
        server.servers = []

        class Lifespan:
            async def shutdown(self):
                await asyncio.to_thread(RUN_GATE.close_when_idle)

        server.lifespan = Lifespan()
        sent = []

        async def send(message):
            sent.append(message["type"])

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        response = Response("accepted", background=BackgroundTask(
            threads._execute_run, run.id, "turn"))
        asgi_task = asyncio.create_task(response(
            {"type": "http", "method": "POST", "path": "/"}, receive, send))
        server.server_state.tasks.add(asgi_task)
        asgi_task.add_done_callback(server.server_state.tasks.discard)
        assert await asyncio.to_thread(finalizing.wait, 5)
        stopping = asyncio.create_task(server.shutdown())
        await asyncio.sleep(0.2)
        assert sent == ["http.response.start", "http.response.body"]
        assert not stopping.done()
        assert service.get("turn", run.id).status == "running"
        finish.set()
        await asyncio.wait_for(stopping, 5)
        await asgi_task

    try:
        asyncio.run(scenario())
    finally:
        finish.set()
    assert service.get("turn", run.id).status == "success"


def test_run_creation_at_closure_is_durable_or_refused(tmp_path, monkeypatch):
    _thread_root(tmp_path, monkeypatch)
    service = threads._runs()
    creating = threading.Event()
    finish = threading.Event()
    original_create = service.create

    def delayed_create(*args, **kwargs):
        creating.set()
        assert finish.wait(5)
        return original_create(*args, **kwargs)

    monkeypatch.setattr(service, "create", delayed_create)
    worker = threading.Thread(target=threads._create_run, args=("turn", "boundary"))
    worker.start()
    assert creating.wait(5)
    closer = threading.Thread(target=RUN_GATE.close_when_idle)
    closer.start()
    try:
        assert closer.is_alive()
        finish.set()
        worker.join(5)
        closer.join(5)
    finally:
        finish.set()
    assert not worker.is_alive() and not closer.is_alive()
    (run,) = service.list("turn")
    assert run.text == "boundary" and run.status == "pending"
    threads._execute_run(run.id, "turn")
    assert service.get("turn", run.id).status == "pending"
    with pytest.raises(DrainClosed):
        threads._create_run("turn", "too late")


def test_late_initializer_leaves_its_durable_first_run_pending(tmp_path, monkeypatch):
    _thread_root(tmp_path, monkeypatch)
    run = threads._create_run("turn", "first")

    def unexpected_initializer(*args):
        pytest.fail("initializer reset or cloned after closure")

    monkeypatch.setattr(threads, "_initialize_thread_active", unexpected_initializer)
    RUN_GATE.close_when_idle()
    threads._initialize_thread("turn", run.id, "repo://example")
    assert threads._runs().get("turn", run.id).status == "pending"


def test_private_child_update_cannot_cancel_waiting_run_after_closure(
        tmp_path, monkeypatch):
    monkeypatch.setattr(state.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit", lambda *args, **kwargs: None)
    threads.MANAGER.reserve("parent")
    parent = threads._create_run("parent", "question")
    metadata = {"parent_thread_id": "parent", "parent_run_id": parent.id,
                "dispatch_key": "task-start"}
    protocol_service.SERVICE.create_thread("sub-child", metadata)
    first = protocol_service.SERVICE.create_run(
        "sub-child", "delegate-agent", "work", metadata=metadata)
    threads._runs().claim("sub-child", first.id)
    threads._runs().transition("sub-child", first.id, "awaiting_approval")
    RUN_GATE.close_when_idle()
    with pytest.raises(HTTPException) as error:
        protocol_service.SERVICE.create_run(
            "sub-child", "delegate-agent", "late", multitask_strategy="interrupt",
            metadata={**metadata, "dispatch_key": "task-update"})
    assert error.value.status_code == 503
    assert [(run.id, run.status) for run in threads._runs().list("sub-child")] == [
        (first.id, "awaiting_approval")]


def test_schedule_stop_joins_fire_after_advance_and_dispatch(monkeypatch):
    advanced = threading.Event()
    dispatching = threading.Event()
    release_advance = threading.Event()
    release_dispatch = threading.Event()
    stopped = threading.Event()
    def dispatch(*args):
        dispatching.set()
        assert release_dispatch.wait(5)

    scheduler = Scheduler(object(), dispatch, lambda: True, tick_seconds=0.01)
    monkeypatch.setattr(scheduler, "reconcile", lambda: None)
    due = type("Due", (), {"id": "one", "thread_id": "turn", "prompt": "wake", "tz": "UTC"})()
    monkeypatch.setattr(scheduler, "poll", lambda: scheduler._fire(due, None, True))

    def advance(*args):
        advanced.set()
        assert release_advance.wait(5)

    monkeypatch.setattr(scheduler, "_advance", advance)
    scheduler.start()
    assert advanced.wait(5)
    stopper = threading.Thread(target=lambda: (scheduler.stop(), stopped.set()))
    stopper.start()
    try:
        assert not stopped.wait(0.1)
        release_advance.set()
        assert dispatching.wait(5)
        assert not stopped.wait(0.1)
        release_dispatch.set()
        assert stopped.wait(5)
    finally:
        release_advance.set()
        release_dispatch.set()
        stopper.join(5)


def test_geo_close_waits_through_proposal_ack_and_defers_later_delivery(
        tmp_path, monkeypatch):
    registry = RegionRegistry(str(tmp_path))
    proposals = ProposalStore(str(tmp_path))
    proposal = Proposal("us/washington", "Washington", (-124, 45, -117, 49),
                        1, "turn", "directions", "2026-10-01T00:00:00+00:00")
    proposals.put(proposal)
    delivered = []
    provisioner = Provisioner(registry, proposals, lambda *args: True,
                              lambda *args: delivered.append(args), lambda: True)
    acknowledging = threading.Event()
    finish_ack = threading.Event()
    original_remove = proposals.remove

    def delayed_remove(slug):
        acknowledging.set()
        assert finish_ack.wait(5)
        return original_remove(slug)

    monkeypatch.setattr(proposals, "remove", delayed_remove)
    assert provisioner.submit("add", proposal.slug) == "queued"
    assert acknowledging.wait(5)
    stopped = threading.Event()
    closer = threading.Thread(target=lambda: (provisioner.close_delivery(), stopped.set()))
    closer.start()
    try:
        assert not stopped.wait(0.1)
        finish_ack.set()
        assert stopped.wait(5)
    finally:
        finish_ack.set()
        closer.join(5)
    provisioner._executor.shutdown(wait=True)
    assert len(delivered) == 1
    proposals.put(proposal)
    provisioner.deliver_pending()
    assert len(delivered) == 1 and proposals.get(proposal.slug) is not None


def test_shutdown_proof_retains_failed_git_fence(tmp_path, monkeypatch):
    _thread_root(tmp_path, monkeypatch)
    monkeypatch.setattr(state.SandboxManager, "_containers", {"work": object()})
    with pytest.raises(RuntimeError, match="generation still tracked"):
        state._verify_shutdown_sandboxes()
    monkeypatch.setattr(state.SandboxManager, "_containers", {})
    fence = tmp_path / "turn" / "git-sync.json"
    fence.write_text(json.dumps({"sandbox_in_flight": True}))
    with pytest.raises(RuntimeError, match="Git sandbox flight"):
        state._verify_shutdown_sandboxes()
    assert json.loads(fence.read_text())["sandbox_in_flight"] is True
    fence.write_text(json.dumps({"version": 1, "source": "repo"}))
    state._verify_shutdown_sandboxes()  # Initial binding precedes any flight field.


def test_failed_shutdown_proof_keeps_lifespan_pending_without_cleanup(tmp_path):
    """A thrown proof must not let Uvicorn finish shutdown or erase the fence."""
    cleanup_marker = tmp_path / "cleanup-called"
    script = r'''
import asyncio
import sys
from manage.web import app, state, threads
state._recover_interrupted_threads = lambda: None
state.MANAGER.list = lambda: []
state.MANAGER.close = lambda: None
state.load_unseen_cache = lambda: None
state.load_urgent_cache = lambda: None
state.CAPTURE_WORKER.start = lambda: None
state.CAPTURE_WORKER.stop = lambda: None
threads.start_scheduler = lambda: None
threads.stop_scheduler = lambda: None
state.SandboxManager.cleanup_all = lambda: open(sys.argv[1], "w").close()
def failed_proof():
    print("proof failed", flush=True)
    raise RuntimeError("retained Git generation")
state._verify_shutdown_sandboxes = failed_proof
async def run():
    async with state.lifespan(app):
        pass
asyncio.run(run())
'''
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(cleanup_marker)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        env={**os.environ, "ASSIST_THREADS_DIR": str(tmp_path / "threads")})
    try:
        while True:
            ready, _, _ = select.select([process.stdout], [], [], 5)
            assert ready, "shutdown proof did not run"
            line = process.stdout.readline()
            assert line, "child exited before shutdown proof"
            if "proof failed" in line:
                break
        assert process.poll() is None
        assert not cleanup_marker.exists()
    finally:
        process.terminate()
        process.wait(timeout=5)
