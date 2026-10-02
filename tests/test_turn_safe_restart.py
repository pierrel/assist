"""Intentional restart waits the actual producers and durable Run boundary."""

import asyncio
import json
import os
import select
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone

import pytest
import uvicorn
from fastapi import HTTPException
from starlette.background import BackgroundTask
from starlette.responses import Response

from assist.geo.provisioner import Provisioner
from assist.geo.proposals import Proposal, ProposalStore
from assist.geo.registry import RegionRegistry
from assist.schedule.scheduler import Scheduler
from assist.schedule.model import Cadence, Schedule
from assist.schedule.store import ScheduleStore
from assist.run_service import RunService
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


def test_sigterm_waits_for_synthetic_generation_cleanup_and_terminal_run(
        tmp_path, monkeypatch):
    """Pinned Uvicorn stop includes Starlette's full BackgroundTask and lifespan."""
    _thread_root(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "ROOT", str(tmp_path))
    monkeypatch.setattr(state, "_recover_interrupted_threads", lambda: None)
    monkeypatch.setattr(state.MANAGER, "list", lambda: [])
    monkeypatch.setattr(state.MANAGER, "close", lambda: None)
    monkeypatch.setattr(state.CAPTURE_WORKER, "start", lambda: None)
    monkeypatch.setattr(state.CAPTURE_WORKER, "stop", lambda: None)
    monkeypatch.setattr(threads, "start_scheduler", lambda: None)
    monkeypatch.setattr(threads, "stop_scheduler", lambda: None)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *args: None)

    run = threads._create_run("turn", "accepted")
    service = threads._runs()
    work_dir = str(tmp_path / "turn" / "work")
    killed = []

    class Generation:
        id = "synthetic-generation"

        def kill(self):
            killed.append(True)

    generation = Generation()
    def confirm_synthetic_generation(container_id):
        assert container_id == generation.id
        generation.kill()
    monkeypatch.setattr(
        "assist.sandbox_manager.confirm_generation_stopped",
        confirm_synthetic_generation)
    monkeypatch.setattr(state.SandboxManager, "_containers", {work_dir: generation})
    tearing_down = threading.Event()
    finish_teardown = threading.Event()
    cleanup_calls = []
    actual_cleanup = state.SandboxManager.cleanup
    actual_cleanup_all = state.SandboxManager.cleanup_all

    def synthetic_generation_cleanup(path, expected_container):
        assert path == work_dir and expected_container is generation
        assert state.SandboxManager.current_container(path) is generation
        tearing_down.set()
        assert finish_teardown.wait(5)
        actual_cleanup(path, expected_container=expected_container)

    def cleanup_all():
        cleanup_calls.append(True)
        actual_cleanup_all()

    def run_model(tid, text, *, _run, **kwargs):
        service.claim(tid, _run.id)
        state.SandboxManager.cleanup(work_dir, expected_container=generation)
        state._set_status(tid, "ready")

    monkeypatch.setattr(state.SandboxManager, "cleanup", synthetic_generation_cleanup)
    monkeypatch.setattr(state.SandboxManager, "cleanup_all", cleanup_all)
    monkeypatch.setattr(threads, "_process_message", run_model)

    async def scenario():
        server = TurnSafeServer(uvicorn.Config("manage.web:app"))
        sent = []

        async def send(message):
            sent.append(message["type"])

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def startup(*, sockets=None):
            await server.lifespan.startup()
            server.servers = []
            server.started = True
            response = Response("accepted", background=BackgroundTask(
                threads._execute_run, run.id, "turn"))
            task = asyncio.create_task(response(
                {"type": "http", "method": "POST", "path": "/"}, receive, send))
            server.server_state.tasks.add(task)
            task.add_done_callback(server.server_state.tasks.discard)

        monkeypatch.setattr(server, "startup", startup)
        serving = asyncio.create_task(server._serve())
        assert await asyncio.to_thread(tearing_down.wait, 5)
        server.handle_exit(signal.SIGTERM, None)
        await asyncio.sleep(0.3)
        assert server.should_exit and not serving.done()
        assert sent == ["http.response.start", "http.response.body"]
        assert service.get("turn", run.id).status == "running"
        assert state.SandboxManager.current_container(work_dir) is generation
        assert cleanup_calls == []
        finish_teardown.set()
        await asyncio.wait_for(serving, 5)

    try:
        asyncio.run(scenario())
    finally:
        finish_teardown.set()
    assert service.get("turn", run.id).status == "success"
    assert state.SandboxManager.current_container(work_dir) is None
    assert killed == [True]
    assert cleanup_calls == [True]


def test_os_sigterm_through_server_run_waits_for_accepted_turn(tmp_path):
    """Production signal installation waits for a Run and synthetic generation cleanup.

    Git ``cleanup_verified`` remains for the sibling composition test.
    """
    script = r'''
import asyncio
import os
import signal
import sys
import threading
from pathlib import Path

import uvicorn
from starlette.background import BackgroundTask
from starlette.responses import Response

from manage.web import state, threads
from manage.web.__main__ import TurnSafeServer
import assist.sandbox_manager as sandbox_module

root = Path(sys.argv[1])
ready_fd = int(sys.argv[2])
(root / "turn").mkdir()
state._recover_interrupted_threads = lambda: None
state.MANAGER.list = lambda: []
state.MANAGER.close = lambda: None
state.CAPTURE_WORKER.start = lambda: None
state.CAPTURE_WORKER.stop = lambda: None
threads.start_scheduler = lambda: None
threads.stop_scheduler = lambda: None
threads._dispatch_pending_after = lambda *args: None
run = threads._create_run("turn", "accepted")
service = threads._runs()
work_dir = str(root / "turn" / "work")
release = threading.Event()

class Generation:
    id = "synthetic-generation"

    def kill(self):
        (root / "killed").touch()

generation = Generation()
def confirm_synthetic_generation(container_id):
    assert container_id == generation.id
    generation.kill()
sandbox_module.confirm_generation_stopped = confirm_synthetic_generation
state.SandboxManager._containers = {work_dir: generation}
actual_cleanup = state.SandboxManager.cleanup
actual_cleanup_all = state.SandboxManager.cleanup_all

def synthetic_generation_cleanup(path, expected_container):
    assert path == work_dir and expected_container is generation
    assert state.SandboxManager.current_container(path) is generation
    os.write(ready_fd, b"H")
    assert release.wait(10)
    actual_cleanup(path, expected_container=expected_container)

def cleanup_all():
    assert (root / "killed").exists()
    (root / "cleanup_all").touch()
    actual_cleanup_all()

def run_model(tid, text, *, _run, **kwargs):
    service.claim(tid, _run.id)
    state.SandboxManager.cleanup(work_dir, expected_container=generation)
    state._set_status(tid, "ready")

state.SandboxManager.cleanup = synthetic_generation_cleanup
state.SandboxManager.cleanup_all = cleanup_all
threads._process_message = run_model
threading.Thread(target=lambda: (sys.stdin.readline(), release.set()), daemon=True).start()
signal.signal(signal.SIGTERM, lambda *args: None)  # Uvicorn replays after its handler exits.

server = TurnSafeServer(uvicorn.Config("manage.web:app"))

async def send(message):
    pass

async def receive():
    return {"type": "http.request", "body": b"", "more_body": False}

async def startup(*, sockets=None):
    await server.lifespan.startup()
    server.servers = []
    server.started = True
    response = Response("accepted", background=BackgroundTask(
        threads._execute_run, run.id, "turn"))
    task = asyncio.create_task(response(
        {"type": "http", "method": "POST", "path": "/"}, receive, send))
    server.server_state.tasks.add(task)
    task.add_done_callback(server.server_state.tasks.discard)

server.startup = startup
server.run()
'''
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), str(write_fd)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        pass_fds=(write_fd,),
        env={**os.environ, "ASSIST_THREADS_DIR": str(tmp_path)})
    os.close(write_fd)
    runs = RunService(str(tmp_path))
    try:
        ready, _, _ = select.select([read_fd], [], [], 10)
        assert ready and os.read(read_fd, 1) == b"H"
        os.kill(process.pid, signal.SIGTERM)
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.3)
        assert runs.list("turn")[0].status == "running"
        assert not (tmp_path / "killed").exists()
        assert not (tmp_path / "cleanup_all").exists()
        process.stdin.write(b"release\n")
        process.stdin.flush()
        assert process.wait(timeout=10) == 0, process.stderr.read().decode()
        assert runs.list("turn")[0].status == "success"
        assert (tmp_path / "killed").exists()
        assert (tmp_path / "cleanup_all").exists()
    finally:
        os.close(read_fd)
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


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


def test_schedule_stop_joins_durable_fire_and_run(tmp_path, monkeypatch):
    _thread_root(tmp_path, monkeypatch)
    now = datetime(2026, 6, 15, 18, 0, tzinfo=timezone.utc)
    store = ScheduleStore(str(tmp_path))
    store.add(Schedule("one", "turn", "wake", Cadence(hour=7), "UTC",
                       next_fire_at="2026-06-15T00:00:00+00:00"))
    advancing = threading.Event()
    dispatching = threading.Event()
    release_advance = threading.Event()
    release_dispatch = threading.Event()
    stopped = threading.Event()
    original_update = store.update

    def delayed_update(*args):
        advancing.set()
        assert release_advance.wait(5)
        return original_update(*args)

    def execute_run(run_id, tid):
        service = threads._runs()
        run = service.get(tid, run_id)
        assert run.text == "wake" and run.origin == "system"
        service.claim(tid, run_id)
        dispatching.set()
        assert release_dispatch.wait(5)
        service.transition(tid, run_id, "success")

    monkeypatch.setattr(store, "update", delayed_update)
    monkeypatch.setattr(threads, "_require_deep_thread", lambda tid: None)
    monkeypatch.setattr(threads, "_execute_run", execute_run)
    scheduler = Scheduler(store, threads._scheduled_dispatch, lambda: True,
                          tick_seconds=0.01, now_fn=lambda: now)
    monkeypatch.setattr(scheduler, "reconcile", lambda: None)
    scheduler.start()
    assert advancing.wait(5)
    stopper = threading.Thread(target=lambda: (scheduler.stop(), stopped.set()))
    stopper.start()
    try:
        assert not stopped.wait(0.1)
        release_advance.set()
        assert dispatching.wait(5)
        assert datetime.fromisoformat(store.for_thread("turn")[0].next_fire_at) > now
        assert threads._runs().list("turn")[0].status == "running"
        assert not stopped.wait(0.1)
        release_dispatch.set()
        assert stopped.wait(5)
    finally:
        release_advance.set()
        release_dispatch.set()
        stopper.join(5)
    assert threads._runs().list("turn")[0].status == "success"


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


def test_shutdown_proof_preserves_historical_git_fence(tmp_path, monkeypatch):
    _thread_root(tmp_path, monkeypatch)
    monkeypatch.setattr(state.SandboxManager, "_containers", {"work": object()})
    with pytest.raises(RuntimeError, match="generation still tracked"):
        state._verify_shutdown_sandboxes()
    monkeypatch.setattr(state.SandboxManager, "_containers", {})
    fence = tmp_path / "turn" / "git-sync.json"
    fence.write_text(json.dumps({"sandbox_in_flight": True}))
    state._verify_shutdown_sandboxes()
    assert json.loads(fence.read_text())["sandbox_in_flight"] is True


def test_capture_deadline_after_proof_does_not_hold_shutdown(tmp_path, monkeypatch):
    _thread_root(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "ROOT", str(tmp_path))
    monkeypatch.setattr(state, "_recover_interrupted_threads", lambda: None)
    monkeypatch.setattr(state.MANAGER, "list", lambda: [])
    events = []
    monkeypatch.setattr(state.MANAGER, "close", lambda: events.append("resources"))
    monkeypatch.setattr(state.CAPTURE_WORKER, "start", lambda: None)
    monkeypatch.setattr(threads, "start_scheduler", lambda: None)
    monkeypatch.setattr(threads, "stop_scheduler", lambda: events.append("scheduler"))
    original_close = RUN_GATE.close_when_idle
    def close_gate():
        events.append("gate")
        original_close()
    monkeypatch.setattr(RUN_GATE, "close_when_idle", close_gate)
    original_verify = state._verify_shutdown_sandboxes
    def verify():
        events.append("proof")
        original_verify()
    monkeypatch.setattr(state, "_verify_shutdown_sandboxes", verify)
    tracked = {"work": object()}
    monkeypatch.setattr(state.SandboxManager, "_containers", tracked)

    def capture_deadline():
        events.append("capture")
        raise RuntimeError("capture worker did not stop before its bounded deadline")

    monkeypatch.setattr(state.CAPTURE_WORKER, "stop", capture_deadline)
    def cleanup():
        events.append("cleanup")
        tracked.clear()
    monkeypatch.setattr(state.SandboxManager, "cleanup_all", cleanup)

    async def scenario():
        async with state.lifespan(None):
            pass

    asyncio.run(asyncio.wait_for(scenario(), 5))
    assert events == ["scheduler", "gate", "cleanup", "proof", "capture", "resources"]


def test_failed_shutdown_proof_keeps_lifespan_pending_after_cleanup(tmp_path):
    """A failed post-cleanup proof must not let Uvicorn finish shutdown."""
    cleanup_marker = tmp_path / "cleanup-called"
    capture_marker = tmp_path / "capture-called"
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
state.CAPTURE_WORKER.stop = lambda: open(sys.argv[2], "w").close()
threads.start_scheduler = lambda: None
threads.stop_scheduler = lambda: None
state.SandboxManager._containers = {"work": object()}
state.SandboxManager.cleanup_all = lambda: open(sys.argv[1], "w").close()
async def run():
    async with state.lifespan(app):
        pass
asyncio.run(run())
'''
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(cleanup_marker), str(capture_marker)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        env={**os.environ, "ASSIST_THREADS_DIR": str(tmp_path / "threads")})
    try:
        while True:
            ready, _, _ = select.select([process.stdout], [], [], 5)
            assert ready, "shutdown proof did not run"
            line = process.stdout.readline()
            assert line, "child exited before shutdown proof"
            if "Intentional stop withheld" in line:
                break
        assert process.poll() is None
        assert cleanup_marker.exists()
        assert not capture_marker.exists()
    finally:
        process.terminate()
        process.wait(timeout=5)
