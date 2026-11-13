"""Integration coverage for the fair-scheduling run path through ``_process_message``
and ``_ResumeScheduler`` (Phase 2 of docs/2026-07-08-fair-scheduling.org).

Exercises the pieces the queue-primitive tests (test_fair_scheduling.py) can't:
the ``except ThreadPauseRequested`` branch, the pending-message carry across a pause
(so the user's bubble doesn't vanish), the resume dispatch via the dedicated scheduler
thread, and the held routing of a new message that arrives while paused.

``MANAGER.get`` / sandbox / domain hooks are stubbed (same shape as
test_web_process_message_e2e.py); the pause is raised where the middleware would raise
it (out of the agent run). The dispatch-contention test waits for the real quantum
timer to request that pause and uses un-mocked model and Run queues.
"""
import contextlib
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from manage import web
from manage.web import threads
from manage.web.state import _get_status
from assist.location import LocationSnapshot
from assist.browser.authority import mark_new_thread
from assist.thread_queue import ThreadAffinityQueue, ThreadPauseRequested


class _PausingChat:
    """message() raises ThreadPauseRequested once (a quantum pause), then resume()
    completes. Records call order so a test can assert the paused turn is NOT re-run."""

    def __init__(self, tid, calls):
        self.thread_id = tid
        self.agent = None
        self.on_queue_state = None
        self._calls = calls

    def message(self, text):
        self._calls.append(("message", text))
        raise ThreadPauseRequested("quantum pause")

    def resume(self):
        self._calls.append(("resume",))
        return "final answer"

    def pending_reply(self):
        return None


@pytest.fixture
def wired(tmp_path, monkeypatch):
    tid = "t-pause"
    (tmp_path / tid).mkdir()
    mark_new_thread(str(tmp_path), tid)
    calls = []
    chat = _PausingChat(tid, calls)

    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(web.MANAGER, "thread_dir", lambda t: str(tmp_path / t))
    # Repoint the legacy journal used by migration-only assertions in this fixture.
    from manage.web.state import MESSAGE_BACKLOG
    monkeypatch.setattr(MESSAGE_BACKLOG, "_root", str(tmp_path))
    monkeypatch.setattr(web.MANAGER, "thread_default_working_dir",
                        lambda t: str(tmp_path / t))
    monkeypatch.setattr(web.MANAGER, "touch", lambda t: None)
    monkeypatch.setattr(
        web.MANAGER, "get",
        lambda t, sandbox_backend=None, on_queue_state=None, configurable=None,
        triage=False, continuation=False: chat)
    monkeypatch.setattr("manage.web.threads._get_sandbox_backend",
                        lambda t, tz=None, **_kwargs: None)
    monkeypatch.setattr("manage.web.threads._get_domain_manager", lambda t: None)
    monkeypatch.setattr("manage.web.threads.get_cached_description", lambda t: "stub")
    monkeypatch.setattr("manage.web.threads.SandboxManager.cleanup", lambda wd, expected=None: None)
    # Drain the scheduler queue so an item from another test can't leak in.
    with contextlib.suppress(Exception):
        while True:
            threads._RESUME_SCHEDULER._q.get_nowait()
    with threads._RESUME_SCHEDULER._q._cond:
        threads._RESUME_SCHEDULER._q._reserved.clear()
    return tid, calls


def test_pause_carries_pending_and_submits_resume(wired):
    tid, calls = wired

    # 1. The initial turn pauses (message() raises ThreadPauseRequested).
    threads._process_message(tid, "hello")

    st = _get_status(tid)
    assert st["stage"] == "paused"
    # fix #1: the user's message bubble survives the pause (not dropped).
    assert st.get("pending_message") == "hello"

    # A resume was queued on the dedicated scheduler (NOT a BackgroundTask), carrying
    # the pending text so the resume keeps the bubble too.
    item = threads._RESUME_SCHEDULER._q.get_nowait()
    run = threads._runs().get(tid, item["run_id"])
    assert item["tid"] == tid and run.resume is True
    assert run.pending_text == "hello"

    # 2. Run the resume as the scheduler would.
    threads._execute_run(item["run_id"], item["tid"])
    assert _get_status(tid)["stage"] == "ready"

    # Lossless + no re-run: the paused message() ran once, then resume() ran once —
    # the turn was NOT restarted with the original message.
    assert calls == [("message", "hello"), ("resume",)]


def test_dispatch_waiter_runs_before_paused_holder_resumes(wired, monkeypatch):
    first_tid, _ = wired
    second_tid = "t-next"
    root = Path(web.MANAGER.root_dir)
    (root / second_tid).mkdir()
    mark_new_thread(str(root), second_tid)
    queue = ThreadAffinityQueue(quantum_s=0.03, hold_timeout_s=1.5)
    scheduler = threads._ResumeScheduler()
    monkeypatch.setattr(threads, "THREAD_QUEUE", queue)
    monkeypatch.setattr(threads, "_RESUME_SCHEDULER", scheduler)
    monkeypatch.setattr(threads, "_recovery_prep", lambda _q: None)
    order = []
    first_started = threading.Event()
    second_started = threading.Event()
    first_resumed = threading.Event()
    resumed_worker_done = threading.Event()
    boundary_wait = threading.Event()

    class Chat:
        agent = None
        on_queue_state = None

        def __init__(self, tid):
            self.thread_id = tid

        def message(self, text):
            if self.thread_id == second_tid:
                order.append("second")
                second_started.set()
                return "second answer"
            order.append("first")
            first_started.set()
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                if threads.active_handle().pause_requested:
                    raise ThreadPauseRequested("quantum pause")
                boundary_wait.wait(0.005)
            raise AssertionError("the dispatch waiter never requested a pause")

        def resume(self):
            order.append("first resume")
            first_resumed.set()
            return "first answer"

        def pending_reply(self):
            return None

        def get_messages(self):
            return []

        def get_raw_messages(self):
            return []

    chats = {tid: Chat(tid) for tid in (first_tid, second_tid)}
    monkeypatch.setattr(web.MANAGER, "get", lambda tid, **_kwargs: chats[tid])
    first = threads._create_run(first_tid, "long turn")
    second = threads._create_run(second_tid, "short turn")
    execute = threads._execute_run

    def execute_and_signal(run_id, tid, **kwargs):
        execute(run_id, tid, **kwargs)
        if tid == first_tid and run_id != first.id:
            resumed_worker_done.set()

    monkeypatch.setattr(threads, "_execute_run", execute_and_signal)

    scheduler.start()
    scheduler.submit(first.id, first_tid)
    assert first_started.wait(1)
    scheduler.submit(second.id, second_tid)

    assert resumed_worker_done.wait(2)
    assert first_resumed.is_set()
    assert second_started.is_set()
    assert order == ["first", "second", "first resume"]
    first_runs = threads._runs().list(first_tid)
    assert [run.status for run in first_runs] == ["interrupted", "success"]
    assert first_runs[1].work_id == first.work_id
    assert threads._runs().get(second_tid, second.id).status == "success"


@pytest.mark.parametrize("real_browser,handoff_failure,same_parent_child", [
    (False, None, False), (False, "reserve", False), (False, "commit", False),
    (False, None, True), pytest.param(
    True, None, False, marks=pytest.mark.skipif(not os.getenv("ASSIST_BROWSER_FAIRNESS_TEST_IMAGE"),
                                   reason="requires explicit isolated Docker image")), pytest.param(
    True, None, True, marks=pytest.mark.skipif(not os.getenv("ASSIST_BROWSER_FAIRNESS_TEST_IMAGE"),
                                   reason="requires explicit isolated Docker image"))])
def test_browser_form_survives_competing_thread_yield(
        wired, monkeypatch, real_browser, handoff_failure, same_parent_child):
    """Real Run/affinity queues retain the exact page and filled input on resume."""
    import json
    from types import SimpleNamespace
    from assist.browser import authority, manager as browser
    first_tid, _ = wired
    second_tid = "sub-browser-child" if same_parent_child else "t-browser-competitor"
    root = Path(web.MANAGER.root_dir)
    (root / second_tid).mkdir()
    if same_parent_child:
        (root / second_tid / ".subagent").write_text("{}")
        monkeypatch.setattr(threads, "EGRESS_STORE", None)
        monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_a: None)
    mark_new_thread(str(root), second_tid)
    queue = ThreadAffinityQueue(quantum_s=0.03, hold_timeout_s=3)
    scheduler = threads._ResumeScheduler()
    monkeypatch.setattr(threads, "THREAD_QUEUE", queue)
    monkeypatch.setattr(threads, "_RESUME_SCHEDULER", scheduler)
    failed_handoff = threading.Event()
    if handoff_failure:
        def fail_handoff(*_args, **_kwargs):
            failed_handoff.set()
            raise OSError("synthetic handoff failure")
        monkeypatch.setattr(scheduler, handoff_failure, fail_handoff)
    monkeypatch.setattr(threads, "_recovery_prep", lambda _q: None)
    monkeypatch.setattr(browser, "configured_directory", lambda: str(root))
    monkeypatch.setattr(browser, "browser_records", lambda *_a: {})
    monkeypatch.setattr(browser.BrowserManager, "reconcile_startup", lambda _r: True)
    browser.BrowserManager._reconciled_roots.add(str(root))
    containers, workers, calls, controls, frozen = {}, {}, [], [], []
    created = []
    original_cli = browser._bounded_cli
    monkeypatch.setattr(browser.SandboxManager, "_containers", containers)

    def backend(tid, **kwargs):
        if real_browser:
            import docker
            agent_dir = root / (tid + "-agent")
            agent_dir.mkdir(exist_ok=True, mode=0o700)
            if tid == first_tid:
                (agent_dir / "memory.md").write_text("SHARED_AGENT_CANARY\n")
            container = docker.from_env().containers.run(
                os.environ["ASSIST_BROWSER_FAIRNESS_TEST_IMAGE"], detach=True,
                remove=False, network_mode="none", user="0:0",
                entrypoint=["/opt/assist-browser/bin/python",
                            "/candidate/tests/browser_fairness_worker.py"],
                environment={"BROWSER_SESSION_TOKEN": "synthetic-token",
                             "BROWSER_RUNTIME_DIR": "/run",
                             "BROWSER_DOWNLOAD_DIR": "/downloads"},
                volumes={str(Path(__file__).resolve().parents[1]):
                         {"bind": "/candidate", "mode": "ro"},
                         str(root / tid): {"bind": "/workspace", "mode": "rw"},
                         str(agent_dir): {"bind": "/agent", "mode": "rw"}})
            created.append(container)
            generation = container.id
            deadline = time.monotonic() + 5
            try:
                while time.monotonic() < deadline:
                    if container.exec_run(["test", "-S", "/run/browser.sock"])[0] == 0:
                        break
                    threading.Event().wait(0.02)
                else:
                    raise RuntimeError("synthetic browser fixture socket did not become ready")
            except Exception:
                startup_log = root / tid / "browser-startup.log"
                startup_log.write_bytes(container.logs())
                raise AssertionError(f"fixture startup failed; worker log: {startup_log}")
        else:
            generation = f"generation-{len(workers)}"
            container = SimpleNamespace(id=generation)
        containers[str(root / tid)] = container
        workers[generation] = {"filled": "", "submitted": 0}
        with authority.fence(str(root), tid) as state:
            run = threads._runs().get(tid, kwargs["owner_run_id"])
            sequence = max(item.admission_sequence for item in threads._runs().list(tid)
                           if item.user_event_id == item.id)
            state.begin(run.id, sequence)
            state.add_generation(run.id, generation)
        if real_browser:
            from assist.sandbox import DockerSandboxBackend
            result = DockerSandboxBackend(container, native_agent_dir=True)
            monkeypatch.setattr(result, "_proxy_logs_since", lambda *_a: None)
            return result
        return SimpleNamespace(container=container)

    def cleanup(work_dir, expected=None):
        container = containers.pop(work_dir, None)
        if container is not None:
            tid = Path(work_dir).name
            with authority.fence(str(root), tid) as state:
                owner = state.lease["owner_run_id"]
                state.remove_generation(owner, container.id)
                state.clear(owner)

    def start(session):
        if real_browser:
            session.token = "synthetic-token"
        session.identity = browser._ContainerIdentity(
            str(root), "172.20.0.2", session.sandbox_generation)

    def transport(generation, payload, **kwargs):
        command = json.loads(payload)
        operation, args = command["operation"], command["args"]
        worker = workers[generation]
        if real_browser:
            import subprocess
            return subprocess.run(
                ["docker", "exec", "-i", generation,
                 "/opt/assist-browser/bin/python", "/candidate/assist/browser/runner.py", "call"],
                input=payload, capture_output=True, timeout=20, check=True).stdout
        if operation == "act":
            assert args["page_id"] == generation + "-page"
            if args["action"] == "fill":
                worker["filled"] = args["target"]["text"]
            else:
                assert worker["filled"] == "synthetic composer"
                worker["submitted"] += 1
        return json.dumps({"result": {"page_id": generation + "-page",
                                     "snapshot_id": "observed"}}).encode()

    monkeypatch.setattr(threads, "_get_sandbox_backend", backend)
    monkeypatch.setattr(browser.SandboxManager, "cleanup", cleanup)
    monkeypatch.setattr(browser.BrowserSession, "_start", start)
    monkeypatch.setattr(browser.BrowserSession, "_capture_storage", lambda _s: None)
    monkeypatch.setattr(browser, "_docker_exec", transport)
    def control(argv, **kwargs):
        controls.append(argv)
        return original_cli(argv, **kwargs) if real_browser else b""

    monkeypatch.setattr(browser, "_bounded_cli", control)
    monkeypatch.setattr(browser, "disarm_sandbox_browser", lambda *_a: None)
    monkeypatch.setattr(browser, "forget_client", lambda *_a: None)
    if not real_browser:
        monkeypatch.setattr(browser, "_kill_container_confirmed", lambda _g: None)
    opened, finished = threading.Event(), threading.Event()
    observation = {}

    class Chat:
        agent = None
        on_queue_state = None

        def __init__(self, tid, tools, sandbox):
            self.thread_id, self.tools = tid, {tool.__name__: tool for tool in tools}
            self.sandbox = sandbox

        def invoke(self, name, **kwargs):
            response = self.tools[name](**kwargs)
            assert response.startswith("{"), response
            parsed = json.loads(response)
            assert "result" in parsed, parsed
            return parsed["result"]

        def message(self, text):
            if self.thread_id == second_tid:
                calls.append("competitor")
                if handoff_failure:
                    return "short answer"
                session = browser.BrowserManager.current_session(first_tid)
                frozen.append(session is not None and session.paused)
                if same_parent_child:
                    assert self.sandbox is session.sandbox_backend
                    assert not self.tools
                if real_browser:
                    counter = root / first_tid / "counter"
                    before = counter.read_text()
                    threading.Event().wait(0.1)
                    if same_parent_child:
                        assert counter.read_text() != before
                        read = self.sandbox.execute("cat /workspace/allowed-input.txt")
                        assert read.exit_code == 0 and "allowed child input" in read.output
                        assert self.sandbox.write("/workspace/child-result.txt", "child findings").error is None
                        readback = self.sandbox.read("/workspace/child-result.txt")
                        assert readback.error is None and "child findings" in str(readback.file_data)
                        assert "SHARED_AGENT_CANARY" in self.sandbox.execute("cat /agent/memory.md").output
                        assert self.sandbox.write("/agent/child-note.md", "CHILD_AGENT_NOTE").error is None
                        assert self.sandbox.execute("printf 'CHILD_APPEND\\n' >> /agent/memory.md").exit_code == 0
                    else:
                        assert counter.read_text() == before
                return "child findings" if same_parent_child else "short answer"
            calls.append("open/fill")
            observation.update(self.invoke("browser_open", url=(
                "http://127.0.0.1:8000/" if real_browser else "https://example.com/form")))
            filled = self.invoke("browser_act", page_id=observation["page_id"],
                        snapshot_id=observation["snapshot_id"], action="fill",
                        target={"role": "textbox", "name": "Artist", "text": "synthetic composer"}
                        if real_browser else
                        {"ref": "query", "text": "synthetic composer"})
            observation.update(filled)
            opened.set()
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                if threads.active_handle().pause_requested:
                    raise ThreadPauseRequested("quantum pause")
                threading.Event().wait(0.005)
            raise AssertionError("competing turn did not trigger fairness")

        def resume(self):
            calls.append("resume/submit")
            if real_browser and same_parent_child:
                assert "CHILD_AGENT_NOTE" in self.sandbox.execute("cat /agent/child-note.md").output
                assert "CHILD_APPEND" in str(self.sandbox.read("/agent/memory.md").file_data)
            self.invoke("browser_act", page_id=observation["page_id"],
                        snapshot_id=observation["snapshot_id"], action="click",
                        target={"role": "button", "name": "SEARCH"})
            finished.set()
            return "submitted once"

        def pending_reply(self):
            return None

        def get_messages(self):
            return []

        get_raw_messages = get_messages

    monkeypatch.setattr(web.MANAGER, "get", lambda tid, **kwargs:
                        Chat(tid, kwargs.get("browser_tools", ()), kwargs.get("sandbox_backend")))
    first = threads._create_run(first_tid, "Find the composer", user_origin=True)
    if same_parent_child:
        second = threads._runs().create(
            second_tid, "delegate-agent", "Read the input and write findings", mode="child",
            parent_thread_id=first_tid, parent_run_id=first.id, dispatch_key="synthetic-child")
        (root / first_tid / "allowed-input.txt").write_text("allowed child input")
    else:
        second = threads._create_run(second_tid, "Quick question", user_origin=True)
    original_execute = threads._execute_run
    first_done, second_done, resume_done = threading.Event(), threading.Event(), threading.Event()

    def execute_and_finish(run_id, tid, **kwargs):
        try:
            return original_execute(run_id, tid, **kwargs)
        finally:
            (first_done if run_id == first.id else
             second_done if tid == second_tid else resume_done).set()

    monkeypatch.setattr(threads, "_execute_run", execute_and_finish)
    scheduler.start()
    try:
        scheduler.submit(first.id, first_tid)
        assert opened.wait(12 if real_browser else 2)
        scheduler.submit(second.id, second_tid)
        if handoff_failure:
            assert failed_handoff.wait(2)
            deadline = time.monotonic() + 1
            while browser.BrowserManager.current_session(first_tid) and time.monotonic() < deadline:
                threading.Event().wait(0.005)
            assert browser.BrowserManager.current_session(first_tid) is None
            assert str(root / first_tid) not in containers
            with authority.fence(str(root), first_tid) as state:
                assert state.lease is None
            assert "resume/submit" not in calls
            assert first_done.wait(2) and second_done.wait(2)
            return
        assert finished.wait(12 if real_browser else 3), calls
        assert resume_done.wait(2)
        deadline = time.monotonic() + 1
        while browser.BrowserManager.current_session(first_tid) and time.monotonic() < deadline:
            threading.Event().wait(0.005)
        assert calls == ["open/fill", "competitor", "resume/submit"]
        assert frozen == [not same_parent_child]
        if same_parent_child:
            child = threads._runs().get(second_tid, second.id)
            assert child.status == "success" and child.result == "child findings"
            assert any(run.origin == "task-completion" for run in threads._runs().list(first_tid))
        if real_browser:
            assert (root / first_tid / "submissions").read_text().splitlines() == ["synthetic composer"]
            if same_parent_child:
                print({"shared_container": created[0].id,
                       "page_id": observation["page_id"], "child_run": second.id,
                       "child_status": child.status, "submissions": 1,
                       "shared_agent_state_readable": True,
                       "child_agent_write_persisted": True})
        else:
            assert sum(worker["submitted"] for worker in workers.values()) == 1
        assert len(workers) == (1 if same_parent_child else 2)
        expected_generation = created[0].id if real_browser else "generation-0"
        expected = [
            ["docker", "pause", expected_generation],
            ["docker", "unpause", expected_generation]]
        if same_parent_child:
            expected *= 2
        assert [argv for argv in controls if argv[1] in {"pause", "unpause"}] == expected
        assert browser.BrowserManager.current_session(first_tid) is None
        assert containers == {}
    finally:
        browser.BrowserManager._sessions.pop(first_tid, None)
        browser.BrowserManager._sessions.pop(second_tid, None)
        for container in created:
            with contextlib.suppress(Exception):
                container.remove(force=True)


@pytest.mark.skipif(not os.getenv("ASSIST_BROWSER_FAIRNESS_TEST_IMAGE"),
                    reason="requires explicit isolated Docker image")
def test_exact_stop_proof_kills_a_frozen_generation():
    import docker
    from assist.browser import manager as browser
    client = docker.from_env()
    container = client.containers.run(
        os.environ["ASSIST_BROWSER_FAIRNESS_TEST_IMAGE"], ["sleep", "30"],
        detach=True, remove=True, network_mode="none")
    try:
        container.pause()
        browser._kill_container_confirmed(container.id)
        with pytest.raises(docker.errors.NotFound):
            client.containers.get(container.id)
    finally:
        with contextlib.suppress(docker.errors.NotFound):
            container.remove(force=True)


def test_pause_publishes_paused_before_an_inline_successor_or_follower(wired, monkeypatch):
    """An eager scheduler cannot let the old pausing slice overwrite its child.

    The deterministic scheduler below is deliberately more adversarial than the
    production worker: it executes the same-work successor inline after commit
    and records the status it can observe.  A follower remains pending until
    that logical work finishes.
    """
    tid, calls = wired
    head = threads._create_run(tid, "hello")
    follower = threads._create_run(tid, "follow-up")
    submitted, status_at_submit, admission_released, successor_at_commit = [], [], [], []
    commit = threads._RESUME_SCHEDULER.commit

    def execute_inline(ticket):
        unlocked = threads._RUN_ADMISSION_LOCK.acquire(blocking=False)
        admission_released.append(unlocked)
        if unlocked:
            threads._RUN_ADMISSION_LOCK.release()
        status_at_submit.append(_get_status(tid)["stage"])
        successor_at_commit.append(next(
            run.id for run in threads._runs().list(tid)
            if run.work_id == head.work_id and run.id != head.id))
        commit(ticket)
        item = threads._RESUME_SCHEDULER._q.get_nowait()
        submitted.append(item["run_id"])
        run = threads._runs().get(tid, item["run_id"])
        if run.work_id == head.work_id:
            threads._execute_run(item["run_id"], tid)

    monkeypatch.setattr(threads._RESUME_SCHEDULER, "commit", execute_inline)

    threads._execute_run(head.id, tid)

    successor = next(run for run in threads._runs().list(tid)
                     if run.work_id == head.work_id and run.id != head.id)
    assert status_at_submit[0] == "paused"
    assert admission_released[0] is True
    assert successor_at_commit == [successor.id]
    assert successor.status == "success"
    assert threads._runs().get(tid, follower.id).status == "pending"
    assert submitted == [successor.id]
    assert threads._RESUME_SCHEDULER._q.get_nowait()["run_id"] == follower.id
    assert calls == [("message", "hello"), ("resume",)]


def test_pause_successor_keeps_the_original_location_snapshot(wired):
    """A later browser fix must not move an already-paused 'from here' turn."""
    tid, _ = wired
    snapshot = LocationSnapshot(37.7749, -122.4194, datetime.now(timezone.utc))
    original = threads._create_run(tid, "walk from here", location=snapshot)

    threads._execute_run(original.id, tid)

    item = threads._RESUME_SCHEDULER._q.get_nowait()
    successor = threads._runs().get(tid, item["run_id"])
    assert threads._location_from_fields(successor.location) == snapshot


def test_new_message_while_paused_routes_through_scheduler(wired, monkeypatch):
    tid, _ = wired
    from manage.web.state import _set_status
    _set_status(tid, "paused", pending_message="hello")

    submitted = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                        lambda run_id, t: submitted.append((run_id, t)))
    spawned = []
    monkeypatch.setattr(threads.BackgroundTasks, "add_task",
                        lambda self, fn, *a, **k: spawned.append(fn))

    from fastapi.testclient import TestClient
    TestClient(web.app).post(f"/thread/{tid}/message", data={"text": "follow-up"},
                             follow_redirects=False)

    # The new message is persisted but deliberately not notified yet. The queued
    # resume releases followers only after it becomes terminal, so a required-child
    # resume created later cannot be overtaken by this message.
    (run,) = threads._runs().list(tid)
    assert run.text == "follow-up" and run.status == "pending"
    assert submitted == []
    assert threads._execute_run not in spawned


def test_resume_scheduler_user_priority_and_stable_promotion():
    q = threads._PriorityRunQueue()
    q.put({"run_id": "bg-1", "tid": "other"})
    q.put({"run_id": "user-1", "tid": "user", "user_priority": True})
    q.put({"run_id": "parent-resume", "tid": "parent"})
    q.put({"run_id": "bg-2", "tid": "other"})

    q.promote("parent")

    assert [q.get_nowait()["run_id"] for _ in range(4)] == [
        "user-1", "parent-resume", "bg-1", "bg-2"]


def test_resume_scheduler_reservation_is_invisible_until_committed_once():
    q = threads._PriorityRunQueue()
    ticket = q.reserve({"run_id": "resume", "tid": "thread"})

    assert q.empty()
    with pytest.raises(threads.queue.Empty):
        q.get_nowait()

    q.commit(ticket)
    q.commit(ticket)

    assert q.get_nowait()["run_id"] == "resume"
    with pytest.raises(threads.queue.Empty):
        q.get_nowait()


def test_dispatch_probe_ignores_same_thread_and_uncommitted_reservations():
    dispatch = threads._PriorityRunQueue()
    affinity = ThreadAffinityQueue(quantum_s=0.02, hold_timeout_s=1)
    affinity.set_dispatch_waiter_probe(
        lambda tid: len(dispatch.waiting_for_other(tid)))
    ticket = dispatch.reserve({"run_id": "other", "tid": "other"})

    with affinity.acquire("holder") as handle:
        dispatch.put({"run_id": "same", "tid": "holder"})
        threading.Event().wait(0.08)
        assert not handle.pause_requested
        dispatch.commit(ticket)
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline and not handle.pause_requested:
            threading.Event().wait(0.005)
        assert handle.pause_requested


def test_dispatch_probe_skips_cancelled_and_browser_held_wakes(wired):
    _, _ = wired
    tid = "t-held"
    root = Path(web.MANAGER.root_dir)
    (root / tid).mkdir()
    mark_new_thread(str(root), tid)
    scheduler = threads._ResumeScheduler()
    cancelled = threads._create_run(tid, "cancelled")
    threads._runs().transition(tid, cancelled.id, "cancelled")
    reset = threads._create_run(tid, "reset", user_origin=True,
                                revocation_pending=True)
    held = threads._create_run(tid, "behind reset")
    scheduler.submit(cancelled.id, tid)
    scheduler.submit(held.id, tid)

    assert scheduler._dispatch_waiter_count("other") == 0
    assert threads._runs().get(tid, held.id).status == "pending"
    threads._runs().transition(tid, reset.id, "cancelled")
    assert scheduler._dispatch_waiter_count("other") == 1

    scheduler._q.get_nowait()  # cancelled
    scheduler._q.get_nowait()  # held, now runnable
    unbound = threads._runs().create(tid, "general-agent", "unbound",
                                     approval_id="missing")
    scheduler.submit(unbound.id, tid)
    assert scheduler._dispatch_waiter_count("other") == 0


def test_dispatch_probe_skips_interrupted_predecessor_behind_approval(wired):
    tid, _ = wired
    service = threads._runs()
    predecessor = threads._create_run(tid, "first")
    service.claim(tid, predecessor.id)
    service.transition(tid, predecessor.id, "interrupted")
    successor = threads._create_run(
        tid, None, work_id=predecessor.work_id, resume=True)
    service.claim(tid, successor.id)
    action = {"name": "send_email", "args": {"to": "reader@example.test"}}
    service.publish_approval(
        tid, successor.id, action=action, requests=[action],
        proposal={"kind": "send_email", "action": action},
        interrupt_id="gate", checkpoint_id="checkpoint")
    scheduler = threads._ResumeScheduler()
    scheduler.submit(predecessor.id, tid)

    assert service.pending_approval(tid) is not None
    assert service.get(tid, predecessor.id).status == "interrupted"
    assert scheduler._dispatch_waiter_count("other") == 0


def test_dispatch_probe_skips_predecessor_with_reset_held_successor(wired):
    tid, _ = wired
    service = threads._runs()
    predecessor = threads._create_run(tid, "first")
    service.claim(tid, predecessor.id)
    service.transition(tid, predecessor.id, "interrupted")
    reset = threads._create_run(tid, "reset", user_origin=True,
                                revocation_pending=True)
    successor = threads._create_run(
        tid, None, work_id=predecessor.work_id, resume=True)
    scheduler = threads._ResumeScheduler()
    scheduler.submit(predecessor.id, tid)

    assert successor.status == "pending"
    assert successor not in threads._runs_before_browser_reset(service.list(tid))
    threads._execute_run_active(successor.id, tid)
    assert service.get(tid, successor.id).status == "pending"
    assert scheduler._dispatch_waiter_count("other") == 0
    service.transition(tid, reset.id, "cancelled")
    assert scheduler._dispatch_waiter_count("other") == 1


def test_resume_scheduler_dedupes_only_queued_run_ids():
    q = threads._PriorityRunQueue()
    item = {"kind": "run", "run_id": "same", "tid": "thread"}
    q.put(item)
    q.put(dict(item))
    reserved = q.reserve(dict(item))
    q.commit(reserved)
    assert q.get_nowait()["run_id"] == "same"
    with pytest.raises(threads.queue.Empty):
        q.get_nowait()
    # Once the scheduler has taken the item, a paused turn can legitimately
    # queue its next wake before this invocation has fully unwound.
    q.put(dict(item))
    assert q.get_nowait()["run_id"] == "same"
    with pytest.raises(threads.queue.Empty):
        q.get_nowait()


@pytest.mark.parametrize("via_reservation", [False, True])
def test_duplicate_run_wake_upgrades_existing_background_ticket(via_reservation):
    q = threads._PriorityRunQueue()
    q.put({"kind": "run", "run_id": "sms", "tid": "thread"})
    q.put({"kind": "run", "run_id": "unrelated", "tid": "other"})
    upgraded = {"kind": "run", "run_id": "sms", "tid": "thread",
                "user_priority": True}
    if via_reservation:
        q.commit(q.reserve(upgraded))
    else:
        q.put(upgraded)
    assert [q.get_nowait()["run_id"] for _ in range(2)] == ["sms", "unrelated"]
    with pytest.raises(threads.queue.Empty):
        q.get_nowait()


def test_resume_scheduler_promotes_reservations_on_both_sides_of_commit():
    q = threads._PriorityRunQueue()
    q.put({"run_id": "user-first", "tid": "user", "user_priority": True})
    q.put({"run_id": "background", "tid": "other"})
    before = q.reserve({"run_id": "before", "tid": "parent"})
    q.promote("parent")
    q.commit(before)
    after = q.reserve({"run_id": "after", "tid": "later"})
    q.commit(after)
    q.promote("later")

    assert [q.get_nowait()["run_id"] for _ in range(4)] == [
        "user-first", "before", "after", "background"]


def test_pause_reservation_keeps_a_concurrent_user_promotion(wired, monkeypatch):
    """A user admitted between reserve and commit still promotes the paused resume."""
    tid, calls = wired
    head = threads._create_run(tid, "hello")
    entered = threading.Event()
    release = threading.Event()
    commit = threads._RESUME_SCHEDULER.commit

    def block_commit(ticket):
        entered.set()
        assert release.wait(timeout=2)
        commit(ticket)

    monkeypatch.setattr(threads._RESUME_SCHEDULER, "commit", block_commit)
    worker = threading.Thread(target=threads._execute_run, args=(head.id, tid))
    worker.start()
    assert entered.wait(timeout=2)

    follower, busy = threads._accept_message_run(tid, "follow-up")
    assert busy is True
    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()

    successor = next(run for run in threads._runs().list(tid)
                     if run.work_id == head.work_id and run.id != head.id)
    queued = threads._RESUME_SCHEDULER._q.get_nowait()
    assert queued["run_id"] == successor.id
    assert queued["user_priority"] is True
    # The user submission remains held until exact browser reconciliation,
    # even though it has already promoted the paused predecessor's ticket.
    assert threads._runs().get(tid, follower.id).status == "revocation_pending"
    assert calls == [("message", "hello")]


def test_committed_reservation_skips_a_cancelled_run(wired):
    tid, calls = wired
    run = threads._create_run(tid, "cancel me")
    ticket = threads._RESUME_SCHEDULER.reserve(run.id, tid)
    threads._runs().transition(tid, run.id, "cancelled")

    threads._RESUME_SCHEDULER.commit(ticket)
    queued = threads._RESUME_SCHEDULER._q.get_nowait()
    threads._execute_run(queued["run_id"], queued["tid"])

    assert threads._runs().get(tid, run.id).status == "cancelled"
    assert calls == []


def test_committed_reservation_never_recreates_a_deleted_thread(wired):
    tid, calls = wired
    run = threads._create_run(tid, "delete me")
    ticket = threads._RESUME_SCHEDULER.reserve(run.id, tid)
    web.MANAGER.hard_delete(tid)

    threads._RESUME_SCHEDULER.commit(ticket)
    queued = threads._RESUME_SCHEDULER._q.get_nowait()
    threads._execute_run(queued["run_id"], queued["tid"])

    assert not (Path(web.MANAGER.root_dir) / tid).exists()
    assert calls == []


def test_busy_user_submission_promotes_both_wait_points(wired, monkeypatch):
    tid, _ = wired
    from manage.web.state import _set_status
    _set_status(tid, "paused", pending_message="hello")
    scheduler_promotions = []
    affinity_promotions = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "promote",
                        scheduler_promotions.append)
    monkeypatch.setattr(threads.THREAD_QUEUE, "promote", affinity_promotions.append)

    run, busy = threads._accept_message_run(tid, "change course")

    assert busy is True
    assert run.text == "change course"
    assert scheduler_promotions == [tid]
    assert affinity_promotions == [tid]


def test_pending_dispatch_selects_user_before_older_background(wired, monkeypatch):
    tid, _ = wired
    background = threads._create_run(tid, "child done", origin="task-completion")
    user = threads._create_run(tid, "new direction")
    submitted = []
    monkeypatch.setattr(
        threads._RESUME_SCHEDULER, "submit",
        lambda run_id, thread_id, **kwargs:
            submitted.append((run_id, thread_id, kwargs)))

    threads._dispatch_pending_after(tid)

    assert submitted == [(user.id, tid, {"user_priority": True})]
    assert background.status == "pending"
