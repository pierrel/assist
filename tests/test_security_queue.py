"""CPU-only production queue/Run dispatch integration for the private executor."""
from contextlib import nullcontext
import json
import threading
import time
from types import SimpleNamespace

import pytest

from edd.security_eval import Evidence
from edd import security_execution


@pytest.mark.parametrize("ending", ["finish", "wall", "interrupt-join",
                                    "interrupt-before-start", "interrupt-after-start"])
def test_private_executor_queue_and_cleanup(monkeypatch, tmp_path, ending):
    from assist import hermes_runtime, sandbox_manager
    from manage.web import threads, protocol_service

    root = tmp_path / "private"
    (root / "threads").mkdir(parents=True)
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(root / "threads"))
    for name in ("THREAD_QUEUE", "_HERMES_RUNTIME", "HERMES_PREVIEW", "_RESUME_SCHEDULER"):
        monkeypatch.setattr(threads, name, getattr(threads, name))
    monkeypatch.setenv("ASSIST_MODEL_URL", "http://127.0.0.1:8000/v1")
    stopped = []
    monkeypatch.setattr(hermes_runtime, "HermesRuntimeManager",
                        lambda **kw: SimpleNamespace(stop=lambda tid: stopped.append(tid)))
    # Model, sandbox and unrelated parent wake effects are outside this CPU
    # fixture. SERVICE submission, scheduler, queue, child dispatcher and journal
    # cancellation remain real; no Docker/model request is possible.
    monkeypatch.setattr(security_execution, "instrument", lambda *a: nullcontext())
    monkeypatch.setattr(security_execution, "require_managed_execution", lambda *a: nullcontext())
    monkeypatch.setattr(sandbox_manager.SandboxManager, "_containers", {})
    monkeypatch.setattr(threads, "_get_sandbox_backend", lambda *a, **kw: None)
    monkeypatch.setattr(threads, "_complete_child_handoff", lambda *a: None)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *a: None)
    effects, children, owner_workers, errors = [], [], [], []
    cancelled, queued, parent_ready = threading.Event(), threading.Event(), threading.Event()
    interrupted = ending.startswith("interrupt")
    runs = threads._runs()
    original_cancel = runs.cancel

    def record_cancel(tid, run_id):
        result = original_cancel(tid, run_id)
        cancelled.set()
        return result

    monkeypatch.setattr(runs, "cancel", record_cancel)
    monkeypatch.setattr(threads.MANAGER, "get", lambda *a, **kw: SimpleNamespace(
        message=lambda text: effects.append(text) or "synthetic child complete"))

    def parent_work(run_id, tid):
        if runs.get(tid, run_id).status == "cancelled":
            return
        parent = runs.claim(tid, run_id)
        with threads.THREAD_QUEUE.acquire(tid):
            parent_ready.set()
            if not interrupted:
                acquire = threads.THREAD_QUEUE.acquire

                def observed_acquire(*a, **kw):
                    return acquire(*a, **kw, on_state_change=lambda state:
                                   queued.set() if state == "queued" else None)

                monkeypatch.setattr(threads.THREAD_QUEUE, "acquire", observed_acquire)
                metadata = {"parent_thread_id": tid, "parent_run_id": parent.id,
                            "dispatch_key": f"{parent.work_id}:synthetic-child"}
                protocol_service.SERVICE.create_thread("sub-synthetic-child", metadata)
                children.append(protocol_service.SERVICE.create_run(
                    "sub-synthetic-child", "context-agent", "synthetic task", metadata=metadata))
                assert queued.wait(2), "SERVICE child never reached the real private queue"
            if ending == "finish":
                assert not cancelled.wait(5.3)
                assert runs.get(children[0].thread_id, children[0].id).status == "pending"
            else:
                assert cancelled.wait(3), "executor never cancelled its owned journal work"
        if ending == "finish":
            runs.transition(tid, run_id, "success", result="synthetic parent complete")

    def dispatch(run_id, tid, **kw):
        try:
            row = runs.get(tid, run_id)
            if row.mode == "child":
                return threads._execute_child_run(row)
            return parent_work(run_id, tid)
        except BaseException as error:
            errors.append(error)
            raise

    monkeypatch.setattr(threads, "_execute_run", dispatch)
    original_start, original_join = threading.Thread.start, threading.Thread.join
    injected = False

    def start(thread):
        nonlocal injected
        # The only main-thread start besides the production scheduler is the
        # executor's owner worker. Timer threads are started by its body.
        if (threading.current_thread() is threading.main_thread()
                and thread is not threads._RESUME_SCHEDULER._thread):
            owner_workers.append(thread)
            if ending == "interrupt-before-start":
                injected = True
                raise KeyboardInterrupt("Synthetic CLI interruption")
            result = original_start(thread)
            if ending == "interrupt-after-start":
                assert parent_ready.wait(2)
                injected = True
                raise KeyboardInterrupt("Synthetic CLI interruption")
            return result
        return original_start(thread)

    def join(thread, timeout=None):
        nonlocal injected
        if (ending == "interrupt-join" and not injected and thread in owner_workers
                and threading.current_thread() is threading.main_thread()):
            assert parent_ready.wait(2)
            injected = True
            raise KeyboardInterrupt("Synthetic CLI interruption")
        return original_join(thread, timeout)

    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(threading.Thread, "join", join)
    services = SimpleNamespace(name="synthetic-services")
    fixture = (object(), SimpleNamespace(name="synthetic-proxy"), "172.22.0.1",
               SimpleNamespace(name="synthetic-execution"), services,
               SimpleNamespace(attrs={"NetworkSettings": {"Networks": {
                   services.name: {"IPAddress": "172.22.0.2"}}}}), [])
    args = SimpleNamespace(engine="deepagents", model_url="http://127.0.0.1:8000/v1",
                           model_name="synthetic", context_len=4096, steps=10,
                           wall_seconds=1 if ending == "wall" else 8,
                           sandbox_image="unused", native_image="unused")
    evidence = Evidence(tmp_path / "evidence")
    started = time.monotonic()
    original_protocol_scheduler = protocol_service._RESUME_SCHEDULER
    try:
        if interrupted:
            with pytest.raises(KeyboardInterrupt, match="Synthetic CLI interruption"):
                security_execution.execute(root, evidence, args, threads, fixture)
            assert injected
        else:
            termination = security_execution.execute(root, evidence, args, threads, fixture)
            assert termination == ("wall_limit" if ending == "wall" else "finish"), errors
        assert not errors
        assert len(stopped) == 1
        assert len(children) == (0 if interrupted else 1)
        if children:
            child = runs.get(children[0].thread_id, children[0].id)
            assert child.status == ("success" if ending == "finish" else "cancelled")
        assert effects == (["synthetic task"] if ending == "finish" else [])
        rows = [json.loads(line) for line in evidence.path.read_text().splitlines()]
        final = next(row for row in rows if row["kind"] == "execution_complete")
        assert final["errors"] == []
        assert final["termination"] == ("error" if interrupted else termination)
        assert all(row["status"] == ("success" if ending == "finish" else "cancelled")
                   for row in final["runs"])
        assert all(not worker.is_alive() for worker in owner_workers)
        assert not threads._RESUME_SCHEDULER._thread.is_alive()
        assert protocol_service._RESUME_SCHEDULER is original_protocol_scheduler
        assert threads.THREAD_QUEUE.peek_holder() is None
        if ending == "wall":
            assert time.monotonic() - started < 3
    finally:
        # Diagnostic cleanup also bounds the intentionally failing baseline.
        # Acceptance assertions above run before this fallback, so it cannot
        # make missing executor cancellation or drain appear successful.
        with threads._RUN_ADMISSION_LOCK:
            for directory in (root / "threads").iterdir():
                if directory.is_dir():
                    for row in runs.list(directory.name):
                        if row.status in {"pending", "running", "interrupted"}:
                            original_cancel(directory.name, row.id)
        cancelled.set()
        for worker in owner_workers:
            if worker.ident is not None:
                original_join(worker, 3)
        scheduler = threads._RESUME_SCHEDULER
        if scheduler._thread and scheduler._thread.ident is not None:
            scheduler._q.put({"kind": "security-stop", "tid": "synthetic", "run_id": "synthetic"})
            original_join(scheduler._thread, 3)
        evidence.stream.close()
        threads._RUN_SERVICES_BY_ROOT.pop(str(root / "threads"), None)
