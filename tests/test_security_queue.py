"""CPU-only queue/Run integration for the isolated security executor."""
from contextlib import nullcontext
import json
import threading
import time
from types import SimpleNamespace

import pytest

from edd.security_eval import Evidence
from edd import security_execution


@pytest.mark.parametrize("cancel", [False, True], ids=["child-after-five-seconds", "aggregate-stop"])
def test_private_executor_child_queue(monkeypatch, tmp_path, cancel):
    from assist import hermes_runtime, sandbox_manager
    from manage.web import threads
    from manage.web import protocol_service

    root = tmp_path / "private"
    (root / "threads").mkdir(parents=True)
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(root / "threads"))
    # execute() replaces these process globals for its disposable child process.
    for name in ("THREAD_QUEUE", "_HERMES_RUNTIME", "HERMES_PREVIEW", "_RESUME_SCHEDULER"):
        monkeypatch.setattr(threads, name, getattr(threads, name))
    monkeypatch.setenv("ASSIST_MODEL_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(hermes_runtime, "HermesRuntimeManager",
                        lambda **kw: SimpleNamespace(stop=lambda tid: None))
    monkeypatch.setattr(security_execution, "instrument", lambda *a: nullcontext())
    monkeypatch.setattr(security_execution, "require_managed_execution", lambda *a: nullcontext())
    monkeypatch.setattr(sandbox_manager.SandboxManager, "_containers", {})
    monkeypatch.setattr(threads, "_get_sandbox_backend", lambda *a, **kw: None)
    monkeypatch.setattr(threads, "_complete_child_handoff", lambda *a: None)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *a: None)
    effects, children, workers, errors = [], [], [], []
    cancelled = threading.Event()
    runs = threads._runs()
    original_cancel = runs.cancel

    def record_cancel(tid, run_id):
        result = original_cancel(tid, run_id)
        cancelled.set()
        return result

    monkeypatch.setattr(runs, "cancel", record_cancel)
    monkeypatch.setattr(threads.MANAGER, "get", lambda *a, **kw: SimpleNamespace(
        message=lambda text: effects.append(text) or "synthetic child complete"))

    def child_work(child):
        try:
            threads._execute_child_run(child)
        except BaseException as error:
            errors.append(error)

    def parent_work(run_id, tid, **kw):
        parent = runs.claim(tid, run_id)
        with threads.THREAD_QUEUE.acquire(tid):
            metadata = {"parent_thread_id": tid, "parent_run_id": parent.id,
                        "dispatch_key": f"{parent.work_id}:synthetic-child"}
            protocol_service.SERVICE.create_thread("sub-synthetic-child", metadata)
            # Keep the real persisted child but drive its dispatcher directly,
            # avoiding the model and the unrelated asynchronous handoff loop.
            with monkeypatch.context() as local:
                local.setattr(protocol_service._RESUME_SCHEDULER, "submit", lambda *a, **kw: None)
                child = protocol_service.SERVICE.create_run(
                    "sub-synthetic-child", "context-agent", "synthetic task", metadata=metadata)
            children.append(child)
            queued = threading.Event()
            acquire = threads.THREAD_QUEUE.acquire

            def observed_acquire(*a, **kw):
                return acquire(*a, **kw, on_state_change=lambda state:
                               queued.set() if state == "queued" else None)

            monkeypatch.setattr(threads.THREAD_QUEUE, "acquire", observed_acquire)
            worker = threading.Thread(target=child_work, args=(child,))
            workers.append(worker)
            worker.start()
            assert queued.wait(2), "child never entered the real queue wait"
            if cancel:
                assert cancelled.wait(3), "aggregate cancellation never reached the journal"
            else:
                # Use real elapsed time: the old fixture expires the child here.
                assert not cancelled.wait(5.3)
                assert runs.get(child.thread_id, child.id).status == "pending"
        worker.join(3)
        assert not worker.is_alive()
        if not cancel:
            runs.transition(tid, run_id, "success", result="synthetic parent complete")

    def dispatch(*a, **kw):
        try:
            return parent_work(*a, **kw)
        except BaseException as error:
            errors.append(error)
            raise

    monkeypatch.setattr(threads, "_execute_run", dispatch)
    services = SimpleNamespace(name="synthetic-services")
    fixture = (object(), SimpleNamespace(name="synthetic-proxy"), "172.22.0.1",
               SimpleNamespace(name="synthetic-execution"), services,
               SimpleNamespace(attrs={"NetworkSettings": {"Networks": {
                   services.name: {"IPAddress": "172.22.0.2"}}}}), [])
    args = SimpleNamespace(engine="deepagents", model_url="http://127.0.0.1:8000/v1",
                           model_name="synthetic", context_len=4096, steps=10,
                           wall_seconds=1 if cancel else 8,
                           sandbox_image="unused", native_image="unused")
    evidence = Evidence(tmp_path / "evidence")
    started = time.monotonic()
    try:
        termination = security_execution.execute(root, evidence, args, threads, fixture)
        assert termination == ("wall_limit" if cancel else "finish"), errors
        assert not errors
        assert len(children) == 1
        child = runs.get(children[0].thread_id, children[0].id)
        assert child.status == ("cancelled" if cancel else "success")
        assert effects == ([] if cancel else ["synthetic task"])
        rows = [json.loads(line) for line in evidence.path.read_text().splitlines()]
        final = next(row for row in rows if row["kind"] == "execution_complete")
        assert final["errors"] == []
        assert all(row["status"] == ("cancelled" if cancel else "success")
                   for row in final["runs"])
        if cancel:
            assert time.monotonic() - started < 3
        assert threads.THREAD_QUEUE.peek_holder() is None
    finally:
        cancelled.set()
        for worker in workers:
            worker.join(3)
        evidence.stream.close()
        threads._RUN_SERVICES_BY_ROOT.pop(str(root / "threads"), None)
