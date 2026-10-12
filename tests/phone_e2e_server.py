"""Test-only model and shell-container adapters for the phone E2E harness.

The real HTTP app, Run journal, workers and Git lifecycle remain in use.
This simulator does not validate the production Docker/browser implementation.
Never import it from a production entry point.
"""
from __future__ import annotations

import base64
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

from deepagents.backends.protocol import ExecuteResponse, FileUploadResponse
from fastapi import APIRouter, Request
from langchain_core.messages import AIMessage
import requests

from assist.browser import authority
import assist.browser.manager as browser
from assist.sandbox_manager import SandboxManager
from manage.web import app, state, threads
from manage.web.phone_api import _authenticate
from manage.web.protocol_service import SERVICE


GENERATIONS = {}
HOLDS = {}
RESET_REFUSALS = []
ROUTER = APIRouter()
_ANY_GENERATION = object()


class Shell:
    def __init__(self, worktree, scope, owner):
        self.worktree, self.scope, self.owner = worktree, scope, owner
        self.container = SimpleNamespace(id=uuid.uuid4().hex)
        if owner:
            with authority.fence(*scope) as record:
                sequence = max((r.admission_sequence for r in threads._runs().list(scope[1])
                                if r.user_event_id == r.id), default=0)
                record.begin(owner, sequence)
                record.add_generation(owner, self.container.id)
        GENERATIONS[worktree] = self

    def call(self, operation, **data):
        response = requests.post("http://shell:8080/" + operation,
                                 json={"worktree": self.worktree, **data}, timeout=30)
        response.raise_for_status()
        return response.json()

    def execute(self, command):
        return ExecuteResponse(**self.call("execute", command=command))

    def upload_files(self, files):
        self.call("upload", files=[[p, base64.b64encode(b).decode()] for p, b in files])
        return [FileUploadResponse(path=p) for p, _ in files]


def backend(worktree, *, thread_scope=None, owner_run_id=None, **kwargs):
    return Shell(worktree, thread_scope, owner_run_id)


def cleanup(worktree, expected=_ANY_GENERATION):
    shell = GENERATIONS.get(worktree)
    if shell is None or expected is None or (
            expected is not _ANY_GENERATION and expected is not shell.container):
        return
    if shell.owner:
        with authority.fence(*shell.scope) as record:
            record.remove_generation(shell.owner, shell.container.id)
            record.clear(shell.owner)
    if GENERATIONS.get(worktree) is shell:
        GENERATIONS.pop(worktree)


def cleanup_verified(worktree, expected):
    """Reject absent or changed simulator generations before host Git reads."""
    if expected is None or getattr(GENERATIONS.get(worktree), "container", None) is not expected:
        raise RuntimeError("Git sandbox generation is unavailable")
    cleanup(worktree, expected)
    if worktree in GENERATIONS:
        raise RuntimeError("Git sandbox generation changed during teardown")


def deterministic_thread(tid, working_dir=None, sandbox_backend=None,
                         configurable=None, on_queue_state=None, **kwargs):
    """Use real Thread/LangGraph/SQLite with one deterministic output node."""
    from langgraph.graph import StateGraph, MessagesState, START, END
    from assist.thread import Thread

    def output_node(values):
        text = values["messages"][-1].content
        if text.startswith("Hold "):
            import threading
            hold = HOLDS.setdefault(text, threading.Event())
            if not hold.wait(40):
                raise TimeoutError("E2E model hold exceeded 40 seconds")
        if text == "Hold publication":
            sandbox_backend.upload_files([("/workspace/server-during.txt", b"server-race-result")])
        if text.startswith("Write "):
            filename, content = text[6:].split(" ", 1)
            sandbox_backend.upload_files([("/workspace/" + filename, content.encode())])
            result = "Saved " + filename
        elif text.startswith("Read "):
            result = sandbox_backend.execute("cat -- /workspace/" + text[5:]).output
        else:
            result = "Completed: " + text
        return {"messages": [AIMessage(content=result)]}

    builder = StateGraph(MessagesState)
    builder.add_node("model", output_node)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    graph = builder.compile(checkpointer=state.MANAGER.checkpointer)
    chat = Thread(working_dir or state.MANAGER.thread_default_working_dir(tid),
                  thread_id=tid, agent=graph, model=SimpleNamespace(),
                  configurable=configurable, on_queue_state=on_queue_state,
                  checkpointer=state.MANAGER.checkpointer)
    chat.description = lambda: "E2E thread"
    return chat


def install():
    app.include_router(ROUTER)
    state.MANAGER.get = deterministic_thread
    threads._llm_reachable = lambda: True  # The in-process deterministic graph is ready.
    SandboxManager.get_pi_sandbox_backend = staticmethod(backend)
    SandboxManager.get_git_verification_backend = staticmethod(backend)
    threads._get_sandbox_backend = lambda tid, **kw: backend(
        state.MANAGER.thread_default_working_dir(tid),
        thread_scope=(state.MANAGER.root_dir, tid), **kw)
    SandboxManager.current_container = staticmethod(
        lambda worktree: getattr(GENERATIONS.get(worktree), "container", None))
    SandboxManager.cleanup = staticmethod(cleanup)
    SandboxManager.cleanup_verified = staticmethod(cleanup_verified)
    SandboxManager.reap_orphans = staticmethod(lambda root: None)
    SandboxManager.cleanup_all = staticmethod(lambda: [cleanup(w) for w in list(GENERATIONS)])
    SandboxManager._ensure_egress_proxy_bounded = staticmethod(lambda: None)
    browser._sandbox_generations = lambda root, tid=None: {
        s.container.id for s in GENERATIONS.values() if s.scope == (root, tid)}
    browser._bounded_cli = lambda *a, **kw: b""  # No sidecars in this simulator.
    browser._kill_container_confirmed = lambda gen: [
        cleanup(w) for w, s in list(GENERATIONS.items()) if s.container.id == gen]
    browser.BrowserManager.ready_for_browser = classmethod(lambda cls, *a, **kw: False)
    original_reset = threads._drain_held_browser_events

    def reset(tid):
        shell = GENERATIONS.get(state.MANAGER.thread_default_working_dir(tid))
        held = [r.id for r in threads._runs().list(tid) if r.status == "revocation_pending"]
        result = original_reset(tid)
        if result is False and shell and held:
            RESET_REFUSALS.append({"thread_id": tid, "owner_run_id": shell.owner,
                                   "generation": shell.container.id, "held_run_ids": held})
        return result
    threads._drain_held_browser_events = reset


@ROUTER.get("/__e2e/state/{tid}")
def inspect_state(tid: str, request: Request):
    _authenticate(request)
    with authority.fence(state.MANAGER.root_dir, tid) as record:
        lease = record.lease
    return {"runs": [asdict(r) for r in threads._runs().list(tid)],
            "holds": list(HOLDS), "reset_refusals": list(RESET_REFUSALS), "lease": lease}


@ROUTER.post("/__e2e/release")
def release(body: dict, request: Request):
    _authenticate(request)
    HOLDS[body["text"]].set()
    return {"released": True}


@ROUTER.post("/__e2e/child/{tid}")
def child(tid: str, request: Request):
    _authenticate(request)
    parent = threads._runs().list(tid)[-1]
    child_id = "sub-" + uuid.uuid4().hex
    metadata = {"parent_thread_id": tid, "parent_run_id": parent.id,
                "dispatch_key": parent.work_id + ":e2e-child"}
    SERVICE.create_thread(child_id, metadata)
    run = SERVICE.create_run(child_id, "context-agent", "Hold child", metadata=metadata)
    return {"thread_id": child_id, "run_id": run.id}


if __name__ == "__main__":
    install()
    os.environ["ASSIST_PHONE_API_TOKEN"] = Path("/secrets/token").read_text().strip()
    from manage.web.__main__ import TurnSafeServer
    import uvicorn
    TurnSafeServer(uvicorn.Config(app, host="0.0.0.0", port=8443,
        ssl_certfile="/secrets/server.crt", ssl_keyfile="/secrets/server.key",
        log_level="warning", access_log=False)).run()
