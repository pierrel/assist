"""Run one natural research question in an isolated Assist eval.

The web-main profile exercises Web submission and final delivery. Direct leaf,
research-lead and delegate profiles are capability controls. The supervisor
requires product bytes to match the declared installed source, records the
eval-only overlay, then owns the hard deadline and post-timeout reconciliation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def _private_root(path: Path) -> None:
    if not path.is_dir() or (path / "threads").exists():
        raise RuntimeError("supervisor did not provide a fresh private trial directory")


def _case(case_id: str) -> dict:
    manifest = Path(__file__).with_name("research_live_cases.json")
    cases = json.loads(manifest.read_text())["cases"]
    return next(item for item in cases if item["id"] == case_id)


def _load_retrieval_and_model_env(source: Path) -> None:
    """Use only the existing local model/search credentials for this private eval."""
    allowed = {"ASSIST_MODEL_URL", "ASSIST_SEARCH_URL", "ASSIST_API_KEY",
               "TAVILY_API_KEY"}
    for line in source.read_text().splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() in allowed:
            os.environ.setdefault(key.strip(), value.strip())


def _current_children(service, parent_thread_id: str) -> list:
    """One latest Run per durable child work ID, including continuation slices."""
    latest = {}
    for run in service.scan_children():
        if run.parent_thread_id == parent_thread_id:
            latest[(run.thread_id, run.work_id)] = run
    return list(latest.values())


def _checkpoint_reply(checkpoint: Path, thread_id: str,
                      completed_child_id: str | None) -> tuple[str, str | None] | None:
    """Read the canonical final AI message after the relevant durable turn."""
    from assist.visible_conversation import TASK_COMPLETION_RIDER
    from edd.eval.research_trace import reconstruct_messages

    rows = reconstruct_messages(checkpoint, [thread_id])
    messages = next(row["messages"] for row in rows if row["checkpoint_ns"] == "")
    rider = (f"{TASK_COMPLETION_RIDER}Task ID: {completed_child_id}\n"
             if completed_child_id else None)
    relevant = [index for index, message in enumerate(messages)
                if message["type"] == "human" and (
                    rider is None or str(message["data"].get("content", "")).startswith(rider))]
    if not relevant:
        return None
    end = next((index for index in range(relevant[-1] + 1, len(messages))
                if messages[index]["type"] == "human"), len(messages))
    replies = [message["data"] for message in messages[relevant[-1] + 1:end]
               if message["type"] == "ai" and message["data"].get("content")
               and not message["data"].get("tool_calls")]
    if not replies:
        return None
    return str(replies[-1]["content"]), replies[-1].get("id")


def _completed_parent_run(parent_runs, children, required_role=None):
    """Accept only the final parent Run after all launched child work has settled."""
    if not parent_runs or parent_runs[-1].status != "success":
        return None
    parent = parent_runs[-1]
    if not children:
        return None if required_role else parent
    if any(child.status not in {"success", "error", "timeout", "cancelled"}
           for child in children):
        return None
    # A cancelled sibling has no completion wake. Require the latest child
    # that can deliver a result, rather than an earlier draft's receipt.
    completed = max((child for child in children if child.status != "cancelled"),
                    key=lambda child: datetime.fromisoformat(child.updated_at),
                    default=None)
    if completed is None:
        # A parent may cancel all delegated work and still return a final answer.
        return (parent if not required_role and datetime.fromisoformat(
            parent.updated_at) >= max(datetime.fromisoformat(child.updated_at)
                                     for child in children) else None)
    if parent.work_id != f"task-completion:{completed.thread_id}":
        return None
    if datetime.fromisoformat(parent.created_at) < datetime.fromisoformat(
            completed.updated_at):
        return None
    if required_role and (completed.assistant_id != required_role
                          or completed.status != "success"):
        return None
    return parent


def _completed_child_id(parent) -> str | None:
    """Return a child ID only for an actual task-completion parent Run."""
    work_id = parent.work_id if parent is not None else None
    return (work_id.split(":", 1)[1]
            if work_id and work_id.startswith("task-completion:") else None)


def _synthetic_domain_source(artifact: Path, selected: dict) -> Path:
    """Give the real web initializer a private minimal Git source to clone."""
    source = artifact / "synthetic-domain-source"
    source.mkdir(mode=0o700)
    (source / "README.md").write_text("# Private research evaluation workspace\n")
    for name, content in selected.get("workspace_files", {}).items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    commands = (
        ["git", "init", "-q", "-b", "main", str(source)],
        ["git", "-C", str(source), "add", "-A"],
        ["git", "-C", str(source), "-c", "user.name=Research Eval",
         "-c", "user.email=research-eval@invalid", "commit", "-qm", "Initial workspace"],
    )
    for command in commands:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return source


def _saved_note_updates(domain: Path, originals: dict[str, str]) -> dict[str, str]:
    """Return only changed, readable requested notes from this private domain."""
    updates = {}
    for name, original in originals.items():
        path = domain / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(domain):
            continue
        content = path.read_text()
        if content and content != original:
            updates[name] = content
    return updates


def _requested_updates(selected: dict, updates: dict[str, str]) -> bool:
    """Require only this case's requested note paths to have changed."""
    requested = selected.get("expected_updated_files",
                             list(selected.get("workspace_files", {})))
    return bool(requested) and all(path in updates for path in requested)


def _decline_private_egress(client, egress_routes, tid: str) -> list[dict]:
    """Resolve this private trial's pending network cards through the web route."""
    declined = []
    for proposal in egress_routes.approval_previews(tid):
        if proposal["state"] != "pending":
            continue
        response = client.post("/egress/decline", data={
            "token": egress_routes.EGRESS_CSRF, "review": proposal["token"],
            "tid": tid, "host": proposal["host"], "port": str(proposal["port"]),
        }, follow_redirects=False)
        if response.status_code != 303:
            raise RuntimeError(f"private egress decline returned HTTP {response.status_code}")
        declined.append({"host": proposal["host"], "port": proposal["port"]})
    return declined


def _run_research_leaf(threads, selected: dict, artifact: Path) -> None:
    """Exercise the deployed search worker with the original user request."""
    from assist.sandbox_manager import SandboxManager
    from edd.eval.research_live_trace import append_private

    manager = threads.MANAGER
    tid = manager.reserve()
    workspace = Path(manager.make_default_working_dir(manager.thread_dir(tid)))
    (workspace / "README.md").write_text("# Private research evaluation workspace\n")
    for relative, content in selected.get("workspace_files", {}).items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    append_private(artifact / "lifecycle.jsonl", "leaf_started", thread_id=tid)
    sandbox = None
    try:
        sandbox = threads._get_sandbox_backend(tid, include_agent=False)
        if sandbox is None:
            raise RuntimeError("live research leaf requires the ordinary Docker sandbox")
        leaf = manager.get(tid, sandbox_backend=sandbox, assistant_id="research-agent")
        answer = leaf.message(selected["prompt"])
        (artifact / "leaf-answer.txt").write_text(answer)
        append_private(artifact / "lifecycle.jsonl", "leaf_answer_available",
                       thread_id=tid, answer=answer)
    finally:
        if sandbox is not None:
            SandboxManager.cleanup(str(workspace), sandbox.container)


def _run_research_manager_capability(threads, selected: dict, artifact: Path) -> None:
    """Enter the existing nested research lead directly as a capability control."""
    from assist.agent import create_research_agent
    from assist.sandbox_manager import SandboxManager
    from assist.thread import Thread
    from edd.eval.research_live_trace import append_private

    manager = threads.MANAGER
    tid = manager.reserve()
    workspace = Path(manager.make_default_working_dir(manager.thread_dir(tid)))
    for relative, content in selected.get("workspace_files", {}).items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    sandbox = None
    append_private(artifact / "lifecycle.jsonl", "research_manager_capability_started",
                   thread_id=tid)
    try:
        sandbox = threads._get_sandbox_backend(tid, include_agent=False)
        if sandbox is None:
            raise RuntimeError("nested research capability requires Docker sandbox")
        graph = create_research_agent(manager.model, str(workspace), manager.checkpointer,
                                      sandbox_backend=sandbox, leaf=False)
        lead = Thread(str(workspace), agent=graph, thread_id=tid,
                      checkpointer=manager.checkpointer, model=manager.model,
                      sandbox_backend=sandbox)
        answer = lead.message(selected["prompt"])
        (artifact / "research-manager-answer.txt").write_text(answer)
        append_private(artifact / "lifecycle.jsonl", "research_manager_answer_available",
                       thread_id=tid, answer=answer)
    finally:
        if sandbox is not None:
            SandboxManager.cleanup(str(workspace), sandbox.container)


def _run_delegate_capability(threads, selected: dict, artifact: Path) -> None:
    """Enter the existing delegate profile directly; no main routing claim."""
    from assist.sandbox_manager import SandboxManager
    from edd.eval.research_live_trace import append_private

    manager = threads.MANAGER
    tid = manager.reserve()
    workspace = Path(manager.make_default_working_dir(manager.thread_dir(tid)))
    for relative, content in selected.get("workspace_files", {}).items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    append_private(artifact / "lifecycle.jsonl", "delegate_capability_started",
                   thread_id=tid, assistant_id="delegate-agent")
    sandbox = None
    try:
        sandbox = threads._get_sandbox_backend(tid, include_agent=False)
        if sandbox is None:
            raise RuntimeError("delegate capability requires the ordinary Docker sandbox")
        delegate = manager.get(tid, sandbox_backend=sandbox,
                               assistant_id="delegate-agent")
        answer = delegate.message(selected["prompt"])
        (artifact / "delegate-answer.txt").write_text(answer)
        append_private(artifact / "lifecycle.jsonl", "delegate_answer_available",
                       thread_id=tid, answer=answer)
    finally:
        if sandbox is not None:
            SandboxManager.cleanup(str(workspace), sandbox.container)


def run(case_id: str, artifact: Path, egress_proof: Path, env_path: Path,
        *, profile: str = "web-main") -> None:
    os.umask(0o077)
    selected = _case(case_id)
    _load_retrieval_and_model_env(env_path)
    egress = json.loads(egress_proof.read_text())
    _private_root(artifact)
    threads_root = artifact / "threads"
    captures_root = artifact / "captures"
    threads_root.mkdir(mode=0o700)
    captures_root.mkdir(mode=0o700)
    domain_source = _synthetic_domain_source(artifact, selected)
    os.environ.update({
        "ASSIST_THREADS_DIR": str(threads_root),
        "ASSIST_CAPTURES_DIR": str(captures_root),
        "ASSIST_DOMAINS": str(domain_source),
        "RESEARCH_CONVERGENCE_DIR": str(artifact),
        "ASSIST_EGRESS_APPROVALS_DIR": egress["production_env_paths"]["ASSIST_EGRESS_APPROVALS_DIR"],
        "ASSIST_EGRESS_RUNTIME_DIR": egress["reuse_validation"]["runtime_dir"],
    })
    # Import the selected deployed-source copy only after private roots exist.
    from fastapi.testclient import TestClient
    import manage.web as web
    if not Path(web.__file__).resolve().is_relative_to(
            Path(os.environ["RESEARCH_LIVE_PRODUCT_ROOT"]).resolve()):
        raise RuntimeError("web app import did not use the pinned product copy")
    from manage.web import threads
    from manage.web import egress as egress_routes
    from assist.run_service import RunService
    from edd.eval.research_live_trace import (
        append_private, install_existing_egress_guard,
        install_http_trace, install_model_and_tool_trace,
    )
    from edd.eval.research_live_fixtures import install_frozen_retrieval
    from assist.browser.manager import BrowserManager

    search_url = os.environ.get("ASSIST_SEARCH_URL")
    if not search_url:
        raise RuntimeError("configured ASSIST_SEARCH_URL is unavailable")
    from assist.model_manager import current_model_config
    model = current_model_config()
    (artifact / "runtime.json").write_text(json.dumps({
        "evaluation_profile": profile,
        "assist_source": str(Path(web.__file__).resolve()),
        "model": model.model, "model_url": model.url,
        "model_context_length": model.context_len,
        "search_url": search_url,
        "tavily_configured": bool(os.environ.get("TAVILY_API_KEY")),
        "python": sys.version.split()[0],
    }, indent=2) + "\n")
    install_existing_egress_guard(egress_proof)
    # TestClient startup may reap browsers; only this private trial's state is
    # disposable, so it must not prune the shared production client map.
    BrowserManager.prune_absent_clients = classmethod(lambda _cls, _map_dir: None)
    fixture_name = selected.get("fixture")
    if not isinstance(fixture_name, str) or not fixture_name:
        raise RuntimeError("research suite case has no frozen retrieval fixture")
    install_frozen_retrieval(artifact, search_url, fixture_name)
    install_http_trace(artifact, search_url)
    install_model_and_tool_trace(artifact)

    if profile == "research-leaf":
        with TestClient(web.app):
            _run_research_leaf(threads, selected, artifact)
        return
    if profile == "research-manager-capability":
        with TestClient(web.app):
            _run_research_manager_capability(threads, selected, artifact)
        return
    if profile == "delegate-capability":
        with TestClient(web.app):
            _run_delegate_capability(threads, selected, artifact)
        return
    if profile != "web-main":
        raise ValueError(f"unknown research evaluation profile: {profile}")

    changed = threading.Condition()
    original_write = RunService._write
    status_lock = threading.Lock()
    last_run_status: dict[tuple[str, str], str] = {}

    def write_and_notify(self, thread_id, records):
        result = original_write(self, thread_id, records)
        if self.root_dir == str(threads_root):
            with status_lock:
                for record in records:
                    key = (thread_id, record.id)
                    if last_run_status.get(key) != record.status:
                        last_run_status[key] = record.status
                        append_private(artifact / "lifecycle.jsonl", "run_status",
                                       run_thread_id=thread_id, run_id=record.id,
                                       parent_run_id=record.parent_run_id,
                                       assistant_id=record.assistant_id,
                                       status=record.status)
            with changed:
                changed.notify_all()
        return result

    RunService._write = write_and_notify
    accepted_at = time.time()
    deadline = time.monotonic() + selected["deadline_seconds"] - 180
    append_private(artifact / "lifecycle.jsonl", "trial_started",
                   case_id=case_id, deadline_seconds=selected["deadline_seconds"])
    with TestClient(web.app) as client:
        response = client.post("/threads/with-message", data={"text": selected["prompt"]},
                               follow_redirects=False)
        if response.status_code != 303:
            raise RuntimeError(f"web submit returned HTTP {response.status_code}")
        tid = response.headers["location"].rstrip("/").split("/")[-1]
        append_private(artifact / "lifecycle.jsonl", "submitted", parent_thread_id=tid,
                       queue_or_setup_ms=round((time.time() - accepted_at) * 1000, 2))
        service = threads._runs()
        delivered = False
        natural_candidate = False
        cancel_requested: set[tuple[str, str]] = set()
        while time.monotonic() < deadline:
            parent_runs = service.list(tid)
            children = _current_children(service, tid)
            for declined in _decline_private_egress(client, egress_routes, tid):
                append_private(artifact / "lifecycle.jsonl", "private_egress_declined",
                               parent_thread_id=tid, **declined)
            if (parent_runs and parent_runs[-1].status in {"error", "timeout", "cancelled"}
                    and not children):
                append_private(artifact / "lifecycle.jsonl", "parent_terminal_error",
                               parent_thread_id=tid, status=parent_runs[-1].status,
                               error=parent_runs[-1].error)
                break
            domain = threads_root / tid / "domain"
            report_dir = domain / "references"
            reports = [path for suffix in ("*.md", "*.org")
                       for path in sorted(report_dir.glob(suffix))
                       if path.is_file() and path.stat().st_size]
            originals = selected.get("workspace_files", {})
            updates = _saved_note_updates(domain, originals) if originals else {}
            artifact_ready = (_requested_updates(selected, updates)
                              if originals else bool(reports))
            completed_parent = _completed_parent_run(
                parent_runs, children, selected.get("required_delivery_role"))
            completed_child_id = _completed_child_id(completed_parent)
            reply = (_checkpoint_reply(threads_root / "threads.db", tid, completed_child_id)
                     if artifact_ready and completed_parent and not delivered else None)
            if reply is not None and not delivered:
                delivered = True
                if originals:
                    (artifact / "saved-note-updates.json").write_text(
                        json.dumps(updates, indent=2) + "\n")
                append_private(artifact / "lifecycle.jsonl",
                               "note_update_available" if originals else "report_available",
                               parent_thread_id=tid,
                               note_paths=list(updates),
                               report_paths=[str(path.relative_to(threads_root))
                                             for path in reports],
                               answer=reply[0], message_id=reply[1],
                               parent_run_id=completed_parent.id)
                append_private(artifact / "lifecycle.jsonl", "child_status_at_delivery",
                               children=[{"thread_id": child.thread_id,
                                          "run_id": child.id, "status": child.status}
                                         for child in children])
            if (not delivered and selected.get("required_delivery_role")
                    and not any(child.assistant_id == selected["required_delivery_role"]
                                for child in children)
                    and artifact_ready and children
                    and all(child.status in {"success", "error", "timeout", "cancelled"}
                            for child in children)):
                ordinary_parent = _completed_parent_run(parent_runs, children)
                ordinary_child_id = _completed_child_id(ordinary_parent)
                ordinary_reply = (_checkpoint_reply(threads_root / "threads.db", tid,
                                                    ordinary_child_id)
                                  if ordinary_parent else None)
                if ordinary_reply:
                    natural_candidate = True
                    append_private(artifact / "lifecycle.jsonl", "natural_delivery_candidate",
                                   parent_thread_id=tid, parent_run_id=ordinary_parent.id,
                                   answer=ordinary_reply[0], message_id=ordinary_reply[1],
                                   report_paths=[str(path.relative_to(threads_root))
                                                 for path in reports],
                                   verifier_integration="not_observed")
                    break
            if delivered:
                from manage.web.protocol_service import SERVICE
                for child in children:
                    key = child.thread_id, child.work_id
                    if (child.status not in {"success", "error", "timeout", "cancelled"}
                            and key not in cancel_requested):
                        SERVICE.cancel_run(child.thread_id, child.id)
                        cancel_requested.add(key)
                        append_private(artifact / "lifecycle.jsonl", "child_cancel_requested",
                                       child_thread_id=child.thread_id, run_id=child.id)
            if delivered and all(child.status in {"success", "error", "timeout", "cancelled"}
                                 for child in children):
                break
            with changed:
                changed.wait(timeout=min(60.0, max(0.0, deadline - time.monotonic())))
        observation_cutoff = (not delivered and not natural_candidate
                              and time.monotonic() >= deadline)
        if observation_cutoff:
            append_private(artifact / "lifecycle.jsonl", "observation_cutoff",
                           parent_thread_id=tid,
                           parent_statuses=[run.status for run in service.list(tid)],
                           child_statuses=[run.status for run in _current_children(service, tid)])
        from manage.web.protocol_service import SERVICE
        for child in _current_children(service, tid):
            key = child.thread_id, child.work_id
            if (child.status not in {"success", "error", "timeout", "cancelled"}
                    and key not in cancel_requested):
                SERVICE.cancel_run(child.thread_id, child.id)
                cancel_requested.add(key)
                append_private(artifact / "lifecycle.jsonl", "child_cancel_requested",
                               child_thread_id=child.thread_id, run_id=child.id)
        settle_by = time.monotonic() + 60
        while time.monotonic() < settle_by and any(
                child.status not in {"success", "error", "timeout", "cancelled"}
                for child in _current_children(service, tid)):
            with changed:
                changed.wait(timeout=min(5.0, settle_by - time.monotonic()))
    append_private(artifact / "lifecycle.jsonl", "worker_finished",
                   parent_thread_id=tid, delivered=delivered,
                   natural_candidate=natural_candidate,
                   observation_cutoff=observation_cutoff,
                   evaluator_cancel_requested=bool(cancel_requested),
                   parent_statuses=[run.status for run in service.list(tid)],
                   child_statuses=[run.status for run in _current_children(service, tid)])
    if not delivered and not natural_candidate:
        raise RuntimeError("no completed requested artifact before trial deadline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("egress_proof", type=Path)
    parser.add_argument("env_path", type=Path)
    parser.add_argument("--profile", choices=("web-main", "research-leaf",
                                              "research-manager-capability",
                                              "delegate-capability"),
                        default="web-main", help="web delivery or direct capability")
    args = parser.parse_args()
    run(args.case_id, args.artifact, args.egress_proof, args.env_path,
        profile=args.profile)
