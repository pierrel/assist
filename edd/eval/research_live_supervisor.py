"""Bound one private web research trial and preserve its partial evidence.

The worker runs through shared model admission. This process owns the hard
wall deadline and reconciles only Runs and sandboxes under its private trial root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from edd.eval.research_trace import export_trace


def _fingerprint(root: Path) -> dict:
    digest = hashlib.sha256()
    count = 0
    for subtree in ("assist", "dockerfiles", "manage"):
        for path in sorted((root / subtree).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                relative = path.relative_to(root).as_posix()
                digest.update(relative.encode() + b"\0")
                digest.update(hashlib.sha256(path.read_bytes()).digest())
                count += 1
    return {"files": count, "sha256": digest.hexdigest()}


def _stop_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def _checkpoint_ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        if not connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='checkpoints'"
        ).fetchone():
            return []
        return [row[0] for row in connection.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")]
    finally:
        connection.close()


def _reconcile_private_runs(root: Path) -> dict:
    """Terminalize private Runs after their worker process has stopped."""
    from assist.run_service import RunService, TERMINAL_STATUSES

    if not root.is_absolute() or root != root.resolve():
        raise RuntimeError("private Runs root must be an absolute real path")
    service = RunService(str(root))
    def observed(run):
        return {name: getattr(run, name) for name in (
            "id", "thread_id", "work_id", "assistant_id", "mode", "status",
            "parent_thread_id", "parent_run_id", "created_at", "updated_at",
            "active_ms")}
    before = {}
    after = {}
    entries = sorted(root.iterdir())
    for directory in entries:
        if directory.is_symlink() or (directory / "runs.json").is_symlink():
            raise RuntimeError("private Runs path is a symlink")
    for directory in entries:
        runs_file = directory / "runs.json"
        if not directory.is_dir() or not runs_file.exists():
            continue
        thread_id = directory.name
        runs = service.list(thread_id)
        before[thread_id] = [observed(run) for run in runs]
        for run in runs:
            if run.status not in TERMINAL_STATUSES:
                service.cancel(thread_id, run.id)
        settled = service.list(thread_id)
        after[thread_id] = [observed(run) for run in settled]
        if any(run.status not in TERMINAL_STATUSES for run in settled):
            raise RuntimeError(f"private Run remains active: {thread_id}")
    return {"before": before, "after": after}


def preflight(case_id: str, product: Path, deployed: Path,
              *, profile: str = "web-main") -> tuple[dict, int]:
    """Prove the future worker's runtime and eval bytes before admission."""
    if profile not in {"web-main", "research-leaf", "research-manager-capability",
                       "delegate-capability"}:
        raise ValueError(f"unknown research evaluation profile: {profile}")
    source = _fingerprint(product)
    deployed_source = _fingerprint(deployed)
    if source != deployed_source:
        raise RuntimeError("private product differs from the declared deployed snapshot")
    eval_names = [
        "research_trace.py", "research_nonconvergence.py",
        *[path.name for path in sorted(Path(__file__).parent.glob("research_live_*"))
          if path.is_file() and path.suffix in {".py", ".json"}],
    ]
    source["eval_files"] = {}
    for name in eval_names:
        local = Path(__file__).parent / name
        copied = product / "edd" / "eval" / name
        if not copied.is_file() or copied.read_bytes() != local.read_bytes():
            raise RuntimeError(f"private snapshot has missing or changed eval overlay: {name}")
        source["eval_files"][name] = hashlib.sha256(local.read_bytes()).hexdigest()
    source["evaluation_profile"] = profile
    recorded = product.parent / "product-source.json"
    if recorded.is_file():
        source["deployed_source_record"] = json.loads(recorded.read_text())
    manifest = json.loads((Path(__file__).with_name("research_live_cases.json")).read_text())
    selected = next(case for case in manifest["cases"] if case["id"] == case_id)
    duration = (min(selected["deadline_seconds"], 600)
                if profile != "web-main" else selected["deadline_seconds"])
    if not 0 < duration <= 3600:
        raise ValueError("case exceeds one-hour hard wall")
    return source, duration


def run(case_id: str, artifact: Path, product: Path, deployed: Path,
        agentic_cli: Path, egress_proof: Path, env_path: Path,
        *, profile: str = "web-main") -> int:
    os.umask(0o077)
    if not artifact.is_absolute() or artifact != artifact.resolve():
        raise RuntimeError("private artifact must be an absolute real path")
    if artifact.exists():
        raise FileExistsError(artifact)
    source, duration = preflight(case_id, product, deployed, profile=profile)
    sys.path.insert(0, str(product))
    artifact.mkdir(mode=0o700, parents=True)
    (artifact / "source.json").write_text(json.dumps(source) + "\n")
    environment = os.environ.copy()
    production_threads = environment.get("AGENTIC_PRODUCTION_THREADS_DIR")
    if not production_threads or not Path(production_threads).is_absolute():
        raise RuntimeError("AGENTIC_PRODUCTION_THREADS_DIR must name the production run root")
    environment.update({
        "PYTHONPATH": str(product),
        "PYTHONDONTWRITEBYTECODE": "1",
        "RESEARCH_LIVE_PRODUCT_ROOT": str(product),
    })
    overlap = []
    if os.getenv("RESEARCH_LIVE_ALLOW_PRODUCTION_BUSY") == "1":
        reference = os.getenv("RESEARCH_LIVE_USER_MESSAGE_REF")
        if not reference:
            raise RuntimeError("production-busy override requires a current user message ref")
        overlap = ["--allow-production-busy", "--user-message-ref", reference]
    command = [str(agentic_cli), "resource", "run", *overlap, "llm", "--",
               sys.executable, "-m", "edd.eval.research_live_outcome", case_id,
               str(artifact), str(egress_proof), str(env_path), "--profile", profile]
    started = time.time()
    start_monotonic = time.monotonic()
    timed_out = False
    with (artifact / "worker.stdout.log").open("wb") as stdout, (
            artifact / "worker.stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, cwd="/tmp", env=environment,
                                   stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            process.wait(timeout=duration - 90)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_group(process)
        finally:
            _stop_group(process)
    # The existing orphan reaper filters Docker mounts by this exact private root.
    os.environ["ASSIST_THREADS_DIR"] = str(artifact / "threads")
    egress = json.loads(egress_proof.read_text())
    os.environ["ASSIST_EGRESS_APPROVALS_DIR"] = egress[
        "production_env_paths"]["ASSIST_EGRESS_APPROVALS_DIR"]
    os.environ["ASSIST_EGRESS_RUNTIME_DIR"] = egress[
        "reuse_validation"]["runtime_dir"]
    os.environ.pop("ASSIST_EGRESS_CLIENT_MAP_DIR", None)
    from assist.sandbox_manager import SandboxManager
    from assist.browser.manager import BrowserManager
    # This private eval must not prune unrelated client-map records.
    BrowserManager.prune_absent_clients = classmethod(lambda _cls, _map_dir: None)
    cleanup_error = None
    run_error = None
    run_reconciliation = None
    previous_alarm = signal.getsignal(signal.SIGALRM)

    def postflight_timeout(_signal, _frame):
        raise TimeoutError("private teardown exceeded remaining trial wall time")

    try:
        remaining = start_monotonic + duration - time.monotonic() - 5
        if remaining <= 0:
            raise TimeoutError("no time remains for private teardown")
        signal.signal(signal.SIGALRM, postflight_timeout)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        threads_root = artifact / "threads"
        if threads_root != threads_root.resolve():
            raise RuntimeError("private threads path is a symlink")
        if threads_root.is_dir():
            BrowserManager.reap_orphans(str(threads_root))
            SandboxManager.reap_orphans(str(threads_root))
    except Exception as error:
        cleanup_error = f"{type(error).__name__}: {error}"
    if cleanup_error is None and (artifact / "threads").is_dir():
        try:
            run_reconciliation = _reconcile_private_runs(artifact / "threads")
        except Exception as error:
            run_error = f"{type(error).__name__}: {error}"
    checkpoint = artifact / "threads" / "threads.db"
    ids = []
    export_error = None
    try:
        if cleanup_error is None and run_error is None:
            ids = _checkpoint_ids(checkpoint)
            if ids:
                export_trace(checkpoint, artifact / "checkpoint-latest.json", ids)
        else:
            export_error = "skipped because scoped teardown or Runs reconciliation failed"
    except Exception as error:
        export_error = f"{type(error).__name__}: {error}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
    summary = {
        "case_id": case_id, "evaluation_profile": profile,
        "started_at": started, "elapsed_seconds": time.monotonic() - start_monotonic,
        "hard_wall_seconds": duration, "timed_out": timed_out,
        "worker_exit_code": process.returncode, "checkpoint_threads": ids,
        "sandbox_reaper": "verified" if cleanup_error is None else cleanup_error,
        "run_reconciliation": run_reconciliation if run_error is None else run_error,
        "checkpoint_export": "verified" if export_error is None else export_error,
    }
    (artifact / "supervisor.json").write_text(json.dumps(summary, indent=2) + "\n")
    if cleanup_error is not None:
        raise RuntimeError("private sandbox teardown is unconfirmed: " + cleanup_error)
    if run_error is not None:
        raise RuntimeError("private Runs reconciliation failed: " + run_error)
    if export_error is not None:
        raise RuntimeError("private checkpoint export failed: " + export_error)
    return 124 if timed_out else process.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case_id", "artifact", "product", "deployed", "agentic_cli",
                 "egress_proof", "env_path"):
        parser.add_argument(name, type=Path if name != "case_id" else str)
    parser.add_argument("--profile", choices=("web-main", "research-leaf",
                                              "research-manager-capability",
                                              "delegate-capability"),
                        default="web-main", help="web delivery or direct-role capability")
    parser.add_argument("--preflight-only", action="store_true",
                        help="verify source/case bytes without model admission or side effects")
    args = parser.parse_args()
    if args.preflight_only:
        source, duration = preflight(args.case_id, args.product, args.deployed,
                                     profile=args.profile)
        print(json.dumps({"case_id": args.case_id, "profile": args.profile,
                          "deadline_seconds": duration, "source": source}, indent=2))
        raise SystemExit(0)
    raise SystemExit(run(args.case_id, args.artifact, args.product, args.deployed,
                         args.agentic_cli, args.egress_proof, args.env_path,
                         profile=args.profile))
