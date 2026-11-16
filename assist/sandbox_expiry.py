"""Host age check for ownerless execution; live authority always owns teardown."""
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys

LABEL = "assist.turn-expiry"
MARKER = "host-3h-v1"
logger = logging.getLogger(__name__)


def _identity(generation):
    if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{64}", generation):
        raise ValueError("expiry requires an immutable Docker ID")
    return "assist-turn-expiry-" + generation


def _environment():
    # System services do not inherit their user's session bus variables.
    env = os.environ.copy()
    runtime = f"/run/user/{os.geteuid()}"
    env["XDG_RUNTIME_DIR"] = runtime
    env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={runtime}/bus"
    return env


def arm(generation, *, thread_id=None, threads_root=None, lifetime=10800):
    """Retain thread identity; callers must confirm teardown if arming fails."""
    unit = _identity(generation)
    created = subprocess.run(
        ["docker", "inspect", "--format", "{{.Created}}", generation],
        capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    deadline = datetime.fromisoformat(created.replace("Z", "+00:00")) + timedelta(seconds=lifetime)
    if deadline <= datetime.now(timezone.utc):
        raise RuntimeError("sandbox age-check time already passed")
    subprocess.run([
        "systemd-run", "--user", "--quiet", "--collect", "--unit", unit,
        "--on-calendar", deadline.strftime("%Y-%m-%d %H:%M:%S.%f UTC"),
        "--timer-property=AccuracySec=1us", "--property=Restart=on-failure",
        "--timer-property=RemainAfterElapse=no",
        "--property=RestartSec=10s", "--property=StartLimitIntervalSec=0",
        "--property=RestartPreventExitStatus=2", "--property=TimeoutStartSec=30s",
        sys.executable, str(Path(__file__).resolve()), generation,
        # Hex keeps systemd's specifier/environment expansion out of thread IDs.
        *((thread_id.encode("utf-8").hex(),) if thread_id is not None else ()),
        *((os.path.realpath(threads_root).encode("utf-8").hex(),)
          if threads_root is not None and thread_id is not None else ()),
    ], env=_environment(), capture_output=True, timeout=10, check=True)


def cancel(generation):
    """Best effort, only after the caller proved this exact generation absent."""
    try:
        unit = _identity(generation)
    except ValueError:
        return  # Legacy callers can stop named or shortened, untimed generations.
    try:
        subprocess.run(["systemctl", "--user", "stop", unit + ".timer", unit + ".service"],
                       env=_environment(), capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _event(generation, thread_id, outcome):
    logger.info("%s", json.dumps({
        "event": "sandbox_expiry", "timestamp": datetime.now(timezone.utc).isoformat(),
        "generation": generation, "thread_id": thread_id, "outcome": outcome,
    }))


def expire(generation, *, thread_id=None, threads_root=None):
    """Remove only provably ownerless execution under its existing authority fences."""
    _identity(generation)
    missing = re.compile(r"(?:error|error response from daemon): no such (?:object|container): "
                         + re.escape(generation), re.IGNORECASE)
    _event(generation, thread_id, "attempt")
    outcome = "inspect_failed"
    try:
        found = subprocess.run([
            "docker", "inspect", "--format",
            '{{index .Config.Labels "assist.sandbox"}} {{index .Config.Labels "' + LABEL + '"}}',
            generation], capture_output=True, text=True, timeout=5)
        if found.returncode:
            if missing.fullmatch(found.stderr.strip()):
                outcome = "already_absent"
                return
            raise RuntimeError("expiry could not inspect exact generation")
        if found.stdout.strip() != "true " + MARKER:
            outcome = "ownership_rejected"
            raise ValueError("expiry generation is not host-owned managed execution")
        outcome = "retained_unknown"
        # Older timers lack authority context. Never infer inactivity from age,
        # a missing Run, or a Docker-paused process. Normal cleanup/recovery owns it.
        if thread_id is None or threads_root is None:
            return
        from assist.browser import authority
        with ExitStack() as stack:
            try:
                stack.enter_context(authority.generation_fence(threads_root, thread_id, timeout=5))
                state = stack.enter_context(authority.fence(threads_root, thread_id))
            except (OSError, RuntimeError, ValueError, TimeoutError):
                return
            if not state.covered:
                return
            if state.lease is not None and generation in state.lease["generations"]:
                outcome = "retained_owned"
                return
            # Creation and later browser admission use these same fences. Keep
            # them through exact removal so an owner cannot claim it in between.
            outcome = "removal_unconfirmed"
            removed = subprocess.run(["docker", "rm", "--force", generation],
                                     capture_output=True, text=True, timeout=10)
            if removed.returncode:
                if not missing.fullmatch(removed.stderr.strip()):
                    raise RuntimeError("expiry exact generation removal is unconfirmed")
                outcome = "already_absent"
            else:
                outcome = "removed"
    finally:
        _event(generation, thread_id, outcome)


if __name__ == "__main__":
    # The transient unit invokes this exact deployed file from any working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        expire(sys.argv[1], thread_id=(
            bytes.fromhex(sys.argv[2]).decode("utf-8") if len(sys.argv) > 2 else None),
            threads_root=(bytes.fromhex(sys.argv[3]).decode("utf-8")
                          if len(sys.argv) > 3 else None))
    except (ValueError, IndexError):
        raise SystemExit(2)
