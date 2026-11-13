"""Host-owned deadline for managed execution, including Docker-paused time."""
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
import subprocess
import sys

LABEL = "assist.turn-expiry"
MARKER = "host-3h-v1"


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


def arm(generation, *, lifetime=10800):
    """Arm once at creation; failure must be followed by confirmed teardown."""
    unit = _identity(generation)
    created = subprocess.run(
        ["docker", "inspect", "--format", "{{.Created}}", generation],
        capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    deadline = datetime.fromisoformat(created.replace("Z", "+00:00")) + timedelta(seconds=lifetime)
    if deadline <= datetime.now(timezone.utc):
        raise RuntimeError("sandbox deadline already expired")
    subprocess.run([
        "systemd-run", "--user", "--quiet", "--collect", "--unit", unit,
        "--on-calendar", deadline.strftime("%Y-%m-%d %H:%M:%S.%f UTC"),
        "--timer-property=AccuracySec=1us", "--property=Restart=on-failure",
        "--timer-property=RemainAfterElapse=no",
        "--property=RestartSec=10s", "--property=StartLimitIntervalSec=0",
        "--property=RestartPreventExitStatus=2", "--property=TimeoutStartSec=20s",
        sys.executable, str(Path(__file__).resolve()), generation,
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


def expire(generation):
    """Remove only the exact host-labelled managed execution generation."""
    _identity(generation)
    missing = re.compile(r"(?:error|error response from daemon): no such (?:object|container): "
                         + re.escape(generation), re.IGNORECASE)
    found = subprocess.run([
        "docker", "inspect", "--format",
        '{{index .Config.Labels "assist.sandbox"}} {{index .Config.Labels "' + LABEL + '"}}',
        generation], capture_output=True, text=True, timeout=5)
    if found.returncode:
        if missing.fullmatch(found.stderr.strip()):
            return
        raise RuntimeError("expiry could not inspect exact generation")
    if found.stdout.strip() != "true " + MARKER:
        raise ValueError("expiry generation is not host-owned managed execution")
    removed = subprocess.run(["docker", "rm", "--force", generation],
                             capture_output=True, text=True, timeout=10)
    if removed.returncode and not missing.fullmatch(removed.stderr.strip()):
        raise RuntimeError("expiry exact generation removal is unconfirmed")


if __name__ == "__main__":
    try:
        expire(sys.argv[1])
    except (ValueError, IndexError):
        raise SystemExit(2)
