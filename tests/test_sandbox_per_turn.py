"""One container per turn: each request/response gets a fresh sandbox that is
killed at turn end, so a container's age can never exceed its turn's age.

This is what closes the mid-flight-reap caveat: because the container lives
only for its single turn and a turn is hard-capped at the LLM-queue hold
timeout, a wall-clock backstop set ABOVE that cap can never fire during a
legitimate in-progress turn (TestBackstopExceedsHoldCap pins exactly that).

Covered here:
  - cleanup() SIGKILLs (a sandbox has nothing to flush; PID 1 ignores SIGTERM).
  - get_sandbox_backend never reuses — a second call reaps the stale
    container and creates a fresh one.
  - the Dockerfile backstop TTL > the queue hold cap (the safety invariant).
  - real-Docker: a killed container is actually gone (symptom, un-mocked).
"""
from __future__ import annotations

import itertools
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from assist.sandbox_manager import SandboxManager


def _fake_container(cid: str) -> MagicMock:
    c = MagicMock()
    c.id = cid
    c.status = "running"
    return c


class _SandboxStateBase(TestCase):
    def setUp(self) -> None:
        self._saved = SandboxManager._containers
        SandboxManager._containers = {}

    def tearDown(self) -> None:
        SandboxManager._containers = self._saved


class TestCleanupKills(_SandboxStateBase):
    def test_cleanup_uses_sigkill(self):
        c = _fake_container("c0")
        SandboxManager._containers["w"] = c
        SandboxManager.cleanup("w")
        c.kill.assert_called_once_with()
        c.stop.assert_not_called()
        self.assertNotIn("w", SandboxManager._containers)  # registry pruned

    def test_cleanup_missing_workdir_is_noop(self):
        SandboxManager.cleanup("absent")  # must not raise


class TestNoReuse(_SandboxStateBase):
    """get_sandbox_backend creates a fresh container every call and reaps any
    stale one left in the registry (the "new container per request" guarantee).
    """

    def _patches(self):
        ids = itertools.count()
        client = MagicMock()
        client.containers.run.side_effect = lambda *a, **k: _fake_container(
            f"created-{next(ids)}")
        st = MagicMock()
        st.st_uid, st.st_gid = 1000, 1000
        return [
            patch.object(SandboxManager, "_get_docker_client", return_value=client),
            patch.object(SandboxManager, "_ensure_egress_proxy_running"),
            patch("assist.sandbox_manager.os.stat", return_value=st),
            patch("assist.sandbox.DockerSandboxBackend", lambda *a, **k: MagicMock()),
        ]

    def test_second_call_reaps_stale_and_creates_fresh(self):
        p = self._patches()
        with p[0], p[1], p[2], p[3]:
            SandboxManager.get_sandbox_backend("/ws/t")
            first = SandboxManager._containers["/ws/t"]
            # Second turn for the SAME thread: must NOT reuse `first`.
            SandboxManager.get_sandbox_backend("/ws/t")
            second = SandboxManager._containers["/ws/t"]

        self.assertIsNot(second, first, "container was reused across turns")
        first.kill.assert_called_once()  # stale one reaped (SIGKILL)
        # never calls reload() — there is no reuse path that would inspect it
        first.reload.assert_not_called()


class TestBackstopExceedsHoldCap(TestCase):
    """The load-bearing safety invariant: the container's wall-clock backstop
    TTL must exceed the LLM-queue hold cap.  Since a per-turn container's age
    equals its turn's age and a turn can't outlive the hold cap, a backstop
    above the cap can never reap a legitimate in-progress turn.  Guards against
    a careless lowering of the Dockerfile sleep back toward the old 1h value.
    """

    def test_dockerfile_sleep_exceeds_hold_timeout(self):
        from assist.thread_queue import DEFAULT_HOLD_TIMEOUT_S

        dockerfile = Path(__file__).resolve().parent.parent / "dockerfiles" / "Dockerfile.sandbox"
        text = dockerfile.read_text()
        m = re.search(r'CMD\s*\[\s*"sleep"\s*,\s*"(\d+)"\s*\]', text)
        self.assertIsNotNone(m, "no `sleep` backstop CMD found in Dockerfile.sandbox")
        backstop = int(m.group(1))
        self.assertGreater(
            backstop, DEFAULT_HOLD_TIMEOUT_S,
            f"backstop sleep {backstop}s must exceed the queue hold cap "
            f"{DEFAULT_HOLD_TIMEOUT_S}s — otherwise a legitimate long turn "
            f"(bounded by the hold cap) could be reaped mid-flight, "
            f"reintroducing the caveat this design removes",
        )


class TestPerTurnTeardownRealDocker(unittest.TestCase):
    """Real Docker (no skip, mirroring test_sandbox_egress_integration.py): a
    container reaped with kill=True is actually gone — the un-mocked symptom.
    """

    def test_kill_actually_destroys_the_container(self):
        import docker
        from docker.errors import NotFound

        client = docker.from_env()
        work_dir = tempfile.mkdtemp(prefix="per-turn-")
        try:
            # Start a bare sandbox container directly and register it, the
            # same shape get_sandbox_backend leaves in the registry.
            c = client.containers.run(
                "assist-sandbox", "sleep 10800", detach=True, remove=True,
                volumes={work_dir: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace", labels={"assist.sandbox": "true"},
            )
            SandboxManager._containers[work_dir] = c
            cid = c.id

            SandboxManager.cleanup(work_dir)

            self.assertNotIn(work_dir, SandboxManager._containers)
            # kill() is synchronous for the SIGKILL but Docker's --rm removal
            # is async, so poll: the container must actually disappear (proving
            # it was killed, not just dropped from the registry).
            import time
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    client.containers.get(cid)
                    time.sleep(0.1)
                except NotFound:
                    break
            else:
                self.fail("container still present 10s after kill() — not reaped")
        finally:
            SandboxManager._containers.pop(work_dir, None)
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
