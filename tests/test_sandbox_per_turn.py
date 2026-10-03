"""One container per turn: each request/response gets a fresh sandbox.
Successful cleanup confirms it stopped at turn end. Failed confirmation keeps
the generation tracked and may leave it alive past the turn.

The active turn is hard-capped at the LLM-queue hold timeout, so a wall-clock
backstop set ABOVE that cap cannot fire during a legitimate in-progress turn
(TestBackstopExceedsHoldCap pins exactly that).

Covered here:
  - cleanup() confirms the captured generation stopped before registry release.
  - get_sandbox_backend confirms a stale generation before creating a fresh one.
  - the Dockerfile backstop TTL > the queue hold cap (the safety invariant).
  - real-Docker: cleanup confirms the container is gone (un-mocked).
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


class TestCleanupGeneration(_SandboxStateBase):
    def test_cleanup_requests_exact_generation_confirmation(self):
        c = _fake_container("c0")
        SandboxManager._containers["w"] = c
        with patch("assist.sandbox_manager.confirm_generation_stopped") as stopped:
            SandboxManager.cleanup("w")
        stopped.assert_called_once_with("c0")
        c.stop.assert_not_called()
        self.assertNotIn("w", SandboxManager._containers)  # registry pruned

    def test_cleanup_missing_workdir_is_noop(self):
        SandboxManager.cleanup("absent")  # must not raise

    def test_stale_owner_cannot_kill_replacement(self):
        old = _fake_container("old")
        replacement = _fake_container("replacement")
        SandboxManager._containers["w"] = replacement

        with patch("assist.sandbox_manager.confirm_generation_stopped") as stopped:
            SandboxManager.cleanup("w", old)
        self.assertIs(SandboxManager._containers["w"], replacement)
        replacement.kill.assert_not_called()
        stopped.assert_called_once_with("old")

        with patch("assist.sandbox_manager.confirm_generation_stopped") as stopped:
            SandboxManager.cleanup("w", replacement)
        stopped.assert_called_once_with("replacement")


class TestNoReuse(_SandboxStateBase):
    """get_sandbox_backend confirms stale cleanup before creating a fresh one.
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
            # the persistent-/tmp dir is created for real; the work_dir here is a
            # fake path, so stub the mkdir like os.stat above.
            patch("assist.sandbox_manager.os.makedirs"),
            patch("assist.sandbox_manager.confirm_generation_stopped"),
        ]

    def test_second_call_confirms_stale_and_creates_fresh(self):
        p = self._patches()
        with p[0], p[1], p[2], p[3], p[4], p[5] as stopped:
            SandboxManager.get_sandbox_backend("/ws/t")
            first = SandboxManager._containers["/ws/t"]
            # Second turn for the SAME thread: must NOT reuse `first`.
            SandboxManager.get_sandbox_backend("/ws/t")
            second = SandboxManager._containers["/ws/t"]

        self.assertIsNot(second, first, "container was reused across turns")
        stopped.assert_called_once_with(first.id)
        # never calls reload() — there is no reuse path that would inspect it
        first.reload.assert_not_called()


class TestBackstopExceedsHoldCap(TestCase):
    """The load-bearing safety invariant: the container's wall-clock backstop
    TTL must exceed the LLM-queue hold cap. During an active turn, container
    age cannot exceed the hold cap; failed cleanup may retain it afterward.
    A backstop above the cap cannot reap a legitimate in-progress turn. This
    guards against lowering the Dockerfile sleep back toward the old 1h value.
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
    container cleaned up through exact-ID confirmation is absent afterward.
    """

    def test_cleanup_confirms_container_absent(self):
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
            # cleanup() already waited for absence; verify it through the SDK too.
            with self.assertRaises(NotFound):
                client.containers.get(cid)
        finally:
            SandboxManager._containers.pop(work_dir, None)
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
