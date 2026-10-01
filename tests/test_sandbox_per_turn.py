"""One container per turn, with confirmed stop before replacement.

The wall-clock TTL bounds orphan lifetime after a host outage. Exceeding the
normal queue hold cap reduces premature expiry; it is not a proof that a
force-released live turn cannot outlast the TTL.

Covered here:
  - cleanup() proves the captured generation stopped before registry release.
  - get_sandbox_backend never reuses — a second call reaps the stale
    container and creates a fresh one.
  - the Dockerfile backstop TTL > the queue hold cap (the safety invariant).
  - real-Docker: a killed container is actually gone (symptom, un-mocked).
"""
from __future__ import annotations

import itertools
import os
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
    def test_cleanup_requires_confirmed_stop(self):
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
        stopped.assert_called_once_with("old")

        with patch("assist.sandbox_manager.confirm_generation_stopped") as stopped:
            SandboxManager.cleanup("w", replacement)
        stopped.assert_called_once_with("replacement")


class TestNoReuse(_SandboxStateBase):
    """get_sandbox_backend creates a fresh container every call and reaps any
    stale one left in the registry (the "new container per request" guarantee).
    """

    def _patches(self, work_dir):
        ids = itertools.count()
        client = MagicMock()
        client.containers.run.side_effect = lambda *a, **k: _fake_container(
            f"created-{next(ids)}")
        st = MagicMock()
        st.st_uid, st.st_gid = 1000, 1000
        real_stat = os.stat
        return [
            patch.object(SandboxManager, "_get_docker_client", return_value=client),
            patch.object(SandboxManager, "_ensure_egress_proxy_running"),
            patch("assist.sandbox_manager.os.stat", side_effect=lambda path, **kwargs: (
                st if str(path) == work_dir else real_stat(path, **kwargs))),
            patch("assist.sandbox.DockerSandboxBackend", lambda *a, **k: MagicMock()),
            # Attribution is covered separately; this lifecycle test uses a
            # fake proxy without inspectable Docker mount attributes.
            patch.object(SandboxManager, "_record_egress_client"),
            patch("assist.sandbox_manager.confirm_generation_stopped"),
        ]

    def test_second_call_reaps_stale_and_creates_fresh(self):
        with tempfile.TemporaryDirectory(prefix="sandbox-no-reuse-") as root:
            from assist.browser import authority
            from assist.browser.manager import BrowserManager
            work_dir = str(Path(root) / "thread" / "domain")
            Path(work_dir).mkdir(parents=True)
            authority.mark_new_thread(root, "thread")
            p = self._patches(work_dir)
            with p[0], p[1], p[2], p[3], p[4], p[5] as stopped, \
                 patch.object(BrowserManager, "reconcile_startup", return_value=True), \
                 patch.object(BrowserManager, "confirm_owner_stopped", return_value=False):
                SandboxManager.get_sandbox_backend(
                    work_dir, thread_scope=(root, "thread"))
                first = SandboxManager._containers[work_dir]
                # Second turn for the SAME thread: must NOT reuse `first`.
                SandboxManager.get_sandbox_backend(
                    work_dir, thread_scope=(root, "thread"))
                second = SandboxManager._containers[work_dir]

        self.assertIsNot(second, first, "container was reused across turns")
        stopped.assert_called_once_with(first.id)
        # never calls reload() — there is no reuse path that would inspect it
        first.reload.assert_not_called()


class TestBackstopExceedsHoldCap(TestCase):
    """Keep the orphan TTL above the normal queue hold cap."""

    def test_dockerfile_sleep_exceeds_hold_timeout(self):
        from assist.thread_queue import DEFAULT_HOLD_TIMEOUT_S

        dockerfile = Path(__file__).resolve().parent.parent / "dockerfiles" / "Dockerfile.sandbox"
        text = dockerfile.read_text()
        m = re.search(r'CMD\s*\[\s*"sleep"\s*,\s*"(\d+)"\s*\]', text)
        self.assertIsNotNone(m, "no `sleep` backstop CMD found in Dockerfile.sandbox")
        backstop = int(m.group(1))
        self.assertGreater(
            backstop, DEFAULT_HOLD_TIMEOUT_S,
            f"backstop sleep {backstop}s must exceed the normal queue hold cap "
            f"{DEFAULT_HOLD_TIMEOUT_S}s",
        )


class TestPerTurnTeardownRealDocker(unittest.TestCase):
    """Explicit isolated Docker opt-in proves cleanup removes a container."""

    @unittest.skipUnless(os.getenv("ASSIST_SANDBOX_TEST_IMAGE"),
                         "requires an explicit isolated sandbox image tag")
    def test_kill_actually_destroys_the_container(self):
        import docker
        from docker.errors import NotFound

        client = docker.from_env()
        work_dir = tempfile.mkdtemp(prefix="per-turn-")
        try:
            # Start a bare sandbox container directly and register it, the
            # same shape get_sandbox_backend leaves in the registry.
            c = client.containers.run(
                os.environ["ASSIST_SANDBOX_TEST_IMAGE"], "sleep 10800",
                detach=True, remove=True,
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
