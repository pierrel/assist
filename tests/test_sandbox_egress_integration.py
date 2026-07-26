"""End-to-end integration tests for the sandbox egress allowlist.

Unlike ``test_sandbox_egress.py`` (which mocks Docker) and the shell
smoke at ``dockerfiles/test-sandbox-egress.sh`` (which bypasses
SandboxManager), this exercises the *production* code path:

    SandboxManager.get_sandbox_backend(tmp_dir)
        → _ensure_egress_proxy_running (real proxy + network)
        → containers.run (real sandbox on internal network)
    → DockerSandboxBackend.execute("curl ...")
        → container.exec_run (same call the agent's tool makes)

If anything in that chain regresses — proxy bring-up, network attach,
env-var injection, hash-based recreate — these tests catch it.

No skip: the egress contract is too important to silently no-op.
Docker is pre-installed on ``ubuntu-latest`` GitHub runners (the
CI environment) and is always available on the deploy host.  If
Docker is genuinely missing, setUpClass fails loudly with a real
error — that's better than a silent skip that masks the regression.
"""
import shutil
import os
import tempfile
import unittest

from assist.sandbox_manager import SandboxManager


class TestSandboxEgressEndToEnd(unittest.TestCase):
    """Real Docker.  A single sandbox backend is created in
    ``setUpClass`` and shared across all tests in the class — each
    test runs its own ``curl`` through ``backend.execute`` but
    against the same container.  That's cheap and correct here
    because curl has no persistent state that one test could leak
    into another.

    Sets a 6s curl timeout — the proxy denies almost instantly (403)
    and a real allowed host responds in <2s; anything past that
    means something's hung and the test should fail loudly rather
    than block the suite for minutes.
    """

    CURL_TIMEOUT = 6

    @classmethod
    def setUpClass(cls):
        cls.work_dir = tempfile.mkdtemp(prefix="egress-e2e-")
        # SandboxManager refuses root-owned workspaces.  mkdtemp uses
        # the current uid, so we're fine.
        cls.backend = SandboxManager.get_sandbox_backend(cls.work_dir)
        if cls.backend is None:
            raise RuntimeError(
                "SandboxManager.get_sandbox_backend returned None — "
                "Docker daemon is up but the sandbox couldn't start.  "
                "Check that the egress-proxy image is built and the "
                "current user can bind-mount work_dir."
            )

    @classmethod
    def tearDownClass(cls):
        SandboxManager.cleanup(cls.work_dir)
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def _curl_status(self, url: str) -> tuple[int, str]:
        """Return (exit_code, output) of curl printing just the HTTP status."""
        cmd = (
            f"curl --max-time {self.CURL_TIMEOUT} -sS -o /dev/null "
            f"-w '%{{http_code}}' {url}"
        )
        resp = self.backend.execute(cmd)
        return resp.exit_code, resp.output

    @staticmethod
    def _reached_upstream(exit_code: int, output: str) -> bool:
        """True iff curl reached a real upstream — any 2xx or 3xx is
        evidence the request was allowed through.  A 3xx redirect is
        a real response from an upstream we shouldn't have reached,
        so it counts as a leak for negative cases.
        """
        if exit_code != 0:
            return False
        status = output.strip()
        return len(status) == 3 and status[0] in ("2", "3")

    def test_off_allowlist_host_is_denied(self):
        """Negative case: curl to a host not in the allowlist must NOT
        reach upstream.  The proxy filters on the CONNECT hostname and
        emits ``HTTP/1.1 403 Forbidden``; curl reports the failure
        either as a non-zero exit (connection refused / proxy error)
        or as "000" (no response from upstream — proxy closed the
        tunnel).
        """
        exit_code, output = self._curl_status("https://example.com/")
        self.assertFalse(
            self._reached_upstream(exit_code, output),
            f"Off-allowlist curl unexpectedly reached upstream: "
            f"exit={exit_code}, output={output!r}",
        )

    def test_allowlisted_host_succeeds(self):
        """Positive case: curl to an allowlisted host must reach the
        upstream and return a real HTTP response (typically 200).
        pypi.org is the canonical allowlisted host — covered by the
        committed ``dockerfiles/egress-allowlist.conf``.
        """
        exit_code, output = self._curl_status("https://pypi.org/")
        self.assertEqual(
            exit_code, 0,
            f"Allowlisted curl failed: exit={exit_code}, output={output!r}",
        )
        # pypi.org responds 200; the body is the status code line.
        # Tolerate 3xx (redirect) too — the test is "we reached
        # upstream", not "we got 200 specifically".
        status = output.strip()
        self.assertTrue(
            status.startswith("2") or status.startswith("3"),
            f"Expected 2xx/3xx from pypi.org, got: {status!r}",
        )

    def test_direct_ip_connect_is_denied(self):
        """Negative case: ``curl --resolve example.com:443:1.1.1.1``
        sends ``CONNECT 1.1.1.1:443`` to the proxy (the SNI / Host
        is set by --resolve, but the CONNECT line uses the IP
        literal).  The allowlist matches on the CONNECT hostname,
        which for a literal IP is never on the list.

        This is the DNS-bypass attack surface — agent hardcodes an
        IP to avoid hitting an allowlist that's filtered by name.
        """
        cmd = (
            f"curl --max-time {self.CURL_TIMEOUT} -sS -o /dev/null "
            f"-w '%{{http_code}}' --resolve example.com:443:1.1.1.1 "
            f"https://example.com/"
        )
        resp = self.backend.execute(cmd)
        self.assertFalse(
            self._reached_upstream(resp.exit_code, resp.output),
            f"Direct-IP curl unexpectedly reached upstream: "
            f"exit={resp.exit_code}, output={resp.output!r}",
        )


class TestThreadMemorySandboxContainment(unittest.TestCase):
    """A child container cannot read the parent's private bind mount."""

    def test_parent_file_and_shell_access_child_has_neither(self):
        thread_dir = tempfile.mkdtemp(prefix="agent-memory-e2e-")
        work_dir = os.path.join(thread_dir, "domain")
        agent_dir = os.path.join(thread_dir, "agent")
        os.makedirs(work_dir)
        os.makedirs(agent_dir)
        with open(os.path.join(agent_dir, "memory.md"), "w") as stream:
            stream.write("private-parent-canary")
        try:
            parent = SandboxManager.get_sandbox_backend(
                work_dir, agent_dir=agent_dir)
            parent_read = parent.read("/agent/memory.md")
            self.assertIsNone(parent_read.error)
            self.assertIn("private-parent-canary", parent_read.file_data["content"])
            self.assertEqual(
                parent.execute("grep -q private-parent-canary /agent/memory.md").exit_code,
                0)
            SandboxManager.cleanup(work_dir)

            child = SandboxManager.get_sandbox_backend(work_dir)
            child_read = child.read("/agent/memory.md")
            child_content = ((child_read.file_data or {}).get("content") or "")
            self.assertNotIn("private-parent-canary", child_content)
            self.assertEqual(child.execute("test ! -e /agent/memory.md").exit_code, 0)
        finally:
            SandboxManager.cleanup(work_dir)
            shutil.rmtree(thread_dir, ignore_errors=True)
