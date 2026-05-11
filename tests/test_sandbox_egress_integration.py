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

Auto-skips when Docker isn't available or the required images aren't
built, so ``make test`` on a developer machine without Docker still
passes.
"""
import os
import shutil
import tempfile
import unittest

from assist.sandbox_manager import SandboxManager


def _docker_available() -> bool:
    """True iff the local Docker daemon is reachable and both required
    images are built.  Either condition missing → skip the test class.
    """
    try:
        import docker
    except ImportError:
        return False
    try:
        client = docker.from_env()
        client.ping()
    except Exception:
        return False
    needed = {"assist-sandbox", "assist-egress-proxy"}
    have = set()
    for img in client.images.list():
        for tag in (img.tags or []):
            for name in needed:
                if tag.startswith(name + ":"):
                    have.add(name)
    return needed <= have


@unittest.skipUnless(
    _docker_available(),
    "Docker + assist-sandbox + assist-egress-proxy images required",
)
class TestSandboxEgressEndToEnd(unittest.TestCase):
    """Real Docker.  Each test spins up a fresh sandbox via
    SandboxManager and runs curl through ``backend.execute``.

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

    def test_off_allowlist_host_is_denied(self):
        """Negative case: curl to a host not in the allowlist must NOT
        return a 2xx.  The proxy filters on the CONNECT hostname and
        emits ``HTTP/1.1 403 Forbidden``; curl reports the failure
        either as a non-zero exit (connection refused / proxy error)
        or as "000" (no response from upstream — proxy closed the
        tunnel).  Either way: not 200.
        """
        exit_code, output = self._curl_status("https://example.com/")
        # example.com is not on the allowlist.  Acceptable evidence
        # of denial: non-zero exit OR an explicit 403/000 status in
        # the body.  Reject only a successful (2xx) response.
        is_success = (exit_code == 0 and output.strip().startswith("2"))
        self.assertFalse(
            is_success,
            f"Off-allowlist curl unexpectedly succeeded: "
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
        is_success = (resp.exit_code == 0 and resp.output.strip().startswith("2"))
        self.assertFalse(
            is_success,
            f"Direct-IP curl unexpectedly succeeded: "
            f"exit={resp.exit_code}, output={resp.output!r}",
        )
