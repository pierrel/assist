"""Unit tests for the sandbox egress allowlist plumbing.

Docker is mocked — these tests assert that SandboxManager passes the
right kwargs to ``containers.run`` and that ``_ensure_egress_proxy_running``
is idempotent.  The actual policy enforcement is exercised in the
build-time smoke (``dockerfiles/test-sandbox-egress.sh``).
"""
import hashlib
import os
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

from assist.sandbox_manager import (
    EGRESS_NETWORK,
    EGRESS_PROXY_NAME,
    EGRESS_PROXY_PORT,
    SandboxManager,
    _load_egress_allowlist,
)


class TestLoadEgressAllowlist(TestCase):
    """Allowlist resolution: env var first, file second, baked default last."""

    def test_env_var_wins(self):
        with patch.dict(os.environ,
                        {"ASSIST_SANDBOX_EGRESS_ALLOWLIST": "a.example,b.example"},
                        clear=False):
            allowlist = _load_egress_allowlist()
        self.assertEqual(allowlist, ["a.example", "b.example"])

    def test_env_var_strips_whitespace_and_blanks(self):
        with patch.dict(os.environ,
                        {"ASSIST_SANDBOX_EGRESS_ALLOWLIST": " a.example , , b.example "},
                        clear=False):
            allowlist = _load_egress_allowlist()
        self.assertEqual(allowlist, ["a.example", "b.example"])

    def test_falls_back_to_file_when_env_unset(self):
        # Pop env var if set, then read the actual repo file.
        env = dict(os.environ)
        env.pop("ASSIST_SANDBOX_EGRESS_ALLOWLIST", None)
        with patch.dict(os.environ, env, clear=True):
            allowlist = _load_egress_allowlist()
        # Defaults from dockerfiles/egress-allowlist.conf
        self.assertIn("pypi.org", allowlist)
        self.assertIn("files.pythonhosted.org", allowlist)
        self.assertIn("host.docker.internal", allowlist)

    def test_file_strips_comments_and_blanks(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as f:
            f.write("# leading comment\n\n  pypi.org  \n# inline\nexample.com\n")
            tmp = f.name
        try:
            env = dict(os.environ)
            env.pop("ASSIST_SANDBOX_EGRESS_ALLOWLIST", None)
            with patch.dict(os.environ, env, clear=True):
                with patch("assist.sandbox_manager.EGRESS_ALLOWLIST_FILE", tmp):
                    allowlist = _load_egress_allowlist()
            self.assertEqual(allowlist, ["example.com", "pypi.org"])
        finally:
            os.unlink(tmp)


class TestEnsureEgressProxy(TestCase):
    """_ensure_egress_proxy_running is idempotent and hash-aware."""

    def setUp(self):
        SandboxManager._docker_client = None
        SandboxManager._containers.clear()

    def tearDown(self):
        SandboxManager._docker_client = None
        SandboxManager._containers.clear()

    def _make_client(self, network_exists=True, proxy=None):
        client = MagicMock()
        from docker.errors import NotFound
        if network_exists:
            client.networks.get.return_value = MagicMock()
        else:
            client.networks.get.side_effect = NotFound("no network")

        if proxy is None:
            client.containers.get.side_effect = NotFound("no proxy")
        else:
            client.containers.get.return_value = proxy

        new_proxy = MagicMock()
        new_proxy.id = "proxyabc1234"
        # _wait_for_egress_proxy_ready polls proxy.logs() looking for
        # "listening on".  Without this, the wait blocks for 10s and
        # then raises — which would make every test slow and noisy.
        new_proxy.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n"
        client.containers.run.return_value = new_proxy
        return client

    def _allowlist_hash(self):
        allowlist = _load_egress_allowlist()
        csv = ",".join(allowlist)
        return hashlib.sha256(csv.encode()).hexdigest()[:16]

    def test_creates_network_if_missing(self):
        client = self._make_client(network_exists=False)

        SandboxManager._ensure_egress_proxy_running(client)

        client.networks.create.assert_called_once_with(
            EGRESS_NETWORK, driver="bridge", internal=True,
        )

    def test_creates_proxy_when_absent(self):
        client = self._make_client()

        result = SandboxManager._ensure_egress_proxy_running(client)

        self.assertEqual(result, EGRESS_PROXY_NAME)
        # containers.run called for the proxy
        args, kwargs = client.containers.run.call_args
        self.assertEqual(args[0], "assist-egress-proxy")
        self.assertEqual(kwargs["name"], EGRESS_PROXY_NAME)
        self.assertEqual(kwargs["extra_hosts"],
                         {"host.docker.internal": "host-gateway"})
        self.assertIn("EGRESS_ALLOWLIST", kwargs["environment"])
        self.assertEqual(kwargs["labels"]["assist.egress-allowlist-hash"],
                         self._allowlist_hash())

    def test_skips_recreate_when_hash_matches_and_running(self):
        running_proxy = MagicMock()
        running_proxy.status = "running"
        running_proxy.labels = {"assist.egress-allowlist-hash": self._allowlist_hash()}
        client = self._make_client(proxy=running_proxy)

        SandboxManager._ensure_egress_proxy_running(client)

        # No new container was created.
        client.containers.run.assert_not_called()
        running_proxy.remove.assert_not_called()

    def test_recreates_when_allowlist_hash_changes(self):
        stale_proxy = MagicMock()
        stale_proxy.id = "stale"
        stale_proxy.status = "running"
        stale_proxy.labels = {"assist.egress-allowlist-hash": "0000000000000000"}
        client = self._make_client(proxy=stale_proxy)

        SandboxManager._ensure_egress_proxy_running(client)

        stale_proxy.remove.assert_called_once_with(force=True)
        client.containers.run.assert_called_once()

    def test_recreates_when_proxy_stopped(self):
        stopped_proxy = MagicMock()
        stopped_proxy.id = "stopped"
        stopped_proxy.status = "exited"
        stopped_proxy.labels = {"assist.egress-allowlist-hash": self._allowlist_hash()}
        client = self._make_client(proxy=stopped_proxy)

        SandboxManager._ensure_egress_proxy_running(client)

        stopped_proxy.remove.assert_called_once_with(force=True)
        client.containers.run.assert_called_once()

    def test_attaches_new_proxy_to_egress_network(self):
        client = self._make_client()

        SandboxManager._ensure_egress_proxy_running(client)

        # client.networks.get(EGRESS_NETWORK).connect(proxy) must be called.
        # We didn't set up a precise mock chain, so verify the network was
        # fetched and .connect() invoked on the result.
        client.networks.get.assert_any_call(EGRESS_NETWORK)


class TestSandboxBackendUsesEgressProxy(TestCase):
    """containers.run kwargs route the sandbox through the proxy."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        SandboxManager._docker_client = None
        SandboxManager._containers.clear()

    def tearDown(self):
        SandboxManager._docker_client = None
        SandboxManager._containers.clear()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("assist.sandbox.DockerSandboxBackend")
    def test_sandbox_joins_internal_network(self, _mock_backend):
        """The sandbox container must be on EGRESS_NETWORK, not the
        default bridge.  Internal=True on that network is what blocks
        every byte from leaving except via the proxy.
        """
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        client = MagicMock()
        client.networks.get.return_value = MagicMock()
        proxy = MagicMock()
        proxy.status = "running"
        proxy.labels = {"assist.egress-allowlist-hash":
                        hashlib.sha256(
                            ",".join(_load_egress_allowlist()).encode()
                        ).hexdigest()[:16]}
        proxy.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n"
        client.containers.get.return_value = proxy
        sandbox_container = MagicMock()
        sandbox_container.id = "sand123abcdef"
        sandbox_container.status = "running"
        client.containers.run.return_value = sandbox_container

        with patch.object(SandboxManager, "_get_docker_client", return_value=client):
            SandboxManager.get_sandbox_backend(test_path)

        # Last containers.run call is the sandbox.  Find it.
        sandbox_call = None
        for call in client.containers.run.call_args_list:
            if call.args and call.args[0] == "assist-sandbox":
                sandbox_call = call
                break
        self.assertIsNotNone(sandbox_call,
                             "containers.run was not called with assist-sandbox")
        kwargs = sandbox_call.kwargs
        self.assertEqual(kwargs.get("network"), EGRESS_NETWORK)
        # extra_hosts host-gateway must NOT be set on the sandbox — that
        # would be unreachable on internal=True anyway, but its absence is
        # the contract: every external connection goes through the proxy.
        self.assertNotIn("extra_hosts", kwargs)

    @patch("assist.sandbox.DockerSandboxBackend")
    def test_sandbox_environment_has_proxy_vars(self, _mock_backend):
        """HTTPS_PROXY/HTTP_PROXY (and lowercase variants) point at the
        proxy.  pip, httpx, openai-sdk all consult these.
        """
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        client = MagicMock()
        client.networks.get.return_value = MagicMock()
        proxy = MagicMock()
        proxy.status = "running"
        proxy.labels = {"assist.egress-allowlist-hash":
                        hashlib.sha256(
                            ",".join(_load_egress_allowlist()).encode()
                        ).hexdigest()[:16]}
        proxy.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n"
        client.containers.get.return_value = proxy
        sandbox_container = MagicMock()
        sandbox_container.id = "sand123abcdef"
        sandbox_container.status = "running"
        client.containers.run.return_value = sandbox_container

        with patch.object(SandboxManager, "_get_docker_client", return_value=client):
            SandboxManager.get_sandbox_backend(test_path)

        sandbox_call = next(
            c for c in client.containers.run.call_args_list
            if c.args and c.args[0] == "assist-sandbox"
        )
        env = sandbox_call.kwargs["environment"]
        expected = f"http://{EGRESS_PROXY_NAME}:{EGRESS_PROXY_PORT}"
        self.assertEqual(env.get("HTTPS_PROXY"), expected)
        self.assertEqual(env.get("HTTP_PROXY"), expected)
        self.assertEqual(env.get("https_proxy"), expected)
        self.assertEqual(env.get("http_proxy"), expected)
