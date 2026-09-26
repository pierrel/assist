"""Unit tests for the sandbox egress allowlist plumbing.

Docker is mocked: proxy tests cover the low-level create request and
idempotent setup; shell tests cover ``containers.run`` and map admission.
The actual policy enforcement is exercised in the
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
    """Allowlist comes from a single source: dockerfiles/egress-allowlist.conf."""

    def test_reads_committed_repo_file(self):
        allowlist = _load_egress_allowlist()
        # Defaults from dockerfiles/egress-allowlist.conf
        self.assertIn("pypi.org", allowlist)
        self.assertIn("files.pythonhosted.org", allowlist)
        self.assertIn("host.docker.internal", allowlist)

    def test_strips_comments_and_blanks(self):
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as f:
            f.write("# leading comment\n\n  pypi.org  \n# inline\nexample.com\n")
            tmp = f.name
        try:
            with patch("assist.sandbox_manager.EGRESS_ALLOWLIST_FILE", tmp):
                allowlist = _load_egress_allowlist()
            self.assertEqual(allowlist, ["example.com", "pypi.org"])
        finally:
            os.unlink(tmp)

    def test_raises_fail_closed_when_file_missing(self):
        """No baked-in fallback — operator misconfiguration must not
        silently degrade to a permissive default.  Saved feedback memory:
        'Threads should die on infrastructure failure rather than
        heal-and-retry'.
        """
        with patch("assist.sandbox_manager.EGRESS_ALLOWLIST_FILE",
                   "/nonexistent/egress-allowlist.conf"):
            with self.assertRaises(FileNotFoundError):
                _load_egress_allowlist()


class TestEnsureEgressProxy(TestCase):
    """_ensure_egress_proxy_running is idempotent and hash-aware."""

    def setUp(self):
        SandboxManager._docker_client = None
        SandboxManager._containers.clear()
        self.runtime = tempfile.TemporaryDirectory(prefix="egress-ledger-test-")
        self.addCleanup(self.runtime.cleanup)
        env = patch.dict(os.environ, {"ASSIST_EGRESS_RUNTIME_DIR": self.runtime.name})
        env.start()
        self.addCleanup(env.stop)

    def tearDown(self):
        SandboxManager._docker_client = None
        SandboxManager._containers.clear()

    def _make_client(self, network_exists=True, proxy=None):
        client = MagicMock()
        client.api._version = "1.43"
        from docker.errors import NotFound
        net = MagicMock()
        net.attrs = {
            "Id": "ordinary-network-id", "Driver": "bridge", "Internal": True,
            "EnableIPv6": False,
            "IPAM": {"Config": [{"Subnet": "172.30.0.0/16"}]},
        }
        if network_exists:
            # Real Docker returns attrs with Internal=True for our network;
            # match that so the manager's "is this internal?" validation
            # passes by intent, not by MagicMock truthiness coincidence.
            client.networks.get.return_value = net
        else:
            created = {"network": False}
            def get_network(key):
                if not created["network"]:
                    raise NotFound("no network")
                return net
            client.networks.get.side_effect = get_network
            def create_network(_name, **kwargs):
                created["network"] = True
                net.attrs["Labels"] = kwargs.get("labels", {})
                return {"Id": "ordinary-network-id"}
            client.api.create_network.side_effect = create_network

        current = {"proxy": proxy}
        if proxy is None:
            pass
        def get_proxy(key):
            selected = current["proxy"]
            if selected is None:
                raise NotFound("no proxy")
            return selected
        client.containers.get.side_effect = get_proxy

        new_proxy = MagicMock()
        new_proxy.id = "proxyabc1234"
        new_proxy.attrs = {"NetworkSettings": {"Networks": {
            EGRESS_NETWORK: {"NetworkID": "ordinary-network-id"}}}}
        # _wait_for_egress_proxy_ready polls proxy.logs() looking for
        # "listening on".  Without this, the wait blocks for 10s and
        # then raises — which would make every test slow and noisy.
        new_proxy.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n"
        client.test_proxy = new_proxy
        def create_proxy(**kwargs):
            new_proxy.labels = kwargs.get("labels", {})
            current["proxy"] = new_proxy
            return {"Id": new_proxy.id}
        client.api.create_container.side_effect = create_proxy
        if proxy is not None:
            proxy.attrs = new_proxy.attrs
        return client

    def _allowlist_hash(self):
        # single-sourced from the prod formula so the two can't drift
        from assist.sandbox_manager import _egress_proxy_config_hash
        return _egress_proxy_config_hash(
            ",".join(_load_egress_allowlist()),
            None, "ordinary-network-id:172.30.0.0/16|:")

    def test_creates_network_if_missing(self):
        client = self._make_client(network_exists=False)

        SandboxManager._ensure_egress_proxy_running(client)

        args, kwargs = client.api.create_network.call_args
        self.assertEqual(args, (EGRESS_NETWORK,))
        self.assertEqual(kwargs["driver"], "bridge")
        self.assertTrue(kwargs["internal"])
        self.assertTrue(kwargs["labels"]["assist.egress-generation"])

    def test_creates_proxy_when_absent(self):
        client = self._make_client()

        result = SandboxManager._ensure_egress_proxy_running(client)

        self.assertEqual(result, EGRESS_PROXY_NAME)
        # The low-level POST is followed by inspect and a separate start.
        kwargs = client.api.create_container.call_args.kwargs
        self.assertEqual(kwargs["image"], "assist-egress-proxy")
        self.assertEqual(kwargs["name"], EGRESS_PROXY_NAME)
        self.assertIn("host.docker.internal:host-gateway",
                      kwargs["host_config"]["ExtraHosts"])
        self.assertIn("EGRESS_ALLOWLIST", kwargs["environment"])
        self.assertIn("EGRESS_THROTTLE_BODY", kwargs["environment"])
        self.assertEqual(kwargs["labels"]["assist.egress-allowlist-hash"],
                         self._allowlist_hash())
        client.test_proxy.start.assert_called_once()

    def test_proxy_runs_as_nondefault_service_uid(self):
        from assist.egress import runtime_state
        client = self._make_client()
        with (patch("assist.sandbox_manager.os.getuid", return_value=1001),
              patch("assist.sandbox_manager.os.getgid", return_value=1002),
              patch.object(runtime_state, "runtime_directory", return_value=self.runtime.name)):
            SandboxManager._ensure_egress_proxy_running(client)
        self.assertEqual(client.api.create_container.call_args.kwargs["user"], "1001:1002")

    def test_proxy_never_overrides_nonroot_image_user_with_root(self):
        from assist.egress import runtime_state
        client = self._make_client()
        with (patch("assist.sandbox_manager.os.getuid", return_value=0),
              patch.object(runtime_state, "runtime_directory", return_value=self.runtime.name)):
            with self.assertRaisesRegex(RuntimeError, "cannot run as root"):
                SandboxManager._ensure_egress_proxy_running(client)
        client.api.create_container.assert_not_called()

    def test_skips_recreate_when_hash_matches_and_running(self):
        running_proxy = MagicMock()
        running_proxy.status = "running"
        running_proxy.labels = {"assist.egress-allowlist-hash": self._allowlist_hash()}
        client = self._make_client(proxy=running_proxy)

        SandboxManager._ensure_egress_proxy_running(client)

        # No new container was created.
        client.api.create_container.assert_not_called()
        running_proxy.remove.assert_not_called()

    def test_recreates_when_allowlist_hash_changes(self):
        stale_proxy = MagicMock()
        stale_proxy.id = "stale"
        stale_proxy.status = "running"
        stale_proxy.labels = {"assist.egress-allowlist-hash": "0000000000000000"}
        client = self._make_client(proxy=stale_proxy)

        SandboxManager._ensure_egress_proxy_running(client)

        stale_proxy.remove.assert_called_once_with(force=True)
        client.api.create_container.assert_called_once()

    def test_recreates_when_proxy_policy_schema_changes(self):
        """An older image cannot stay alive behind a matching allowlist.

        Host throttling lives inside the proxy image, so its schema marker is
        part of the label contract that forces a replacement after deployment.
        """
        old_hash = hashlib.sha256(
            (",".join(_load_egress_allowlist()) + "|v2-approvals:").encode()
        ).hexdigest()[:16]
        stale_proxy = MagicMock()
        stale_proxy.id = "old-policy"
        stale_proxy.status = "running"
        stale_proxy.labels = {"assist.egress-allowlist-hash": old_hash}
        client = self._make_client(proxy=stale_proxy)

        SandboxManager._ensure_egress_proxy_running(client)

        stale_proxy.remove.assert_called_once_with(force=True)
        client.api.create_container.assert_called_once()

    def test_replaces_running_host_only_proxy_before_port_policy_admission(self):
        """A rebuilt image alone cannot update an already-running v5 proxy."""
        old_hash = hashlib.sha256((
            ",".join(_load_egress_allowlist())
            + "|v5-browser-policy:||ordinary-network-id:172.30.0.0/16|:"
        ).encode()).hexdigest()[:16]
        stale_proxy = MagicMock()
        stale_proxy.id = "old-host-only-proxy"
        stale_proxy.status = "running"
        stale_proxy.labels = {"assist.egress-allowlist-hash": old_hash}
        client = self._make_client(proxy=stale_proxy)

        SandboxManager._ensure_egress_proxy_running(client)

        stale_proxy.remove.assert_called_once_with(force=True)
        client.api.create_container.assert_called_once()
        self.assertNotEqual(old_hash, self._allowlist_hash())

    def test_recreates_when_proxy_stopped(self):
        stopped_proxy = MagicMock()
        stopped_proxy.id = "stopped"
        stopped_proxy.status = "exited"
        stopped_proxy.labels = {"assist.egress-allowlist-hash": self._allowlist_hash()}
        client = self._make_client(proxy=stopped_proxy)

        SandboxManager._ensure_egress_proxy_running(client)

        stopped_proxy.remove.assert_called_once_with(force=True)
        client.api.create_container.assert_called_once()

    def test_attaches_new_proxy_to_egress_network(self):
        client = self._make_client()

        SandboxManager._ensure_egress_proxy_running(client)

        # Network is fetched + validated (internal=True) by the manager.
        client.networks.get.assert_any_call(EGRESS_NETWORK)
        # And the new proxy container is then attached to it via
        # egress_net.connect(proxy).  This is the load-bearing step —
        # without it the proxy lives only on the default bridge and
        # sandboxes can't reach it.
        egress_net = client.networks.get.return_value
        new_proxy = client.test_proxy
        egress_net.connect.assert_called_once_with(new_proxy)

    def test_rejects_non_internal_egress_network(self):
        """If a same-named network exists but isn't internal=True
        (e.g., hand-rolled `docker network create` without --internal),
        attaching the sandbox would re-open unrestricted egress.
        Fail-closed: raise RuntimeError instead of silently bypassing.
        """
        client = self._make_client()
        # Override: network exists but is NOT internal.
        non_internal = MagicMock()
        non_internal.attrs = {"Internal": False}
        client.networks.get.return_value = non_internal

        with self.assertRaises(RuntimeError) as ctx:
            SandboxManager._ensure_egress_proxy_running(client)
        self.assertIn("not internal", str(ctx.exception).lower())


class TestSandboxBackendUsesEgressProxy(TestCase):
    """containers.run kwargs route the sandbox through the proxy."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.runtime = tempfile.TemporaryDirectory(prefix="egress-ledger-test-")
        self.addCleanup(self.runtime.cleanup)
        env = patch.dict(os.environ, {"ASSIST_EGRESS_RUNTIME_DIR": self.runtime.name})
        env.start()
        self.addCleanup(env.stop)
        SandboxManager._docker_client = None
        SandboxManager._containers.clear()

    def tearDown(self):
        SandboxManager._docker_client = None
        SandboxManager._containers.clear()
        SandboxManager._egress_client_ips.clear()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _backend_client(self, mounts):
        client = MagicMock()
        proxy = MagicMock()
        proxy.status = "running"
        proxy.attrs = {"Mounts": mounts}
        client.containers.get.return_value = proxy
        sandbox = MagicMock()
        sandbox.id = "sandbox-generation"
        client.containers.run.return_value = sandbox
        return client, sandbox

    @patch("assist.sandbox.DockerSandboxBackend")
    def test_invalid_browser_map_preserves_verified_base_only_shell(self, _backend):
        work_dir = os.path.join(self.temp_dir, "thread", "domain")
        os.makedirs(work_dir)
        client, sandbox = self._backend_client([])
        with (patch.object(SandboxManager, "_get_docker_client", return_value=client),
              patch.object(SandboxManager, "_ensure_egress_proxy_running",
                           return_value=EGRESS_PROXY_NAME),
              patch("assist.egress.client_map.configured_directory",
                    side_effect=RuntimeError("unsafe browser map"))):
            self.assertIsNotNone(SandboxManager.get_sandbox_backend(work_dir))
        sandbox.kill.assert_not_called()

    @patch("assist.sandbox.DockerSandboxBackend")
    def test_map_mounted_proxy_cannot_expose_unattributed_shell(self, _backend):
        work_dir = os.path.join(self.temp_dir, "thread", "domain")
        os.makedirs(work_dir)
        for unavailable in (None, RuntimeError("unsafe browser map")):
            client, sandbox = self._backend_client([
                {"Destination": "/client-map", "Source": "/host/old-map"}])
            with (patch.object(SandboxManager, "_get_docker_client", return_value=client),
                  patch.object(SandboxManager, "_ensure_egress_proxy_running",
                               return_value=EGRESS_PROXY_NAME),
                  patch("assist.egress.client_map.configured_directory",
                        side_effect=unavailable if isinstance(unavailable, Exception)
                        else None, return_value=unavailable)):
                with self.assertRaisesRegex(RuntimeError, "map mode is unconfirmed"):
                    SandboxManager.get_sandbox_backend(work_dir)
            sandbox.kill.assert_called_once()

    @patch("assist.sandbox.DockerSandboxBackend")
    def test_proxy_map_path_must_match_record_path(self, _backend):
        work_dir = os.path.join(self.temp_dir, "thread", "domain")
        os.makedirs(work_dir)
        client, sandbox = self._backend_client([
            {"Destination": "/client-map", "Source": "/host/old-map"}])
        with (patch.object(SandboxManager, "_get_docker_client", return_value=client),
              patch.object(SandboxManager, "_ensure_egress_proxy_running",
                           return_value=EGRESS_PROXY_NAME),
              patch("assist.egress.client_map.configured_directory",
                    return_value="/host/new-map")):
            with self.assertRaisesRegex(RuntimeError, "map mode is unconfirmed"):
                SandboxManager.get_sandbox_backend(work_dir)
        sandbox.kill.assert_called_once()

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
        client.networks.get.return_value.attrs = {
            "Id": "ordinary-network-id", "Driver": "bridge", "Internal": True,
            "EnableIPv6": False,
            "IPAM": {"Config": [{"Subnet": "172.30.0.0/16"}]},
        }
        proxy = MagicMock()
        proxy.status = "running"
        proxy.attrs = {"Mounts": [], "NetworkSettings": {"Networks": {
            EGRESS_NETWORK: {"NetworkID": "ordinary-network-id"}}}}
        from assist.sandbox_manager import _egress_proxy_config_hash
        proxy.labels = {"assist.egress-allowlist-hash":
                        _egress_proxy_config_hash(
                            ",".join(_load_egress_allowlist()), None,
                            "ordinary-network-id:172.30.0.0/16|:")}
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
        client.networks.get.return_value.attrs = {
            "Id": "ordinary-network-id", "Driver": "bridge", "Internal": True,
            "EnableIPv6": False,
            "IPAM": {"Config": [{"Subnet": "172.30.0.0/16"}]},
        }
        proxy = MagicMock()
        proxy.status = "running"
        proxy.attrs = {"Mounts": [], "NetworkSettings": {"Networks": {
            EGRESS_NETWORK: {"NetworkID": "ordinary-network-id"}}}}
        from assist.sandbox_manager import _egress_proxy_config_hash
        proxy.labels = {"assist.egress-allowlist-hash":
                        _egress_proxy_config_hash(
                            ",".join(_load_egress_allowlist()), None,
                            "ordinary-network-id:172.30.0.0/16|:")}
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
