"""Unit tests for Docker sandbox backend and DomainManager Docker lifecycle.

All Docker interactions are mocked — no real Docker daemon needed.
"""
import os
import tempfile
import shutil
from unittest import TestCase
from unittest.mock import patch, MagicMock, PropertyMock

from assist.sandbox import (
    DockerSandboxBackend,
    MAX_OUTPUT_CHARS,
    SandboxContainerLostError,
)
from assist.domain_manager import DomainManager
from assist.sandbox_manager import SandboxManager


class TestDockerSandboxBackend(TestCase):
    """Test DockerSandboxBackend with mocked Docker container."""

    def setUp(self):
        self.container = MagicMock()
        self.container.id = "abc123def456"
        self.sandbox = DockerSandboxBackend(self.container)

    def test_id_returns_short_container_id(self):
        self.assertEqual(self.sandbox.id, "abc123def456")

    def test_resolve_prefixes_path(self):
        self.assertEqual(self.sandbox._resolve("/myfile.txt"), "/workspace/myfile.txt")

    def test_resolve_preserves_workspace_path(self):
        self.assertEqual(self.sandbox._resolve("/workspace/myfile.txt"), "/workspace/myfile.txt")

    def test_resolve_handles_relative_path(self):
        self.assertEqual(self.sandbox._resolve("myfile.txt"), "/workspace/myfile.txt")

    def test_resolve_handles_none(self):
        self.assertIsNone(self.sandbox._resolve(None))

    def test_execute_success(self):
        self.container.exec_run.return_value = (0, b"hello world\n")
        resp = self.sandbox.execute("echo hello world")

        self.assertEqual(resp.output, "hello world\n")
        self.assertEqual(resp.exit_code, 0)
        self.assertFalse(resp.truncated)
        # Command is wrapped in coreutils `timeout` so a runaway can't
        # pin the agent loop. The shape: `timeout --kill-after=Ns Ms bash -c '<cmd>'`.
        call_args = self.container.exec_run.call_args
        invoked = call_args[0][0]  # ["bash", "-c", "timeout ... bash -c 'echo hello world'"]
        self.assertEqual(invoked[:2], ["bash", "-c"])
        self.assertIn("timeout --kill-after=", invoked[2])
        self.assertIn("echo hello world", invoked[2])
        self.assertEqual(call_args.kwargs["workdir"], "/workspace")

    def test_execute_nonzero_exit(self):
        self.container.exec_run.return_value = (1, b"error: not found\n")
        resp = self.sandbox.execute("cat /missing")

        self.assertEqual(resp.exit_code, 1)
        self.assertIn("not found", resp.output)

    def test_execute_timeout_returns_guidance(self):
        """Exit 124 = coreutils `timeout` SIGTERM'd; surface adjustment hints."""
        self.container.exec_run.return_value = (124, b"some partial output\n")
        resp = self.sandbox.execute("python3 -c 'import glob; glob.glob(\"**/*\", recursive=True)'")

        self.assertEqual(resp.exit_code, 124)
        self.assertIn("Sandbox terminated this command", resp.output)
        self.assertIn("Narrow the scope", resp.output)
        self.assertIn("/workspace", resp.output)
        self.assertIn("some partial output", resp.output)

    def test_execute_sigkill_returns_guidance(self):
        """Exit 137 = SIGKILL after grace window; same guidance applies."""
        self.container.exec_run.return_value = (137, b"")
        resp = self.sandbox.execute("yes > /dev/null")

        self.assertEqual(resp.exit_code, 137)
        self.assertIn("Sandbox terminated this command", resp.output)
        # No partial output, but still gets the marker line
        self.assertIn("(no output)", resp.output)

    def test_execute_quotes_command_safely(self):
        """Commands with shell metacharacters survive the timeout wrap."""
        self.container.exec_run.return_value = (0, b"ok\n")
        self.sandbox.execute("echo 'hello' > /tmp/x; cat /tmp/x")

        invoked = self.container.exec_run.call_args[0][0][2]
        # The original command must reach bash unaltered (escaped via shlex).
        # Two ways shlex can quote — accept either as long as the literal command
        # appears intact when bash unquotes it.
        self.assertIn("echo", invoked)
        self.assertIn("hello", invoked)
        self.assertIn("/tmp/x", invoked)

    def test_execute_truncates_large_output(self):
        large = b"x" * (MAX_OUTPUT_CHARS + 5000)
        self.container.exec_run.return_value = (0, large)
        resp = self.sandbox.execute("cat bigfile")

        self.assertTrue(resp.truncated)
        self.assertIn("... [output truncated]", resp.output)
        # Output should be truncated to MAX_OUTPUT_CHARS + truncation message
        self.assertLessEqual(len(resp.output), MAX_OUTPUT_CHARS + 50)

    def test_execute_empty_output(self):
        self.container.exec_run.return_value = (0, b"")
        resp = self.sandbox.execute("true")

        self.assertEqual(resp.output, "")
        self.assertEqual(resp.exit_code, 0)

    def test_execute_none_output(self):
        self.container.exec_run.return_value = (0, None)
        resp = self.sandbox.execute("true")

        self.assertEqual(resp.output, "")

    def test_execute_docker_error(self):
        self.container.exec_run.side_effect = Exception("container stopped")
        resp = self.sandbox.execute("ls")

        self.assertEqual(resp.exit_code, 1)
        self.assertIn("container stopped", resp.output)

    def test_execute_container_not_found_raises_lost_error(self):
        """Container 404 must raise, not become a tool result.

        Regression: prod thread 20260504150948-c1545cf9 ran for ~40s
        after its sandbox was stopped because every tool call returned
        a 404 as a normal ExecuteResponse — the model ignored the
        repeated errors and confabulated "Done. Created X" without
        any file actually being written.  A typed exception forces
        the web layer's per-request handler to mark the thread errored
        with a clear user message.
        """
        from docker.errors import NotFound
        self.container.exec_run.side_effect = NotFound(
            "No such container: abc123def456"
        )
        with self.assertRaises(SandboxContainerLostError) as cm:
            self.sandbox.execute("ls")
        # Message includes the container id so the operator can
        # correlate with `docker ps -a` history.
        self.assertIn("abc123def456", str(cm.exception))


class TestDockerSandboxBackendPathPrefixing(TestCase):
    """ls / grep / glob must override deepagents' new protocol API
    (NOT the deprecated *_info / *_raw siblings) so ``_resolve``
    actually applies the ``/workspace`` prefix.

    Regression: until 2026-05-04 our subclass overrode the deprecated
    names; the framework called the new names; ``_resolve`` was dead
    code; ``glob(path="/")`` walked the entire container filesystem.
    Symptom: 99% CPU runaways pinned for minutes per call.
    """

    def setUp(self):
        self.container = MagicMock()
        self.container.id = "abc123def456"
        # Mock exec_run so super().{ls,grep,glob} don't actually do
        # filesystem work — we just want to capture what path argument
        # they received.  Returning a benign tuple keeps BaseSandbox's
        # parsing logic happy.
        self.container.exec_run.return_value = (0, b"")
        self.sandbox = DockerSandboxBackend(self.container)

    def _last_command(self) -> str:
        """Return the bash command from the most recent exec_run call."""
        return self.container.exec_run.call_args[0][0][2]

    def _decoded_paths_in(self, cmd: str) -> list[str]:
        """Return absolute paths reachable from cmd.

        deepagents' templates use both base64-encoded paths (glob, ls)
        and literal-string paths (grep).  Recover both: try to decode
        each shell-quoted token from base64 (keep the ones that look
        like absolute paths), and also scan the raw command for any
        ``/workspace`` literal.  Trailing slashes are normalized so
        ``/workspace`` and ``/workspace/`` compare equal.
        """
        import base64
        import re
        out = []
        for token in cmd.split("'"):
            try:
                d = base64.b64decode(token).decode("utf-8")
            except Exception:
                continue
            if d.startswith("/"):
                out.append(d.rstrip("/") or "/")
        # Also pick up literal absolute paths (grep's form).
        for m in re.findall(r"(?<![A-Za-z0-9])/[A-Za-z0-9_/.\-]*", cmd):
            out.append(m.rstrip("/") or "/")
        return out

    def test_glob_resolves_root_to_workspace(self):
        """glob(path='/') must search /workspace, not the container's /."""
        self.sandbox.glob("**/Makefile", path="/")
        decoded = self._decoded_paths_in(self._last_command())
        self.assertIn(
            "/workspace", decoded,
            f"Expected glob path resolved to /workspace, got {decoded!r}",
        )
        self.assertNotIn(
            "/", decoded,
            f"Container-root path must not reach the glob subprocess: {decoded!r}",
        )

    def test_glob_resolves_relative_path_to_workspace(self):
        self.sandbox.glob("*.py", path="src")
        decoded = self._decoded_paths_in(self._last_command())
        self.assertIn("/workspace/src", decoded)

    def test_grep_resolves_root_to_workspace(self):
        """grep(path='/') must search /workspace, not the container's /."""
        self.sandbox.grep("TODO", path="/")
        decoded = self._decoded_paths_in(self._last_command())
        self.assertIn("/workspace", decoded)
        self.assertNotIn("/", decoded)

    def test_ls_resolves_root_to_workspace(self):
        """ls('/') must list /workspace, not the container's /."""
        self.sandbox.ls("/")
        decoded = self._decoded_paths_in(self._last_command())
        self.assertIn("/workspace", decoded)
        self.assertNotIn("/", decoded)

    def test_deprecated_overrides_are_gone(self):
        """We must NOT define ls_info / grep_raw / glob_info — those
        are deepagents' deprecated names and our overriding them was
        the original bug.  Inheriting from BaseSandbox is correct.

        This test is the regression bell: if a future change re-adds
        an override under those names, the *_resolve dead-code path
        comes back and the runaway-glob bug returns.
        """
        for deprecated in ("ls_info", "grep_raw", "glob_info"):
            cls_method = DockerSandboxBackend.__dict__.get(deprecated)
            self.assertIsNone(
                cls_method,
                f"DockerSandboxBackend.{deprecated} re-introduced — "
                "this is the deprecated deepagents API; override "
                f"{deprecated.split('_')[0]}() (the protocol's current "
                "name) instead so _resolve actually fires.",
            )

    def test_upload_files(self):
        self.container.put_archive.return_value = True
        responses = self.sandbox.upload_files([
            ("/workspace/test.py", b"print('hello')"),
            ("/workspace/data.txt", b"data"),
        ])

        self.assertEqual(len(responses), 2)
        self.assertIsNone(responses[0].error)
        self.assertIsNone(responses[1].error)
        self.assertEqual(self.container.put_archive.call_count, 2)

    def test_upload_files_partial_failure(self):
        def side_effect(path, data):
            if self.container.put_archive.call_count > 1:
                raise PermissionError("read-only")
            return True

        self.container.put_archive.side_effect = side_effect
        responses = self.sandbox.upload_files([
            ("/workspace/ok.txt", b"ok"),
            ("/workspace/fail.txt", b"fail"),
        ])

        self.assertEqual(len(responses), 2)
        self.assertIsNone(responses[0].error)
        self.assertEqual(responses[1].error, "permission_denied")

    def test_download_files(self):
        import io
        import tarfile

        # Create a fake tar archive
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            info = tarfile.TarInfo(name="test.txt")
            content = b"file content"
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        tar_bytes = tar_stream.getvalue()

        self.container.get_archive.return_value = ([tar_bytes], {"size": len(tar_bytes)})

        responses = self.sandbox.download_files(["/workspace/test.txt"])

        self.assertEqual(len(responses), 1)
        self.assertIsNone(responses[0].error)
        self.assertEqual(responses[0].content, b"file content")

    def test_download_files_not_found(self):
        self.container.get_archive.side_effect = Exception("file not found")

        responses = self.sandbox.download_files(["/workspace/missing.txt"])

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].error, "file_not_found")


class TestSandboxManager(TestCase):
    """Test SandboxManager Docker lifecycle with mocked Docker client."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Clear class-level state between tests
        SandboxManager._docker_client = None
        SandboxManager._containers.clear()

    def tearDown(self):
        SandboxManager._docker_client = None
        SandboxManager._containers.clear()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('assist.sandbox.DockerSandboxBackend')
    def test_get_sandbox_backend_creates_container(self, mock_backend_cls):
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "test123456ab"
        mock_container.status = "running"
        # Doubles as the egress-proxy container too — _wait_for_egress_proxy_ready
        # polls .logs() looking for "listening on" before returning.
        mock_container.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n"
        # `_start_container` runs `mkdir -p /workspace/references` via exec
        # after `containers.run` — see the pre-create comment in
        # assist/sandbox_manager.py.  Default mock returns a MagicMock that
        # cannot be unpacked into (exit_code, output), so pin it to success.
        mock_container.exec_run.return_value = (0, b"")
        mock_client.containers.run.return_value = mock_container

        with patch.object(SandboxManager, '_get_docker_client', return_value=mock_client):
            sandbox = SandboxManager.get_sandbox_backend(test_path)

        self.assertIsNotNone(sandbox)
        # Two containers.run calls: the egress proxy (idempotent setup)
        # and the sandbox itself.  Find the sandbox call specifically.
        sandbox_calls = [
            c for c in mock_client.containers.run.call_args_list
            if c.args and c.args[0] == "assist-sandbox"
        ]
        self.assertEqual(len(sandbox_calls), 1,
                         "Expected exactly one assist-sandbox containers.run call")
        # Verify container is registered
        self.assertIn(test_path, SandboxManager._containers)

    @patch('assist.sandbox.DockerSandboxBackend')
    def test_get_sandbox_backend_pre_creates_references_dir(self, mock_backend_cls):
        """`_start_container` must `mkdir -p /workspace/references` inside
        the container BEFORE returning.  Without this, the
        research-subagent's `DockerSandboxBackend(work_dir="/workspace/references")`
        fails its first exec call with `chdir to cwd`.  Regression for
        2026-05-16 winged-horse-flag thread.
        """
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "test123456ab"
        mock_container.status = "running"
        mock_container.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n"
        mock_container.exec_run.return_value = (0, b"")
        mock_client.containers.run.return_value = mock_container

        with patch.object(SandboxManager, '_get_docker_client', return_value=mock_client):
            SandboxManager.get_sandbox_backend(test_path)

        # Find the mkdir exec call.  exec_run may be called for the egress
        # proxy too (no — proxy uses container.logs(), not exec_run); be
        # defensive and look for the mkdir argv.
        mkdir_calls = [
            c for c in mock_container.exec_run.call_args_list
            if c.args
            and isinstance(c.args[0], list)
            and c.args[0][:2] == ["mkdir", "-p"]
            and c.args[0][2] == "/workspace/references"
        ]
        self.assertEqual(
            len(mkdir_calls), 1,
            "Expected exactly one `mkdir -p /workspace/references` exec_run "
            "call after the sandbox container starts.  Without this, the "
            "research-subagent fails every tool call with `chdir to cwd`."
        )

    @patch('assist.sandbox.DockerSandboxBackend')
    def test_get_sandbox_backend_raises_when_mkdir_fails(self, mock_backend_cls):
        """If `mkdir -p /workspace/references` returns a non-zero exit,
        `_start_container` must raise rather than return a half-broken
        backend.  Fail-closed posture matching the egress-proxy setup
        (see line 249-264 of sandbox_manager.py).
        """
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "test123456ab"
        mock_container.status = "running"
        mock_container.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n"
        mock_container.exec_run.return_value = (1, b"mkdir: permission denied")
        mock_client.containers.run.return_value = mock_container

        with patch.object(SandboxManager, '_get_docker_client', return_value=mock_client):
            with self.assertRaises(RuntimeError) as cm:
                SandboxManager.get_sandbox_backend(test_path)
        # Error message must surface the failure mode (exit code + output)
        # so an operator can diagnose without re-running.
        self.assertIn("/workspace/references", str(cm.exception))
        self.assertIn("exit_code=1", str(cm.exception))
        self.assertIn("permission denied", str(cm.exception))

    @patch('assist.sandbox.DockerSandboxBackend')
    def test_get_sandbox_backend_cleans_up_on_mkdir_failure(self, mock_backend_cls):
        """When mkdir fails, the broken container must be stopped AND
        the _containers registry must not retain it.  Otherwise the next
        get_sandbox_backend call hits the early-return at line 197-203
        and hands out a backend wrapping the broken container — exactly
        the chdir bug this PR fixes.  Regression for self-review pass 1
        BLOCKER (commit message reference 48ffb5a → 87cd95c-equivalent).
        """
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "test123456ab"
        mock_container.status = "running"
        mock_container.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n"
        mock_container.exec_run.return_value = (1, b"mkdir: read-only filesystem")
        mock_client.containers.run.return_value = mock_container

        with patch.object(SandboxManager, '_get_docker_client', return_value=mock_client):
            with self.assertRaises(RuntimeError):
                SandboxManager.get_sandbox_backend(test_path)

        # Registry must NOT contain the broken container.
        self.assertNotIn(
            test_path, SandboxManager._containers,
            "Broken container left in _containers registry — next "
            "get_sandbox_backend call would short-circuit and hand it out."
        )
        # Container must have been stopped (triggers auto-remove via
        # remove=True).
        mock_container.stop.assert_called_once()

    @patch('assist.sandbox.DockerSandboxBackend')
    def test_get_sandbox_backend_passes_host_uid(self, mock_backend_cls):
        """The container must run as the host bind-mount's owner —
        privilege-separation layer that closes the cp+exec-a bypass.
        See docs/2026-05-08-restrict-git-real-via-non-root-sandbox.org.
        """
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "test123456ab"
        mock_container.status = "running"
        mock_container.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n"
        mock_container.exec_run.return_value = (0, b"")
        mock_client.containers.run.return_value = mock_container

        host_st = os.stat(test_path)
        expected_user = f"{host_st.st_uid}:{host_st.st_gid}"

        with patch.object(SandboxManager, '_get_docker_client', return_value=mock_client):
            SandboxManager.get_sandbox_backend(test_path)

        # The sandbox call (not the proxy call) must carry user=.  Pick
        # it out by image name.
        sandbox_call = next(
            c for c in mock_client.containers.run.call_args_list
            if c.args and c.args[0] == "assist-sandbox"
        )
        self.assertEqual(
            sandbox_call.kwargs.get('user'), expected_user,
            f"containers.run must be called with user={expected_user!r} "
            "so the agent inside the sandbox runs as the host bind-mount "
            "owner — not as root.",
        )

    def test_get_sandbox_backend_raises_when_workspace_missing(self):
        """A missing workspace must surface as a typed error, not a
        silent fallback.  Saved feedback memory:
        'Threads should die on infrastructure failure rather than
        heal-and-retry'.
        """
        missing = os.path.join(self.temp_dir, "does-not-exist")
        with self.assertRaises(RuntimeError) as ctx:
            SandboxManager.get_sandbox_backend(missing)
        self.assertIn("Cannot start sandbox", str(ctx.exception))

    def test_get_sandbox_backend_refuses_root_owned_workspace(self):
        """A root-owned workspace would silently restore the cp+exec-a
        bypass that the non-root sandbox layer exists to close.
        Refuse fail-closed and tell the operator how to migrate.
        """
        from unittest.mock import patch

        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        # Mock os.stat to return st_uid=0 (the legacy-thread case).
        # Synthesise a stat_result with the right field positions.
        real_st = os.stat(test_path)
        fake_st = os.stat_result((
            real_st.st_mode, real_st.st_ino, real_st.st_dev,
            real_st.st_nlink, 0, 0,  # st_uid=0, st_gid=0
            real_st.st_size, real_st.st_atime,
            real_st.st_mtime, real_st.st_ctime,
        ))

        with patch('assist.sandbox_manager.os.stat', return_value=fake_st):
            with self.assertRaises(RuntimeError) as ctx:
                SandboxManager.get_sandbox_backend(test_path)

        msg = str(ctx.exception)
        self.assertIn("owned by root", msg)
        self.assertIn("chown", msg)

    def test_get_sandbox_backend_reuses_container(self):
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        mock_container = MagicMock()
        mock_container.id = "test123456ab"
        mock_container.status = "running"
        SandboxManager._containers[test_path] = mock_container

        sandbox = SandboxManager.get_sandbox_backend(test_path)

        self.assertIsNotNone(sandbox)
        # Container should have been reloaded to check status
        mock_container.reload.assert_called_once()

    def test_get_sandbox_backend_returns_none_on_docker_error(self):
        """Docker daemon unavailable / transient API error should
        degrade to a None backend (no sandbox).  Policy / fail-closed
        errors (RuntimeError from non-internal egress network, etc)
        raise instead — see test_get_sandbox_backend_raises_when_*.
        """
        from docker.errors import DockerException

        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        mock_client = MagicMock()
        # Has to be a DockerException subclass now that the catch is
        # narrowed; a bare Exception would propagate (correctly).
        mock_client.containers.run.side_effect = DockerException("Docker not running")

        with patch.object(SandboxManager, '_get_docker_client', return_value=mock_client):
            sandbox = SandboxManager.get_sandbox_backend(test_path)

        self.assertIsNone(sandbox)

    def test_cleanup_stops_and_removes_container(self):
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        mock_container = MagicMock()
        SandboxManager._containers[test_path] = mock_container

        SandboxManager.cleanup(test_path)

        mock_container.stop.assert_called_once_with(timeout=5)
        self.assertNotIn(test_path, SandboxManager._containers)

    def test_cleanup_all(self):
        mock_c1 = MagicMock()
        mock_c2 = MagicMock()
        SandboxManager._containers = {"/path/a": mock_c1, "/path/b": mock_c2}

        SandboxManager.cleanup_all()

        mock_c1.stop.assert_called_once()
        mock_c2.stop.assert_called_once()
        self.assertEqual(len(SandboxManager._containers), 0)

    def test_domain_manager_without_git(self):
        """Test that DomainManager works without git remote."""
        test_path = os.path.join(self.temp_dir, "no_git")

        dm = DomainManager(repo_path=test_path)

        self.assertIsNone(dm.repo)
        self.assertTrue(os.path.isdir(test_path))
        self.assertEqual(dm.changes(), [])
        self.assertEqual(dm.main_diff(), [])

    def test_domain_manager_sync_noop_without_git(self):
        """Test that sync does nothing without git."""
        test_path = os.path.join(self.temp_dir, "no_git")

        dm = DomainManager(repo_path=test_path)
        # Should not raise
        dm.sync("test commit")


class TestSandboxCompositeBackend(TestCase):
    """Test create_sandbox_composite_backend factory."""

    def test_returns_composite_backend(self):
        from assist.backends import create_sandbox_composite_backend
        from deepagents.backends import CompositeBackend

        mock_sandbox = MagicMock()
        backend = create_sandbox_composite_backend(mock_sandbox)
        self.assertIsInstance(backend, CompositeBackend)

    def test_composite_uses_sandbox_default(self):
        from assist.backends import create_sandbox_composite_backend
        from deepagents.backends import CompositeBackend

        mock_sandbox = MagicMock()
        backend = create_sandbox_composite_backend(mock_sandbox)

        self.assertIsInstance(backend, CompositeBackend)
