"""Unit tests for Docker sandbox backend and DomainManager Docker lifecycle.

All Docker interactions are mocked — no real Docker daemon needed.
"""
import hashlib
import os
import tempfile
import shutil
from unittest import TestCase, skipIf
from unittest.mock import patch, MagicMock, PropertyMock

from assist.sandbox import (
    DockerSandboxBackend,
    MAX_OUTPUT_CHARS,
    SandboxContainerLostError,
)
from assist.domain_manager import DomainManager
from assist.sandbox_manager import SandboxManager
from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.sandbox import BaseSandbox


def test_stop_proof_waits_for_auto_removed_name_release():
    from assist.sandbox_manager import confirm_generation_stopped

    responses = [
        MagicMock(returncode=0, stdout=b"", stderr=b""),
        MagicMock(returncode=0, stdout=b"false\n", stderr=b""),
        MagicMock(returncode=0, stdout=b"", stderr=b""),
        MagicMock(returncode=1, stdout=b"", stderr=b"No such container"),
    ]
    with patch("assist.sandbox_manager.subprocess.run", side_effect=responses) as run:
        confirm_generation_stopped("exact-generation")
    assert [call.args[0][:2] for call in run.call_args_list] == [
        ["docker", "kill"], ["docker", "inspect"],
        ["docker", "rm"], ["docker", "inspect"]]


class TestDockerSandboxBackend(TestCase):
    """Test DockerSandboxBackend with mocked Docker container."""

    def setUp(self):
        self.container = MagicMock()
        self.container.id = "abc123def456"
        self.sandbox = DockerSandboxBackend(self.container)

    @staticmethod
    def _throttle_event(token="a" * 32, retry_after=8):
        execution_token = "assist-exec-" + token
        correlation = hashlib.sha256(execution_token.encode()).hexdigest()[:16]
        return (
            "egress-proxy: THROTTLE client=172.30.0.2 host=example.com "
            f"retry after {retry_after}s execution={correlation}\n").encode()

    def test_id_returns_short_container_id(self):
        self.assertEqual(self.sandbox.id, "abc123def456")

    def test_resolve_prefixes_path(self):
        self.assertEqual(self.sandbox._resolve("/myfile.txt"), "/workspace/myfile.txt")

    def test_resolve_preserves_workspace_path(self):
        self.assertEqual(self.sandbox._resolve("/workspace/myfile.txt"), "/workspace/myfile.txt")

    def test_resolve_maps_user_alias_to_workspace(self):
        self.assertEqual(self.sandbox._resolve("/user/myfile.txt"), "/workspace/myfile.txt")
        self.assertEqual(self.sandbox._resolve("/user"), "/workspace")
        self.assertEqual(self.sandbox._resolve("/userfoo"), "/workspace/userfoo")

    def test_resolve_handles_relative_path(self):
        self.assertEqual(self.sandbox._resolve("myfile.txt"), "/workspace/myfile.txt")

    def test_resolve_handles_none(self):
        self.assertIsNone(self.sandbox._resolve(None))

    def test_resolve_passes_tmp_through_natively(self):
        # /tmp is a real bind-mount the shell and the file tools share — it must NOT
        # be shadowed under /workspace, so an offloaded /tmp file has one path both
        # resolve to (docs/2026-07-22-offload-to-real-fs.org).
        self.assertEqual(self.sandbox._resolve("/tmp"), "/tmp")
        self.assertEqual(
            self.sandbox._resolve("/tmp/large_tool_results/untrusted-x"),
            "/tmp/large_tool_results/untrusted-x")

    def test_resolve_tmp_scoping_does_not_leak(self):
        # Only the exact /tmp component passes through; /tmpfoo (a workspace file whose
        # name merely starts "tmp") is still prefixed, so the scoping can't be abused.
        self.assertEqual(self.sandbox._resolve("/tmpfoo"), "/workspace/tmpfoo")

    def test_resolve_tmp_passthrough_keyed_on_pre_strip_path(self):
        # In the references-confined sandbox, strip_prefixes turns "/references/tmp/x"
        # into "/tmp/x" — the /tmp passthrough must key on the ORIGINAL path so that
        # doesn't escape references confinement into the shared /tmp (it stays under
        # work_dir); a GENUINE "/tmp/x" input still passes through.
        ref = DockerSandboxBackend(self.container, work_dir="/workspace/references",
                                   strip_prefixes=("references",))
        self.assertEqual(ref._resolve("/references/tmp/x"), "/workspace/references/tmp/x")
        self.assertEqual(ref._resolve("/tmp/x"), "/tmp/x")

    def test_agent_passthrough_requires_explicit_capability(self):
        self.assertEqual(self.sandbox._resolve("/agent/memory.md"),
                         "/workspace/agent/memory.md")
        mounted = DockerSandboxBackend(self.container, native_agent_dir=True)
        self.assertEqual(mounted._resolve("/agent"), "/agent")
        self.assertEqual(mounted._resolve("/agent/memory.md"), "/agent/memory.md")
        self.assertEqual(mounted._resolve("/agentfoo"), "/workspace/agentfoo")

    def test_write_recovers_an_existing_empty_file(self):
        """The sandbox may initialize an existing zero-byte memory source."""
        collision = WriteResult(error="Error: File already exists: '/workspace/AGENTS.md'")
        with patch.object(BaseSandbox, "write", return_value=collision) as write, \
             patch.object(BaseSandbox, "edit",
                          return_value=EditResult(path="/workspace/AGENTS.md",
                                                  occurrences=1)) as edit:
            result = self.sandbox.write("/AGENTS.md", "User has 3 cats.\n")

        self.assertIsNone(result.error)
        self.assertEqual(result.path, "/AGENTS.md")
        write.assert_called_once_with("/workspace/AGENTS.md", "User has 3 cats.\n")
        edit.assert_called_once_with("/workspace/AGENTS.md", "", "User has 3 cats.\n")

    def test_write_keeps_nonempty_file_no_clobber(self):
        """A failed empty-string edit preserves the original collision error."""
        collision = WriteResult(error="Error: File already exists: '/workspace/AGENTS.md'")
        with patch.object(BaseSandbox, "write", return_value=collision), \
             patch.object(BaseSandbox, "edit",
                          return_value=EditResult(error="multiple occurrences")):
            result = self.sandbox.write("/AGENTS.md", "User has 3 cats.\n")

        self.assertEqual(result, collision)

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
        self.container.client.containers.get.assert_not_called()

    def test_execute_nonzero_exit(self):
        self.container.exec_run.return_value = (1, b"error: not found\n")
        resp = self.sandbox.execute("cat /missing")

        self.assertEqual(resp.exit_code, 1)
        self.assertIn("not found", resp.output)

    def test_execute_proxy_throttle_returns_truthful_guidance(self):
        """A bare client result still has trusted local throttle provenance."""
        self.container.attrs = {
            "NetworkSettings": {"Networks": {
                "assist-egress-network": {"IPAddress": "172.30.0.2"}}}}
        proxy = self.container.client.containers.get.return_value
        proxy.logs.return_value = self._throttle_event()
        self.container.exec_run.return_value = (0, b"429")

        with patch("assist.sandbox.secrets.token_hex", return_value="a" * 32):
            resp = self.sandbox.execute("curl https://example.com")

        self.assertIn("throttled locally by Assist", resp.output)
        self.assertIn("a request from this sandbox before it contacted the remote host", resp.output)
        self.assertIn("neither an external outage nor an egress-approval denial", resp.output)
        self.assertIn("at least 8 seconds", resp.output)

    def test_execute_proxy_throttle_preserves_retry_after_header(self):
        self.container.attrs = {
            "NetworkSettings": {"Networks": {
                "assist-egress-network": {"IPAddress": "172.30.0.2"}}}}
        proxy = self.container.client.containers.get.return_value
        proxy.logs.return_value = self._throttle_event()
        self.container.exec_run.return_value = (1, b"Proxy error: 429 Too Many Requests\n")

        with patch("assist.sandbox.secrets.token_hex", return_value="a" * 32):
            resp = self.sandbox.execute("curl http://example.com")

        self.assertIn("at least 8 seconds", resp.output)

    def test_execute_proxy_throttle_clamps_untrusted_retry_after_text(self):
        self.container.attrs = {
            "NetworkSettings": {"Networks": {
                "assist-egress-network": {"IPAddress": "172.30.0.2"}}}}
        proxy = self.container.client.containers.get.return_value
        proxy.logs.return_value = self._throttle_event(retry_after=999999)
        self.container.exec_run.return_value = (1, b"Proxy error: 429 Too Many Requests\n")

        with patch("assist.sandbox.secrets.token_hex", return_value="a" * 32):
            resp = self.sandbox.execute("curl http://example.com")

        self.assertIn("at least 60 seconds", resp.output)

    def test_execute_proxy_throttle_survives_a_busy_proxy_log(self):
        self.container.attrs = {
            "NetworkSettings": {"Networks": {
                "assist-egress-network": {"IPAddress": "172.30.0.2"}}}}
        proxy = self.container.client.containers.get.return_value
        proxy.logs.return_value = (
            b"unrelated event\n" * 101
            + self._throttle_event())
        self.container.exec_run.return_value = (1, b"Proxy error: 429 Too Many Requests\n")

        with patch("assist.sandbox.time.time", return_value=123.5), \
             patch("assist.sandbox.secrets.token_hex", return_value="a" * 32):
            resp = self.sandbox.execute("curl http://example.com")

        self.assertIn("throttled locally by Assist", resp.output)
        proxy.logs.assert_called_once_with(since=123.5)

    def test_execute_ignores_a_throttle_for_another_execution(self):
        self.container.attrs = {
            "NetworkSettings": {"Networks": {
                "assist-egress-network": {"IPAddress": "172.30.0.2"}}}}
        proxy = self.container.client.containers.get.return_value
        proxy.logs.return_value = self._throttle_event(token="b" * 32)
        self.container.exec_run.return_value = (1, b"Proxy error: 429 Too Many Requests\n")

        with patch("assist.sandbox.secrets.token_hex", return_value="a" * 32):
            resp = self.sandbox.execute("echo unrelated")

        self.assertNotIn("throttled locally by Assist", resp.output)

    def test_execute_origin_429_is_not_mislabelled_as_local_throttle(self):
        self.container.exec_run.return_value = (
            1, b"HTTP/1.1 429 Too Many Requests\nServer: nginx proxy\nRetry-After: 30\n")

        resp = self.sandbox.execute("curl https://example.com")

        self.assertNotIn("throttled locally by Assist", resp.output)

    def test_execute_unrelated_429_text_does_not_get_network_guidance(self):
        self.container.exec_run.return_value = (0, b"the 429th record\n")

        resp = self.sandbox.execute("echo 'the 429th record'")

        self.assertNotIn("Network request received HTTP 429", resp.output)
        self.container.client.containers.get.assert_not_called()

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

    def test_grep_catches_value_error_from_exec_failure_parsing(self):
        """When the container exec fails (chdir error, missing binary,
        container died mid-exec), Docker returns the OCI runtime error
        message as stdout.  deepagents 0.6.1's SandboxBackend.grep
        (deepagents/backends/sandbox.py:776-783) splits each line on `:`
        and does `int(parts[1])` — for an exec-failure message like
        `OCI runtime exec failed: exec failed: unable to start...`,
        parts[1] is ` exec failed` and int() raises ValueError.

        Without the wrap in assist/sandbox.py, that ValueError
        propagated to the agent loop and crashed the thread (regression:
        thread 20260516200432-50b5360f, 2026-05-16).

        With the wrap, grep returns a clean GrepResult(error=...,
        matches=None) — the model sees `grep failed` and can adapt
        instead of crashing the whole tool node.
        """
        # Make container.exec_run return exec-failure text.  deepagents'
        # SandboxBackend.grep will then try to parse it as `path:line:text`
        # and crash with ValueError.
        self.container.exec_run.return_value = (
            0,  # docker exec returns 0 for the wrapping shell even on inner failure
            b'OCI runtime exec failed: exec failed: unable to start '
            b'container process: chdir to cwd ("/workspace/references"): '
            b'no such file or directory: unknown\n',
        )
        result = self.sandbox.grep("TODO", path="/")
        # Must NOT raise.
        self.assertIsNotNone(result)
        # Surfaces as a clean error on the GrepResult (deepagents' own
        # error-channel) rather than a crash.
        self.assertIsNotNone(result.error)
        self.assertIn("grep failed", result.error)
        self.assertIsNone(result.matches)

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

    @staticmethod
    def _uploaded_tarinfo(put_archive_mock):
        """Pull the single TarInfo out of a captured put_archive(stream) call."""
        import io
        import tarfile

        stream = put_archive_mock.call_args[0][1]
        stream.seek(0)
        with tarfile.open(fileobj=io.BytesIO(stream.read()), mode="r") as tar:
            members = tar.getmembers()
        assert len(members) == 1, members
        return members[0]

    def test_upload_files_stamps_run_uid_gid(self):
        # Regression: a bare TarInfo (uid/gid default to 0) makes put_archive
        # — which extracts as root — leave write_file output root-owned, which
        # the non-root sandbox then can't edit_file (PermissionError).  The
        # uploaded file must carry the container's run uid/gid instead.
        self.container.attrs = {"Config": {"User": "1007:1009"}}
        self.container.put_archive.return_value = True

        responses = self.sandbox.upload_files([("/workspace/foo.md", b"hello")])

        self.assertIsNone(responses[0].error)
        member = self._uploaded_tarinfo(self.container.put_archive)
        self.assertEqual(member.uid, 1007)
        self.assertEqual(member.gid, 1009)
        # Names stay empty so the numeric ids are authoritative on extract.
        self.assertEqual(member.uname, "")
        self.assertEqual(member.gname, "")

    def test_upload_files_uid_only_user_reuses_for_gid(self):
        self.container.attrs = {"Config": {"User": "1007"}}
        self.container.put_archive.return_value = True

        self.sandbox.upload_files([("/workspace/foo.md", b"hi")])

        member = self._uploaded_tarinfo(self.container.put_archive)
        self.assertEqual((member.uid, member.gid), (1007, 1007))

    @skipIf(os.getuid() == 0, "fallback asserts a non-root owner; meaningless as root")
    def test_upload_files_falls_back_to_process_owner_when_user_unparseable(self):
        # A non-numeric / missing Config.User (e.g. a mocked container) must
        # not silently revert to root-owned uploads.
        self.container.attrs = {"Config": {"User": ""}}
        self.container.put_archive.return_value = True

        self.sandbox.upload_files([("/workspace/foo.md", b"hi")])

        member = self._uploaded_tarinfo(self.container.put_archive)
        self.assertEqual(member.uid, os.getuid())
        self.assertEqual(member.gid, os.getgid())
        self.assertNotEqual(member.uid, 0)


class TestSandboxManager(TestCase):
    """Test SandboxManager Docker lifecycle with mocked Docker client."""

    @staticmethod
    def _attached_proxy(container, client):
        container.status = "running"
        container.attrs = {"Mounts": [], "NetworkSettings": {"Networks": {
            "assist-egress-network": {"NetworkID": "ordinary-net"}}}}
        client.api._version = "1.43"
        client.api.create_container.return_value = {"Id": container.id}
        client.containers.get.return_value = container

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Clear class-level state between tests
        SandboxManager._docker_client = None
        SandboxManager._containers.clear()
        SandboxManager._generation_owners.clear()
        # These lifecycle mocks have no Docker network attributes; live
        # routing validation is covered by the real browser Docker tests.
        self.network_identity = patch(
            'assist.sandbox_manager._network_identity',
            side_effect=lambda _network, *, isolated:
                ('browser-net', '172.31.0.0/16') if isolated
                else ('ordinary-net', '172.30.0.0/16'))
        self.network_identity.start()

    def tearDown(self):
        self.network_identity.stop()
        SandboxManager._docker_client = None
        SandboxManager._containers.clear()
        SandboxManager._generation_owners.clear()
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
        mock_container.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n" + \
            b"egress-proxy: listening on 0.0.0.0:8889\n"
        self._attached_proxy(mock_container, mock_client)
        mock_client.containers.run.return_value = mock_container

        with patch.object(SandboxManager, '_get_docker_client', return_value=mock_client):
            sandbox = SandboxManager.get_sandbox_backend(test_path)

        self.assertIsNotNone(sandbox)
        # The proxy uses create/start; the shell uses containers.run.
        sandbox_calls = [
            c for c in mock_client.containers.run.call_args_list
            if c.args and c.args[0] == "assist-sandbox"
        ]
        self.assertEqual(len(sandbox_calls), 1,
                         "Expected exactly one assist-sandbox containers.run call")
        # Verify container is registered
        self.assertIn(test_path, SandboxManager._containers)

    @patch('assist.sandbox.DockerSandboxBackend')
    def test_browser_capable_turn_gets_private_tmpfs_and_seccomp(self, _backend):
        from assist.browser.authority import mark_new_thread
        from assist.browser.manager import BrowserManager
        test_path = os.path.join(self.temp_dir, "browser-turn", "domain")
        os.makedirs(test_path)
        mark_new_thread(self.temp_dir, "browser-turn")
        client = MagicMock()
        container = MagicMock()
        container.id = "browser-turn-generation"
        container.status = "running"
        container.logs.return_value = (
            b"egress-proxy: listening on 0.0.0.0:8888\n"
            b"egress-proxy: listening on 0.0.0.0:8889\n")
        self._attached_proxy(container, client)
        client.containers.run.return_value = container
        with patch.object(SandboxManager, '_get_docker_client', return_value=client), \
             patch.object(BrowserManager, 'reconcile_startup', return_value=True), \
             patch.object(BrowserManager, 'confirm_owner_stopped', return_value=False):
            SandboxManager.get_sandbox_backend(
                test_path, browser_capable=True,
                thread_scope=(self.temp_dir, "browser-turn"))
        options = client.containers.run.call_args.kwargs
        assert options["tmpfs"]["/run/assist-browser"].endswith(
            "uid=10001,gid=10001,mode=0700")
        assert options["security_opt"][0].startswith("seccomp={")

    @patch('assist.sandbox.DockerSandboxBackend')
    def test_managed_shell_generation_is_journaled_and_cleared(self, _backend):
        from assist.browser import authority
        from assist.browser.manager import BrowserManager

        test_path = os.path.join(self.temp_dir, "managed-turn", "domain")
        os.makedirs(test_path)
        authority.mark_new_thread(self.temp_dir, "managed-turn")
        client = MagicMock()
        container = MagicMock()
        container.id = "managed-generation"
        container.status = "running"
        container.logs.return_value = (
            b"egress-proxy: listening on 0.0.0.0:8888\n"
            b"egress-proxy: listening on 0.0.0.0:8889\n")
        self._attached_proxy(container, client)
        client.containers.run.return_value = container
        with patch.object(SandboxManager, '_get_docker_client', return_value=client), \
             patch.object(BrowserManager, 'reconcile_startup', return_value=True), \
             patch.object(BrowserManager, 'confirm_owner_stopped', return_value=False):
            SandboxManager.get_sandbox_backend(
                test_path, thread_scope=(self.temp_dir, "managed-turn"),
                owner_run_id="claimed-run")
        with authority.fence(self.temp_dir, "managed-turn") as state:
            self.assertEqual(state.lease["owner_run_id"], "claimed-run")
            self.assertEqual(state.lease["generations"], [container.id])
        with patch('assist.sandbox_manager.confirm_generation_stopped') as stopped:
            SandboxManager.cleanup(test_path, container)
        stopped.assert_called_once_with(container.id)
        with authority.fence(self.temp_dir, "managed-turn") as state:
            self.assertIsNone(state.lease)

    def test_generic_workspaces_do_not_access_browser_authority(self):
        from docker.errors import DockerException
        from assist.browser import authority

        work_dirs = [self.temp_dir, os.path.join(self.temp_dir, "scratch"),
                     os.path.join(self.temp_dir, "other"),
                     os.path.join(self.temp_dir, "domain")]
        for work_dir in work_dirs:
            os.makedirs(work_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sandbox-scope-ledger-") as ledger:
            with patch.dict(os.environ, {"ASSIST_EGRESS_RUNTIME_DIR": ledger}), \
                 patch.object(authority, "fence", side_effect=AssertionError(
                     "generic workspace must not access thread authority")) as fence, \
                 patch.object(SandboxManager, "_get_docker_client",
                              side_effect=DockerException("no Docker in regression")):
                for work_dir in work_dirs:
                    self.assertIsNone(SandboxManager.get_sandbox_backend(work_dir))
        fence.assert_not_called()

    def test_browser_sandbox_requires_explicit_scope_before_docker(self):
        with patch.object(SandboxManager, "_get_docker_client") as docker_client:
            with self.assertRaisesRegex(RuntimeError, "explicit thread authority"):
                SandboxManager.get_sandbox_backend(self.temp_dir, browser_capable=True)
        docker_client.assert_not_called()

    def test_cross_thread_workspace_alias_is_rejected_before_authority(self):
        from assist.browser import authority

        first = os.path.join(self.temp_dir, "first")
        second = os.path.join(self.temp_dir, "second", "domain")
        os.makedirs(first)
        os.makedirs(second)
        alias = os.path.join(first, "domain")
        os.symlink(second, alias)
        with patch.object(authority, "fence") as fence, \
             patch.object(SandboxManager, "_get_docker_client") as docker_client:
            for browser_capable in (False, True):
                with self.assertRaisesRegex(RuntimeError, "does not match thread authority"):
                    SandboxManager.get_sandbox_backend(
                        alias, browser_capable=browser_capable,
                        thread_scope=(self.temp_dir, "first"))
            with self.assertRaisesRegex(RuntimeError, "does not match thread authority"):
                SandboxManager.get_pi_sandbox_backend(
                    alias, thread_scope=(self.temp_dir, "first"))
        fence.assert_not_called()
        docker_client.assert_not_called()

    def test_browser_lease_requires_stop_proof_before_any_replacement(self):
        from assist.browser import authority
        from assist.browser.manager import BrowserManager, BrowserUnavailable

        test_path = os.path.join(self.temp_dir, "browser-turn", "domain")
        os.makedirs(test_path)
        authority.mark_new_thread(self.temp_dir, "browser-turn")
        with authority.fence(self.temp_dir, "browser-turn") as state:
            state.begin("prior-run", 1)
            state.add_generation("prior-run", "prior-generation")
        stale = MagicMock(id="prior-generation")
        stale.kill.side_effect = RuntimeError("Docker kill uncertain")
        SandboxManager._containers[test_path] = stale

        with patch.object(BrowserManager, "reconcile_startup", return_value=True), \
             patch.object(BrowserManager, "confirm_owner_stopped",
                          side_effect=BrowserUnavailable("stop unconfirmed")) as proof:
            with self.assertRaisesRegex(BrowserUnavailable, "stop unconfirmed"):
                SandboxManager.get_sandbox_backend(
                    test_path, browser_capable=False,
                    thread_scope=(self.temp_dir, "browser-turn"))
            with self.assertRaisesRegex(BrowserUnavailable, "stop unconfirmed"):
                SandboxManager.get_pi_sandbox_backend(
                    test_path, thread_scope=(self.temp_dir, "browser-turn"))
        stale.kill.assert_not_called()
        self.assertIs(SandboxManager._containers[test_path], stale)
        self.assertEqual(proof.call_count, 2)
        self.assertEqual(proof.call_args_list[0], proof.call_args_list[1])
        proof.assert_called_with(
            self.temp_dir, "browser-turn", None, before_replacement=True)

    def test_covered_prejournal_thread_still_requires_scoped_stop_proof(self):
        from assist.browser import authority
        from assist.browser.manager import BrowserManager, BrowserUnavailable

        test_path = os.path.join(self.temp_dir, "covered-thread", "domain")
        os.makedirs(test_path)
        authority.mark_new_thread(self.temp_dir, "covered-thread")
        with patch.object(BrowserManager, "reconcile_startup", return_value=True), \
             patch.object(BrowserManager, "confirm_owner_stopped",
                          side_effect=BrowserUnavailable("old generation unconfirmed")) as proof:
            with self.assertRaisesRegex(BrowserUnavailable, "old generation unconfirmed"):
                SandboxManager.get_sandbox_backend(
                    test_path, thread_scope=(self.temp_dir, "covered-thread"))
        proof.assert_called_once_with(
            self.temp_dir, "covered-thread", None, before_replacement=True)

    @patch('assist.sandbox.DockerSandboxBackend')
    def test_agent_dir_is_created_mounted_and_enables_native_paths(self,
                                                                  mock_backend_cls):
        test_path = os.path.join(self.temp_dir, "thread", "domain")
        agent_dir = os.path.join(self.temp_dir, "thread", "agent")
        os.makedirs(test_path)
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "test123456ab"
        mock_container.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n" + \
            b"egress-proxy: listening on 0.0.0.0:8889\n"
        self._attached_proxy(mock_container, mock_client)
        mock_client.containers.run.return_value = mock_container

        with patch.object(SandboxManager, '_get_docker_client', return_value=mock_client):
            SandboxManager.get_sandbox_backend(test_path, agent_dir=agent_dir)

        sandbox_call = next(
            c for c in mock_client.containers.run.call_args_list
            if c.args and c.args[0] == "assist-sandbox")
        self.assertEqual(sandbox_call.kwargs["volumes"][agent_dir],
                         {"bind": "/agent", "mode": "rw"})
        self.assertTrue(os.path.isdir(agent_dir))
        mock_backend_cls.assert_called_once_with(
            mock_container, native_agent_dir=True)

    @patch('assist.sandbox.DockerSandboxBackend')
    def test_get_sandbox_backend_does_not_forward_host_only_configuration(self, mock_backend_cls):
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)
        mock_client = MagicMock()
        mock_container = MagicMock(id="test123456ab", status="running")
        mock_container.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n" + \
            b"egress-proxy: listening on 0.0.0.0:8889\n"
        self._attached_proxy(mock_container, mock_client)
        mock_client.containers.run.return_value = mock_container

        with patch.dict(os.environ, {
                "ASSIST_MODEL_URL": "http://model.example.test",
                "EMAIL_RESEND_API_KEY_FILE": "/private/key",
                "EMAIL_FROM_ADDRESS": "assistant@example.test",
                "EMAIL_FROM_NAME": "Assistant",
                "EMAIL_ALWAYS_CC": "oversight@example.test",
                "URGENT_SMS_RECIPIENT": "+15555550100",
                "URGENT_SMS_THREAD_URL_BASE": "https://web.example.test:5050",
        }):
            with patch.object(SandboxManager, '_get_docker_client', return_value=mock_client):
                SandboxManager.get_sandbox_backend(test_path)

        sandbox_call = next(c for c in mock_client.containers.run.call_args_list
                            if c.args and c.args[0] == "assist-sandbox")
        environment = sandbox_call.kwargs["environment"]
        self.assertEqual(environment["ASSIST_MODEL_URL"], "http://model.example.test")
        self.assertFalse(any(key.startswith(("EMAIL_", "URGENT_SMS_"))
                             for key in environment))

    @patch('assist.sandbox.DockerSandboxBackend')
    def test_pi_sandbox_has_no_assist_environment_or_agent_mount(self, mock_backend_cls):
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)
        mock_client = MagicMock()
        mock_container = MagicMock(id="test123456ab", status="running")
        mock_container.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n" + \
            b"egress-proxy: listening on 0.0.0.0:8889\n"
        self._attached_proxy(mock_container, mock_client)
        mock_client.containers.run.return_value = mock_container

        with patch.dict(os.environ, {"ASSIST_MODEL_URL": "http://model.example.test"}):
            with patch.object(SandboxManager, '_get_docker_client', return_value=mock_client):
                with patch.object(SandboxManager, '_record_egress_client') as enroll:
                    SandboxManager.get_pi_sandbox_backend(test_path)
        enroll.assert_called_once_with(
            mock_client, mock_container, test_path, kind="pi")

        sandbox_call = next(c for c in mock_client.containers.run.call_args_list
                            if c.args and c.args[0] == "assist-sandbox")
        self.assertFalse(any(key.startswith("ASSIST_")
                             for key in sandbox_call.kwargs["environment"]))
        self.assertNotIn("/agent", {
            mount["bind"] for mount in sandbox_call.kwargs["volumes"].values()})
        mock_backend_cls.assert_called_once_with(mock_container, native_agent_dir=False)

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
        mock_container.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n" + \
            b"egress-proxy: listening on 0.0.0.0:8889\n"
        self._attached_proxy(mock_container, mock_client)
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

    @patch('assist.sandbox.DockerSandboxBackend')
    def test_get_sandbox_backend_does_not_reuse_reaps_stale(self, mock_backend_cls):
        """Per-turn lifecycle: a container left registered from a prior turn is
        NOT reused — it is killed (SIGKILL, via cleanup) and a fresh one is
        created.  Reusing it would let a container outlive its turn, which is
        the whole thing the per-turn design forbids."""
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        stale = MagicMock()
        stale.id = "stale1234567"
        stale.status = "running"
        SandboxManager._containers[test_path] = stale

        mock_client = MagicMock()
        fresh = MagicMock()
        fresh.id = "fresh1234567"
        fresh.status = "running"
        fresh.logs.return_value = b"egress-proxy: listening on 0.0.0.0:8888\n" + \
            b"egress-proxy: listening on 0.0.0.0:8889\n"
        self._attached_proxy(fresh, mock_client)
        mock_client.containers.run.return_value = fresh

        with patch.object(SandboxManager, '_get_docker_client', return_value=mock_client), \
             patch('assist.sandbox_manager.confirm_generation_stopped') as stopped:
            sandbox = SandboxManager.get_sandbox_backend(test_path)

        self.assertIsNotNone(sandbox)
        # The stale container was SIGKILLed, not reused (never reload()'d).
        stopped.assert_called_once_with(stale.id)
        stale.reload.assert_not_called()
        # A fresh container replaced it in the registry.
        self.assertIs(SandboxManager._containers[test_path], fresh)

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
        mock_client.api.create_container.side_effect = DockerException("Docker not running")

        with patch.object(SandboxManager, '_get_docker_client', return_value=mock_client):
            sandbox = SandboxManager.get_sandbox_backend(test_path)

        self.assertIsNone(sandbox)

    def test_cleanup_kills_and_removes_container(self):
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        mock_container = MagicMock()
        SandboxManager._containers[test_path] = mock_container

        with patch('assist.sandbox_manager.confirm_generation_stopped') as stopped:
            SandboxManager.cleanup(test_path)

        # SIGKILL, not a graceful stop: a sandbox has nothing to flush and its
        # bare-`sleep` PID 1 ignores SIGTERM, so stop() only burns the timeout.
        stopped.assert_called_once_with(mock_container.id)
        mock_container.stop.assert_not_called()
        self.assertNotIn(test_path, SandboxManager._containers)

    def test_cleanup_all(self):
        mock_c1 = MagicMock()
        mock_c2 = MagicMock()
        SandboxManager._containers = {"/path/a": mock_c1, "/path/b": mock_c2}

        with patch('assist.sandbox_manager.confirm_generation_stopped') as stopped:
            SandboxManager.cleanup_all()

        self.assertEqual(stopped.call_count, 2)
        stopped.assert_any_call(mock_c1.id)
        stopped.assert_any_call(mock_c2.id)
        self.assertEqual(len(SandboxManager._containers), 0)

    def test_unconfirmed_cleanup_retains_generation(self):
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)
        container = MagicMock()
        container.id = "old-generation"
        SandboxManager._containers[test_path] = container
        with patch('assist.sandbox_manager.confirm_generation_stopped',
                   side_effect=RuntimeError("stop unconfirmed")):
            with self.assertRaisesRegex(RuntimeError, "stop unconfirmed"):
                SandboxManager.cleanup(test_path, container)
        self.assertIs(SandboxManager._containers[test_path], container)

    def test_stale_cleanup_stops_only_captured_generation(self):
        from assist.browser import authority

        test_path = os.path.join(self.temp_dir, "thread", "domain")
        os.makedirs(test_path)
        authority.mark_new_thread(self.temp_dir, "thread")
        old = MagicMock(id="old-generation")
        new = MagicMock(id="new-generation")
        SandboxManager._containers[test_path] = new
        SandboxManager._generation_owners[old.id] = (
            self.temp_dir, "thread", "old-owner")
        with authority.fence(self.temp_dir, "thread") as state:
            state.begin("new-owner", 2)
            state.add_generation("new-owner", new.id)
        with patch('assist.sandbox_manager.confirm_generation_stopped') as stopped:
            SandboxManager.cleanup(test_path, old)
        stopped.assert_called_once_with("old-generation")
        self.assertIs(SandboxManager._containers[test_path], new)
        with authority.fence(self.temp_dir, "thread") as state:
            self.assertEqual(state.lease["owner_run_id"], "new-owner")
            self.assertEqual(state.lease["generations"], [new.id])

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
