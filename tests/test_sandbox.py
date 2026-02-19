"""Unit tests for Docker sandbox backend and DomainManager Docker lifecycle.

All Docker interactions are mocked — no real Docker daemon needed.
"""
import os
import tempfile
import shutil
from unittest import TestCase
from unittest.mock import patch, MagicMock, PropertyMock

from assist.sandbox import DockerSandboxBackend, MAX_OUTPUT_CHARS
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
        self.container.exec_run.assert_called_once_with(
            ["bash", "-c", "echo hello world"], demux=False, workdir="/workspace")

    def test_execute_nonzero_exit(self):
        self.container.exec_run.return_value = (1, b"error: not found\n")
        resp = self.sandbox.execute("cat /missing")

        self.assertEqual(resp.exit_code, 1)
        self.assertIn("not found", resp.output)

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
        mock_client.containers.run.return_value = mock_container

        with patch.object(SandboxManager, '_get_docker_client', return_value=mock_client):
            sandbox = SandboxManager.get_sandbox_backend(test_path)

        self.assertIsNotNone(sandbox)
        mock_client.containers.run.assert_called_once()
        # Verify container is registered
        self.assertIn(test_path, SandboxManager._containers)

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
        test_path = os.path.join(self.temp_dir, "domain")
        os.makedirs(test_path)

        mock_client = MagicMock()
        mock_client.containers.run.side_effect = Exception("Docker not running")

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

    def test_factory_returns_callable(self):
        from assist.backends import create_sandbox_composite_backend
        mock_sandbox = MagicMock()
        factory = create_sandbox_composite_backend(mock_sandbox)
        self.assertTrue(callable(factory))

    def test_factory_creates_composite_with_sandbox_default(self):
        from assist.backends import create_sandbox_composite_backend
        from deepagents.backends import CompositeBackend

        mock_sandbox = MagicMock()
        factory = create_sandbox_composite_backend(mock_sandbox)

        mock_rt = MagicMock()
        backend = factory(mock_rt)

        self.assertIsInstance(backend, CompositeBackend)
