"""Unit tests for DomainManager.

These tests mock filesystem and git operations to avoid calling models.
"""
import os
import tempfile
import shutil
from unittest import TestCase
from unittest.mock import patch, MagicMock
from assist.domain_manager import DomainManager, Change, git_diff, git_commit, git_push


class TestDomainManager(TestCase):
    """Test DomainManager class without calling models."""

    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('assist.domain_manager.clone_repo')
    def test_domain_manager_creates_directory(self, mock_clone):
        """Test that DomainManager calls clone_repo if directory doesn't exist."""
        test_path = os.path.join(self.temp_dir, "test_domain")
        dm = DomainManager(repo_path=test_path, repo="https://example.com/repo.git")

        self.assertEqual(dm.domain(), test_path)
        self.assertEqual(dm.repo, "https://example.com/repo.git")
        # Should have called clone since directory doesn't exist
        mock_clone.assert_called_once_with("https://example.com/repo.git", test_path)

    @patch('assist.domain_manager.git_repo')
    def test_domain_manager_uses_existing_directory(self, mock_git_repo):
        """Test that DomainManager uses an existing directory with remote."""
        test_path = os.path.join(self.temp_dir, "existing_domain")
        os.makedirs(test_path)

        # Mock that this is already a git repo with remote
        mock_git_repo.return_value = "https://example.com/repo.git"

        dm = DomainManager(repo_path=test_path)
        self.assertEqual(dm.domain(), test_path)
        self.assertEqual(dm.repo, "https://example.com/repo.git")

    @patch('assist.domain_manager.clone_repo')
    def test_domain_manager_with_temp_dir(self, mock_clone):
        """Test that DomainManager creates a temp dir when no path given."""
        dm = DomainManager(repo="https://example.com/repo.git")

        # Should have created a directory
        self.assertTrue(os.path.isdir(dm.domain()))
        # Should start with /tmp (on most systems)
        self.assertTrue(dm.domain().startswith('/tmp') or dm.domain().startswith('/var'))

    def test_domain_manager_works_without_remote(self):
        """Test that DomainManager works without git remote (sandbox-only mode)."""
        test_path = os.path.join(self.temp_dir, "non_repo")

        dm = DomainManager(repo_path=test_path, repo=None)

        self.assertIsNone(dm.repo)
        self.assertTrue(os.path.isdir(test_path))
        self.assertEqual(dm.changes(), [])
        self.assertEqual(dm.main_diff(), [])

    @patch('assist.domain_manager.subprocess.run')
    def test_git_diff_with_changes(self, mock_run):
        """Test git_diff returns Change objects."""
        # Mock git diff --name-only
        mock_name_result = MagicMock()
        mock_name_result.returncode = 0
        mock_name_result.stdout = "file1.txt\nfile2.txt\n"

        # Mock git diff for each file
        mock_diff_result = MagicMock()
        mock_diff_result.returncode = 0
        mock_diff_result.stdout = "diff --git a/file1.txt b/file1.txt\n+new line"

        # Mock git ls-files (no untracked)
        mock_ls_result = MagicMock()
        mock_ls_result.returncode = 0
        mock_ls_result.stdout = ""

        # Set up the mock to return different values for different commands
        def side_effect(cmd, **kwargs):
            if 'diff' in cmd and '--name-only' in cmd:
                return mock_name_result
            elif 'diff' in cmd and '--no-color' in cmd:
                return mock_diff_result
            elif 'ls-files' in cmd:
                return mock_ls_result
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        test_path = os.path.join(self.temp_dir, "git_repo")
        os.makedirs(test_path)

        changes = git_diff(test_path)

        # Should have 2 changes (one for each file)
        self.assertEqual(len(changes), 2)
        self.assertIsInstance(changes[0], Change)
        self.assertEqual(changes[0].path, "file1.txt")

    @patch('assist.domain_manager.subprocess.run')
    def test_git_commit_with_no_changes(self, mock_run):
        """Test git_commit does nothing when no changes."""
        # Mock git add -A
        mock_add_result = MagicMock()
        mock_add_result.returncode = 0

        # Mock git diff --cached --quiet (returns 0 when no changes)
        mock_diff_result = MagicMock()
        mock_diff_result.returncode = 0

        def side_effect(cmd, **kwargs):
            if 'add' in cmd:
                return mock_add_result
            elif 'diff' in cmd and '--cached' in cmd:
                return mock_diff_result
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        test_path = os.path.join(self.temp_dir, "git_repo")
        os.makedirs(test_path)

        # Should not raise an error
        git_commit(test_path, "test commit")

        # Should not have called git commit
        commit_calls = [call for call in mock_run.call_args_list
                       if len(call[0]) > 0 and 'commit' in call[0][0]]
        self.assertEqual(len(commit_calls), 0)

    @patch('assist.domain_manager.subprocess.run')
    def test_git_commit_with_changes(self, mock_run):
        """Test git_commit commits when there are changes."""
        # Mock git add -A
        mock_add_result = MagicMock()
        mock_add_result.returncode = 0

        # Mock git diff --cached --quiet (returns 1 when changes exist)
        mock_diff_result = MagicMock()
        mock_diff_result.returncode = 1

        # Mock git commit
        mock_commit_result = MagicMock()
        mock_commit_result.returncode = 0

        def side_effect(cmd, **kwargs):
            if 'add' in cmd:
                return mock_add_result
            elif 'diff' in cmd and '--cached' in cmd:
                return mock_diff_result
            elif 'commit' in cmd:
                return mock_commit_result
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        test_path = os.path.join(self.temp_dir, "git_repo")
        os.makedirs(test_path)

        git_commit(test_path, "test commit")

        # Should have called git commit
        commit_calls = [call for call in mock_run.call_args_list
                       if len(call[0]) > 0 and 'commit' in call[0][0]]
        self.assertEqual(len(commit_calls), 1)
        self.assertIn("test commit", str(commit_calls[0]))


class TestDomainManagerIntegration(TestCase):
    """Integration tests that create real git repos but don't call models."""

    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('assist.domain_manager.git_repo')
    def test_domain_manager_directory_structure(self, mock_git_repo):
        """Test that domain manager creates proper directory structure."""
        test_path = os.path.join(self.temp_dir, "domain")

        # Mock that this is already a git repo with remote
        mock_git_repo.return_value = "https://example.com/repo.git"

        dm = DomainManager(repo_path=test_path)

        # Create a subdirectory
        subdir = os.path.join(dm.domain(), "gtd")
        os.makedirs(subdir)

        # Create a file
        file_path = os.path.join(subdir, "inbox.org")
        with open(file_path, "w") as f:
            f.write("* Tasks\n")

        # Verify structure
        self.assertTrue(os.path.isdir(subdir))
        self.assertTrue(os.path.isfile(file_path))

        with open(file_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "* Tasks\n")

    @patch('assist.domain_manager.clone_repo')
    def test_domain_manager_handles_empty_directory(self, mock_clone):
        """Test that DomainManager works with pre-existing empty directories."""
        # Simulate what ThreadManager does: create an empty working directory
        test_path = os.path.join(self.temp_dir, "empty_domain")
        os.makedirs(test_path, exist_ok=True)

        # Verify it's empty
        self.assertTrue(os.path.isdir(test_path))
        self.assertEqual(len(os.listdir(test_path)), 0)

        # Create DomainManager with a repo URL (would clone in real scenario)
        dm = DomainManager(repo_path=test_path, repo="https://example.com/repo.git")
        self.assertEqual(dm.domain(), test_path)
        self.assertTrue(os.path.isdir(dm.domain()))
