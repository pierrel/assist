import os
import subprocess
import tempfile
import unittest
from assist.domain_manager import DomainManager


class TestDomainMerge(unittest.TestCase):
    def setUp(self):
        """Create a test git repository with a remote."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a bare repo to act as remote
        self.remote_dir = os.path.join(self.temp_dir, "remote.git")
        subprocess.run(['git', 'init', '--bare', self.remote_dir], check=True, capture_output=True)

        # Clone it to create a working repo
        self.repo_path = os.path.join(self.temp_dir, "work")
        subprocess.run(['git', 'clone', self.remote_dir, self.repo_path], check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=self.repo_path, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=self.repo_path, check=True)

        # Create initial commit on main
        test_file = os.path.join(self.repo_path, "README.md")
        with open(test_file, 'w') as f:
            f.write("# Test Repo\n")
        subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=self.repo_path, check=True)
        subprocess.run(['git', 'push', 'origin', 'main'], cwd=self.repo_path, check=True, capture_output=True)

        # Create and push feature branch
        subprocess.run(['git', 'checkout', '-b', 'feature/test'], cwd=self.repo_path, check=True, capture_output=True)
        with open(test_file, 'a') as f:
            f.write("\n## Features\n- Feature 1\n- Feature 2\n")
        subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Add features'], cwd=self.repo_path, check=True)
        subprocess.run(['git', 'push', '-u', 'origin', 'feature/test'], cwd=self.repo_path, check=True, capture_output=True)

    def tearDown(self):
        """Clean up test repository."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_merge_to_main_without_model(self):
        """Test merge without AI model (uses fallback summary)."""
        # Create DomainManager with repo that has remote
        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)

        # Verify we're on feature branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        self.assertEqual(result.stdout.strip(), 'feature/test')

        # Verify there are changes vs main
        diffs = dm.main_diff()
        self.assertGreater(len(diffs), 0)

        # Perform merge without model (uses fallback)
        summary = dm.merge_to_main(summary_model=None)

        # Verify summary was generated
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

        # Verify we're on a new branch (not the original feature branch)
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        current = result.stdout.strip()
        self.assertTrue(current.startswith('assist/'), f"Should be on new assist/* branch, got: {current}")
        self.assertNotEqual(current, 'feature/test', "Should not be on original branch")

        # Verify new branch has no diff against main
        dm_after = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)
        diffs_after = dm_after.main_diff()
        self.assertEqual(len(diffs_after), 0, "New branch should have no diff against main after merge")

        # Verify main has the changes
        subprocess.run(['git', 'checkout', 'main'], cwd=self.repo_path, check=True, capture_output=True)
        readme_path = os.path.join(self.repo_path, "README.md")
        with open(readme_path, 'r') as f:
            content = f.read()
        self.assertIn('Features', content)
        self.assertIn('Feature 1', content)

        # Verify remote branch was deleted
        result = subprocess.run(
            ['git', 'ls-remote', '--heads', 'origin', 'feature/test'],
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        self.assertEqual('', result.stdout.strip(), "Remote feature branch should be deleted")

    def test_merge_on_main_raises_error(self):
        """Test that merging while on main raises an error."""
        # Checkout main
        subprocess.run(['git', 'checkout', 'main'], cwd=self.repo_path, check=True, capture_output=True)

        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)

        with self.assertRaises(ValueError) as ctx:
            dm.merge_to_main()

        self.assertIn("Already on main branch", str(ctx.exception))

    def test_merge_with_no_changes_raises_error(self):
        """Test that merging with no changes raises an error."""
        # Create a branch with no changes
        subprocess.run(['git', 'checkout', 'main'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'checkout', '-b', 'empty-branch'], cwd=self.repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'push', '-u', 'origin', 'empty-branch'], cwd=self.repo_path, check=True, capture_output=True)

        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)

        with self.assertRaises(ValueError) as ctx:
            dm.merge_to_main()

        self.assertIn("No changes to merge", str(ctx.exception))

    def test_merge_without_remote_returns_no_changes(self):
        """Test that DomainManager without remote returns empty changes."""
        local_dir = os.path.join(self.temp_dir, "no_remote")
        os.makedirs(local_dir)
        subprocess.run(['git', 'init'], cwd=local_dir, check=True, capture_output=True)

        dm = DomainManager(repo_path=local_dir, repo=None)

        # Without a remote, git operations should be no-ops
        self.assertIsNone(dm.repo)
        self.assertEqual(dm.changes(), [])
        self.assertEqual(dm.main_diff(), [])

    def test_merge_deletes_old_remote_branch(self):
        """Test that merge deletes the old remote branch after squash merge."""
        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)

        # Verify remote branch exists before merge
        result = subprocess.run(
            ['git', 'ls-remote', '--heads', 'origin', 'feature/test'],
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        self.assertIn('feature/test', result.stdout, "Remote branch should exist before merge")

        # Perform merge
        summary = dm.merge_to_main(summary_model=None)

        # Verify remote branch was deleted
        result = subprocess.run(
            ['git', 'ls-remote', '--heads', 'origin', 'feature/test'],
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        self.assertEqual('', result.stdout.strip(), "Remote branch should be deleted after merge")


if __name__ == '__main__':
    unittest.main()
