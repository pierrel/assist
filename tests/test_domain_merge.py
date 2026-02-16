import os
import subprocess
import tempfile
import unittest
from assist.domain_manager import DomainManager


class TestDomainMerge(unittest.TestCase):
    def setUp(self):
        """Create a test git repository with a branch."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = os.path.join(self.temp_dir, "test_repo")
        os.makedirs(self.repo_path)

        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=self.repo_path, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=self.repo_path, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=self.repo_path, check=True)

        # Create initial commit on main
        test_file = os.path.join(self.repo_path, "README.md")
        with open(test_file, 'w') as f:
            f.write("# Test Repo\n")
        subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=self.repo_path, check=True)

        # Create a feature branch
        subprocess.run(['git', 'checkout', '-b', 'feature/test'], cwd=self.repo_path, check=True)

        # Make changes on feature branch
        with open(test_file, 'a') as f:
            f.write("\n## Features\n- Feature 1\n- Feature 2\n")
        subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Add features'], cwd=self.repo_path, check=True)

    def tearDown(self):
        """Clean up test repository."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_merge_to_main_without_model(self):
        """Test merge without AI model (uses fallback summary)."""
        # Create DomainManager with local repo (no remote)
        dm = DomainManager(repo_path=self.repo_path, repo=None)

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
        dm_after = DomainManager(repo_path=self.repo_path, repo=None)
        diffs_after = dm_after.main_diff()
        self.assertEqual(len(diffs_after), 0, "New branch should have no diff against main after merge")

        # Verify main has the changes
        subprocess.run(['git', 'checkout', 'main'], cwd=self.repo_path, check=True)
        readme_path = os.path.join(self.repo_path, "README.md")
        with open(readme_path, 'r') as f:
            content = f.read()
        self.assertIn('Features', content)
        self.assertIn('Feature 1', content)

    def test_merge_on_main_raises_error(self):
        """Test that merging while on main raises an error."""
        # Checkout main
        subprocess.run(['git', 'checkout', 'main'], cwd=self.repo_path, check=True)

        dm = DomainManager(repo_path=self.repo_path, repo=None)

        with self.assertRaises(ValueError) as ctx:
            dm.merge_to_main()

        self.assertIn("Already on main branch", str(ctx.exception))

    def test_merge_with_no_changes_raises_error(self):
        """Test that merging with no changes raises an error."""
        # Create a branch with no changes
        subprocess.run(['git', 'checkout', 'main'], cwd=self.repo_path, check=True)
        subprocess.run(['git', 'checkout', '-b', 'empty-branch'], cwd=self.repo_path, check=True)

        dm = DomainManager(repo_path=self.repo_path, repo=None)

        with self.assertRaises(ValueError) as ctx:
            dm.merge_to_main()

        self.assertIn("No changes to merge", str(ctx.exception))

    def test_merge_with_no_repo_raises_error(self):
        """Test that merging without a repo raises an error."""
        # Create DomainManager without a repo
        dm = DomainManager(repo_path=self.temp_dir, repo=None)

        with self.assertRaises(ValueError) as ctx:
            dm.merge_to_main()

        self.assertIn("no repository configured", str(ctx.exception))

    def test_merge_deletes_old_remote_branch(self):
        """Test that merge deletes the old remote branch after squash merge."""
        # Create a bare repo to act as remote
        remote_dir = os.path.join(self.temp_dir, "remote.git")
        subprocess.run(['git', 'init', '--bare', remote_dir], check=True)

        # Clone it to create a working repo
        work_dir = os.path.join(self.temp_dir, "work")
        subprocess.run(['git', 'clone', remote_dir, work_dir], check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=work_dir, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=work_dir, check=True)

        # Create initial commit
        test_file = os.path.join(work_dir, "README.md")
        with open(test_file, 'w') as f:
            f.write("# Test Repo\n")
        subprocess.run(['git', 'add', '.'], cwd=work_dir, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=work_dir, check=True)
        subprocess.run(['git', 'push', 'origin', 'main'], cwd=work_dir, check=True)

        # Create and push feature branch
        subprocess.run(['git', 'checkout', '-b', 'feature/test'], cwd=work_dir, check=True)
        with open(test_file, 'a') as f:
            f.write("\n## Features\n")
        subprocess.run(['git', 'add', '.'], cwd=work_dir, check=True)
        subprocess.run(['git', 'commit', '-m', 'Add features'], cwd=work_dir, check=True)
        subprocess.run(['git', 'push', '-u', 'origin', 'feature/test'], cwd=work_dir, check=True)

        # Verify remote branch exists
        result = subprocess.run(
            ['git', 'ls-remote', '--heads', 'origin', 'feature/test'],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        self.assertIn('feature/test', result.stdout, "Remote branch should exist before merge")

        # Perform merge
        dm = DomainManager(repo_path=work_dir, repo=remote_dir)
        summary = dm.merge_to_main(summary_model=None)

        # Verify remote branch was deleted
        result = subprocess.run(
            ['git', 'ls-remote', '--heads', 'origin', 'feature/test'],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        self.assertEqual('', result.stdout.strip(), "Remote branch should be deleted after merge")


if __name__ == '__main__':
    unittest.main()
