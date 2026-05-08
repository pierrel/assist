import os
import subprocess
import tempfile
import unittest

from assist.domain_manager import (
    DomainManager,
    MergeConflictError,
    OriginAdvancedError,
)


class TestDomainMerge(unittest.TestCase):
    """Exercises the rebase-then-squash merge contract introduced by the
    per-thread web git isolation work (see
    ``docs/2026-05-07-per-thread-web-git-isolation.org``).

    Fixture layout: a bare ``remote.git`` plus two clones —
    ``self.repo_path`` is the deploy-box working clone (under test);
    ``self._external_clone()`` returns a separate clone that simulates
    a different machine pushing to the shared remote.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Bare remote with `main` as the default branch (independent of
        # the runner's init.defaultBranch).
        self.remote_dir = os.path.join(self.temp_dir, "remote.git")
        subprocess.run(
            ['git', 'init', '--bare', '-b', 'main', self.remote_dir],
            check=True, capture_output=True,
        )

        # Working clone under test.
        self.repo_path = os.path.join(self.temp_dir, "work")
        subprocess.run(
            ['git', 'clone', self.remote_dir, self.repo_path],
            check=True, capture_output=True,
        )
        self._configure_identity(self.repo_path)

        # Seed an initial commit on `main` and push it.
        with open(os.path.join(self.repo_path, "README.md"), 'w') as f:
            f.write("# Test Repo\n")
        subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=self.repo_path, check=True)
        subprocess.run(
            ['git', 'push', 'origin', 'main'], cwd=self.repo_path,
            check=True, capture_output=True,
        )

        # Branch off `main` to simulate a thread branch.
        subprocess.run(
            ['git', 'checkout', '-b', 'feature/test'], cwd=self.repo_path,
            check=True, capture_output=True,
        )
        with open(os.path.join(self.repo_path, "README.md"), 'a') as f:
            f.write("\n## Features\n- Feature 1\n- Feature 2\n")
        subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Add features'], cwd=self.repo_path, check=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @staticmethod
    def _configure_identity(repo: str) -> None:
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=repo, check=True)

    def _external_clone(self) -> str:
        """Spawn a separate clone of the bare remote — used to simulate
        another machine pushing to ``origin/main`` while the working
        clone stays unaware until it ``fetch``es.
        """
        path = os.path.join(self.temp_dir, "external")
        subprocess.run(['git', 'clone', self.remote_dir, path], check=True, capture_output=True)
        self._configure_identity(path)
        return path

    def _origin_main_sha(self) -> str:
        result = subprocess.run(
            ['git', '-C', self.repo_path, 'ls-remote', 'origin', 'main'],
            stdout=subprocess.PIPE, text=True, check=True,
        )
        return result.stdout.split()[0] if result.stdout else ""

    # --- Pre-existing contract regressions ----------------------------------

    def test_merge_to_main_without_model(self):
        """Clean ff path: rebase is a no-op, squash lands on main, new
        thread branch is created off the post-merge main.
        """
        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)
        self.assertGreater(len(dm.main_diff()), 0)

        summary = dm.merge_to_main(summary_model=None)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

        # We should have rolled forward onto a fresh `assist/...` branch.
        cur = subprocess.run(
            ['git', '-C', self.repo_path, 'rev-parse', '--abbrev-ref', 'HEAD'],
            stdout=subprocess.PIPE, text=True, check=True,
        ).stdout.strip()
        self.assertTrue(cur.startswith('assist/'), f"expected assist/* branch, got: {cur}")
        self.assertNotEqual(cur, 'feature/test')

        # No diff vs the (post-merge) main.
        self.assertEqual(DomainManager(repo_path=self.repo_path, repo=self.remote_dir).main_diff(), [])

        # The squashed work landed on local main.
        subprocess.run(['git', '-C', self.repo_path, 'checkout', 'main'], check=True, capture_output=True)
        with open(os.path.join(self.repo_path, "README.md")) as f:
            content = f.read()
        self.assertIn('Feature 1', content)

    def test_merge_on_main_raises_error(self):
        subprocess.run(
            ['git', '-C', self.repo_path, 'checkout', 'main'],
            check=True, capture_output=True,
        )
        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)
        with self.assertRaises(ValueError) as ctx:
            dm.merge_to_main()
        self.assertIn("Already on main branch", str(ctx.exception))

    def test_merge_with_no_changes_raises_error(self):
        subprocess.run(['git', '-C', self.repo_path, 'checkout', 'main'], check=True, capture_output=True)
        subprocess.run(['git', '-C', self.repo_path, 'checkout', '-b', 'empty-branch'], check=True, capture_output=True)
        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)
        with self.assertRaises(ValueError) as ctx:
            dm.merge_to_main()
        self.assertIn("No changes to merge", str(ctx.exception))

    def test_merge_without_remote_returns_no_changes(self):
        local_dir = os.path.join(self.temp_dir, "no_remote")
        os.makedirs(local_dir)
        subprocess.run(['git', 'init'], cwd=local_dir, check=True, capture_output=True)
        dm = DomainManager(repo_path=local_dir, repo=None)
        self.assertIsNone(dm.repo)
        self.assertEqual(dm.changes(), [])
        self.assertEqual(dm.main_diff(), [])

    # --- New contract under per-thread web git isolation --------------------

    def test_merge_does_not_push(self):
        """``merge_to_main`` must not write to ``origin/main`` — the
        push is exclusively the new ``push_main`` endpoint's job.
        """
        before = self._origin_main_sha()
        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)
        dm.merge_to_main(summary_model=None)
        after = self._origin_main_sha()
        self.assertEqual(before, after, "merge_to_main pushed to origin/main; it must not")

    def test_merge_rebases_onto_advanced_origin(self):
        """When ``origin/main`` has advanced (clean rebase, no
        conflict), the merge must integrate the remote changes plus
        the thread's work into local main without divergence.
        """
        # Push a non-conflicting change to origin/main from a separate clone.
        external = self._external_clone()
        with open(os.path.join(external, "external.txt"), 'w') as f:
            f.write("from-elsewhere\n")
        subprocess.run(['git', 'add', '.'], cwd=external, check=True)
        subprocess.run(['git', 'commit', '-m', 'External change'], cwd=external, check=True)
        subprocess.run(['git', 'push', 'origin', 'main'], cwd=external, check=True, capture_output=True)

        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)
        dm.merge_to_main(summary_model=None)

        # Local main now contains both the external change and the
        # squashed thread work.
        subprocess.run(['git', '-C', self.repo_path, 'checkout', 'main'], check=True, capture_output=True)
        self.assertTrue(os.path.isfile(os.path.join(self.repo_path, "external.txt")))
        with open(os.path.join(self.repo_path, "README.md")) as f:
            self.assertIn('Feature 1', f.read())

    def test_merge_conflict_aborts_and_raises_typed_error(self):
        """A real rebase conflict must raise ``MergeConflictError`` with
        the unmerged file list, then leave the work tree clean back on
        the thread branch (rebase aborted).
        """
        # External clone modifies README.md in a way that conflicts with
        # the feature branch's edit.
        external = self._external_clone()
        with open(os.path.join(external, "README.md"), 'w') as f:
            f.write("# Different content entirely\n")
        subprocess.run(['git', 'add', '.'], cwd=external, check=True)
        subprocess.run(['git', 'commit', '-m', 'Conflicting external change'], cwd=external, check=True)
        subprocess.run(['git', 'push', 'origin', 'main'], cwd=external, check=True, capture_output=True)

        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)
        with self.assertRaises(MergeConflictError) as ctx:
            dm.merge_to_main(summary_model=None)

        self.assertEqual(ctx.exception.branch, 'feature/test')
        self.assertIn('README.md', ctx.exception.files)

        # Rebase aborted — back on feature/test, no in-progress rebase dir.
        cur = subprocess.run(
            ['git', '-C', self.repo_path, 'rev-parse', '--abbrev-ref', 'HEAD'],
            stdout=subprocess.PIPE, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(cur, 'feature/test')
        self.assertFalse(os.path.isdir(os.path.join(self.repo_path, '.git', 'rebase-merge')))
        self.assertFalse(os.path.isdir(os.path.join(self.repo_path, '.git', 'rebase-apply')))

    def test_merge_dirty_worktree_does_not_raise_fake_conflict(self):
        """A non-conflict rebase failure (dirty worktree, branch
        rename, etc.) must NOT be papered over as ``MergeConflictError``
        — that would send the agent chasing a nonexistent conflict.
        """
        # Dirty the worktree so `git rebase` refuses with a non-conflict
        # error ("Cannot rebase: You have unstaged changes.").
        with open(os.path.join(self.repo_path, "README.md"), 'a') as f:
            f.write("\nuncommitted local edit\n")

        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)
        with self.assertRaises(subprocess.CalledProcessError):
            dm.merge_to_main(summary_model=None)

        # Also assert it specifically didn't raise MergeConflictError
        # — a separate run because the assertRaises above consumed
        # the first.  Re-dirty if needed (still dirty from above).
        try:
            dm.merge_to_main(summary_model=None)
        except MergeConflictError:
            self.fail("dirty-worktree rebase failure was misclassified as MergeConflictError")
        except subprocess.CalledProcessError:
            pass  # expected

    def test_merge_refuses_when_local_main_has_unpushed_commits(self):
        """A second merge before pushing the first must refuse, so two
        threads' worth of unpushed commits don't pile onto local main.
        """
        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)
        dm.merge_to_main(summary_model=None)
        # Now on a fresh assist/* branch.  Add another change to merge.
        with open(os.path.join(self.repo_path, "second.txt"), 'w') as f:
            f.write("second turn\n")
        subprocess.run(['git', '-C', self.repo_path, 'add', '.'], check=True)
        subprocess.run(['git', '-C', self.repo_path, 'commit', '-m', 'Second turn'], check=True)
        with self.assertRaises(ValueError) as ctx:
            dm.merge_to_main(summary_model=None)
        self.assertIn("unpushed commits", str(ctx.exception))

    # --- push_main ----------------------------------------------------------

    def test_push_main_succeeds_when_local_ahead(self):
        before = self._origin_main_sha()
        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)
        dm.merge_to_main(summary_model=None)
        self.assertTrue(dm.has_unpushed_main())

        dm.push_main()
        after = self._origin_main_sha()
        self.assertNotEqual(before, after, "push_main did not advance origin/main")
        self.assertFalse(dm.has_unpushed_main())

    def test_push_main_rejects_when_origin_advanced(self):
        """If origin/main moves between merge and push, push_main must
        refuse — silently overwriting remote work would be the worst
        outcome.
        """
        dm = DomainManager(repo_path=self.repo_path, repo=self.remote_dir)
        dm.merge_to_main(summary_model=None)

        # Externally advance origin/main past the merge commit.
        external = self._external_clone()
        subprocess.run(['git', 'pull', 'origin', 'main'], cwd=external, check=True, capture_output=True)
        with open(os.path.join(external, "race.txt"), 'w') as f:
            f.write("race\n")
        subprocess.run(['git', 'add', '.'], cwd=external, check=True)
        subprocess.run(['git', 'commit', '-m', 'Racing change'], cwd=external, check=True)
        subprocess.run(['git', 'push', 'origin', 'main'], cwd=external, check=True, capture_output=True)

        with self.assertRaises(OriginAdvancedError):
            dm.push_main()


if __name__ == '__main__':
    unittest.main()
