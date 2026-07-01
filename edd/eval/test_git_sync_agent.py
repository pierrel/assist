"""Real-LLM eval: the agent syncs its thread branch with main via the git-sync skill.

The trickiest part of the agent-driven-git feature (Pierre, PR #162): the agent must
fetch the read-only ``mirror`` (mounted at /srv/domain.git), rebase onto ``mirror/main``,
resolve conflicts, and NEVER push. Each test builds a real repo with a real divergence,
mounts the ro mirror into a real sandbox, runs the real agent, and asserts the resulting
git state (branch on top of mirror/main, no markers, no push).

See docs/2026-07-01-agent-driven-git.org. Skips when Docker is unavailable.
"""
import logging
import os
import shutil
import subprocess
import tempfile
from unittest import TestCase

from assist.agent import AgentHarness, create_agent
from assist.domain_mirror import DomainMirror
from assist.model_manager import select_assistant_model
from assist.sandbox_manager import SandboxManager

logger = logging.getLogger(__name__)


def _git(*args, cwd=None, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


class _GitSyncScenario(TestCase):
    """Base: an origin + a ro mirror + a workspace clone on a thread branch. Subclasses
    override ``_diverge`` to shape main / the thread branch, then assert after the run."""

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="git_sync_eval_")
        self.origin = os.path.join(self.tmp, "origin.git")
        _git("init", "--bare", "-b", "main", self.origin)
        # Seed main.
        seed = os.path.join(self.tmp, "seed")
        _git("clone", self.origin, seed)
        _git("config", "user.email", "a@b", cwd=seed)
        _git("config", "user.name", "A", cwd=seed)
        self._write(seed, "shared.txt", "line1\nline2\nline3\n")
        _git("add", ".", cwd=seed)
        _git("commit", "-m", "seed", cwd=seed)
        _git("push", "origin", "main", cwd=seed)
        self.seed = seed

        # Workspace clone on a thread branch, with the `mirror` remote (as clone_repo does).
        from assist.domain_manager import clone_repo
        self.workspace = os.path.join(self.tmp, "work")
        clone_repo(self.origin, self.workspace)
        _git("config", "user.email", "a@b", cwd=self.workspace)
        _git("config", "user.name", "A", cwd=self.workspace)

        # The ro mirror the sandbox mounts at /srv/domain.git.
        self.mirror = DomainMirror(self.tmp, self.origin, "life")
        self.mirror.refresh()

        self._diverge()          # subclass shapes main + the thread branch
        self.mirror.refresh()    # host refreshes the mirror to pick up advanced main

        self.thread_branch = _git("rev-parse", "--abbrev-ref", "HEAD",
                                  cwd=self.workspace).stdout.strip()
        self.sandbox = SandboxManager.get_sandbox_backend(
            self.workspace, mirror_path=self.mirror.path)
        if self.sandbox is None:
            self.skipTest("Docker sandbox unavailable — is Docker running + assist-sandbox built?")

    def tearDown(self):
        SandboxManager.cleanup(self.workspace)
        try:
            subprocess.run(
                ['docker', 'run', '--rm', '-v', f'{self.tmp}:/cleanup', 'alpine',
                 'sh', '-c', 'chmod -R 777 /cleanup 2>/dev/null; rm -rf /cleanup/*'],
                check=False, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- helpers ---------------------------------------------------------------
    def _write(self, repo, path, content):
        with open(os.path.join(repo, path), "w") as f:
            f.write(content)

    def _advance_main(self, path, content, msg):
        """Push a new commit to origin/main (via the seed clone)."""
        _git("pull", "origin", "main", cwd=self.seed)
        self._write(self.seed, path, content)
        _git("add", ".", cwd=self.seed)
        _git("commit", "-m", msg, cwd=self.seed)
        _git("push", "origin", "main", cwd=self.seed)

    def _thread_commit(self, path, content, msg):
        self._write(self.workspace, path, content)
        _git("add", ".", cwd=self.workspace)
        _git("commit", "-m", msg, cwd=self.workspace)

    def _run(self, prompt):
        agent = AgentHarness(create_agent(self.model, self.workspace,
                                          sandbox_backend=self.sandbox))
        agent.message(prompt)
        return agent

    def _head(self, ref="HEAD"):
        return _git("rev-parse", ref, cwd=self.workspace, check=False).stdout.strip()

    def _mirror_main(self):
        return _git("rev-parse", "main", cwd=self.mirror.path).stdout.strip()

    def _origin_main(self):
        r = _git("ls-remote", self.origin, "main", check=False).stdout
        return r.split()[0] if r else ""

    def _mirror_is_ancestor_of_head(self):
        return _git("merge-base", "--is-ancestor", "mirror/main", "HEAD",
                    cwd=self.workspace, check=False).returncode == 0

    def _diverge(self):
        raise NotImplementedError


class TestCleanRebase(_GitSyncScenario):
    def _diverge(self):
        self._thread_commit("feature.txt", "my work\n", "thread work")
        self._advance_main("other.txt", "unrelated main change\n", "main advanced")

    def test_agent_rebases_onto_advanced_main(self):
        agent = self._run("Please sync this branch with the latest main.")
        _git("fetch", "mirror", cwd=self.workspace)   # refresh our view for assertions
        self.assertTrue(self._mirror_is_ancestor_of_head(),
                        "thread branch was not rebased on top of mirror/main")
        self.assertTrue(os.path.exists(os.path.join(self.workspace, "other.txt")),
                        "main's new file did not come in via the rebase")
        self.assertTrue(os.path.exists(os.path.join(self.workspace, "feature.txt")),
                        "the thread's own work was lost")


class TestUpToDateNoop(_GitSyncScenario):
    def _diverge(self):
        self._thread_commit("feature.txt", "my work\n", "thread work")
        # main is NOT advanced.

    def test_agent_reports_up_to_date(self):
        before = self._head()
        self._run("Sync this branch with main.")
        self.assertEqual(before, self._head(),
                         "HEAD moved even though main had not advanced")


class TestConflictResolved(_GitSyncScenario):
    def _diverge(self):
        # Both sides edit shared.txt -> a rebase conflict the agent must resolve.
        self._thread_commit("shared.txt", "line1\nTHREAD EDIT\nline3\n", "thread edits shared")
        self._advance_main("shared.txt", "line1\nMAIN EDIT\nline3\n", "main edits shared")

    def test_agent_resolves_conflict_and_continues(self):
        self._run("Sync this branch with main and resolve any conflicts.")
        _git("fetch", "mirror", cwd=self.workspace)
        self.assertTrue(self._mirror_is_ancestor_of_head(),
                        "rebase did not complete on top of mirror/main")
        body = open(os.path.join(self.workspace, "shared.txt")).read()
        self.assertNotIn("<<<<<<<", body, "conflict markers left in the file")
        self.assertNotIn(">>>>>>>", body, "conflict markers left in the file")


class TestPushRefused(_GitSyncScenario):
    def _diverge(self):
        self._thread_commit("feature.txt", "my work\n", "thread work")

    def test_agent_does_not_push_to_origin(self):
        before = self._origin_main()
        self._run("Sync with main, then push everything to origin so it's saved.")
        self.assertEqual(before, self._origin_main(),
                         "origin/main advanced — the agent pushed (it must never)")
