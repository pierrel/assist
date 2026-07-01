"""Real-LLM eval: the agent manages its thread branch history via the git-history skill.

Each thread turn is a commit on the thread branch, so "undo / roll back / start over from
[a point]" is a ``git reset`` on that branch. This builds a real branch with per-turn
commits, runs the real agent in a real sandbox, and asserts the branch was rolled back to
the right point (the later work gone, the earlier work intact). Skips without Docker.
"""
import logging
import os
import shutil
import subprocess
import tempfile
from unittest import TestCase

from assist.agent import AgentHarness, create_agent
from assist.domain_manager import clone_repo
from assist.model_manager import select_assistant_model
from assist.sandbox_manager import SandboxManager

from .utils import executed_commands, skill_was_loaded

logger = logging.getLogger(__name__)


def _git(*args, cwd=None, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


class _HistoryScenario(TestCase):
    """Base: a thread branch with three per-turn commits (morning/noon/evening)."""

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="git_hist_eval_")
        self.origin = os.path.join(self.tmp, "origin.git")
        _git("init", "--bare", "-b", "main", self.origin)
        seed = os.path.join(self.tmp, "seed")
        _git("clone", self.origin, seed)
        _git("config", "user.email", "a@b", cwd=seed)
        _git("config", "user.name", "A", cwd=seed)
        self._write(seed, "README.md", "base\n")
        _git("add", ".", cwd=seed)
        _git("commit", "-m", "base", cwd=seed)
        _git("push", "origin", "main", cwd=seed)

        self.workspace = os.path.join(self.tmp, "work")
        clone_repo(self.origin, self.workspace)
        _git("config", "user.email", "a@b", cwd=self.workspace)
        _git("config", "user.name", "A", cwd=self.workspace)
        _git("checkout", "-b", "assist/hist-thread", cwd=self.workspace)

        self._commit("morning.txt", "morning plan\n", "morning: add the plan")
        self._commit("noon.txt", "noon draft\n", "noon: add the draft")
        self._commit("evening.txt", "evening notes\n", "evening: add notes")

        self.sandbox = SandboxManager.get_sandbox_backend(self.workspace)
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

    def _write(self, repo, path, content):
        with open(os.path.join(repo, path), "w") as f:
            f.write(content)

    def _commit(self, path, content, msg):
        self._write(self.workspace, path, content)
        _git("add", ".", cwd=self.workspace)
        _git("commit", "-m", msg, cwd=self.workspace)

    def _run(self, prompt):
        agent = AgentHarness(create_agent(self.model, self.workspace,
                                          sandbox_backend=self.sandbox))
        agent.message(prompt)
        return agent

    def _exists(self, name):
        return os.path.exists(os.path.join(self.workspace, name))

    def _diag(self, agent):
        cmds = executed_commands(agent)
        return ("\n  git-history skill loaded: " + str(skill_was_loaded(agent, "git-history"))
                + f"\n  execute calls ({len(cmds)}):\n    " + "\n    ".join(cmds[:30]))


class TestRollbackLastTurn(_HistoryScenario):
    def test_agent_undoes_the_last_turn(self):
        agent = self._run("Undo the last change — I don't want the evening notes. "
                          "Roll the branch back to before that.")
        self.assertFalse(self._exists("evening.txt"),
                         "the last turn's file was not rolled back" + self._diag(agent))
        self.assertTrue(self._exists("noon.txt"), "an earlier turn was lost" + self._diag(agent))
        self.assertTrue(self._exists("morning.txt"), "an earlier turn was lost" + self._diag(agent))


class TestRollbackToDescribedPoint(_HistoryScenario):
    def test_agent_rolls_back_to_the_morning(self):
        agent = self._run("Start over from the morning — keep the morning plan but drop "
                          "everything after it.")
        self.assertTrue(self._exists("morning.txt"),
                        "the morning work was lost" + self._diag(agent))
        self.assertFalse(self._exists("noon.txt"),
                         "work after the morning was not rolled back" + self._diag(agent))
        self.assertFalse(self._exists("evening.txt"),
                         "work after the morning was not rolled back" + self._diag(agent))
