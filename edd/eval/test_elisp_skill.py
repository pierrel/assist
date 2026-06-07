"""Skill-loading + behavior evals for the built-in ``elisp`` skill.

The elisp skill is generic Emacs-Lisp craft: file structure, the headless
``emacs --batch`` invocation, and — its headline contract — the
verify-it-yourself loop (byte-compile / ERT) so the small model doesn't rely on
shaky elisp recall. The main eval runs in a Docker sandbox because that contract
is *behavioral*: the agent must actually run emacs to verify, which needs a real
``execute`` and the emacs bundled in the sandbox image.

- ``test_loads_and_verifies`` — a plain "write this elisp and make sure it
  works" request (no skill vocabulary, no "compile"/"test"/"lint"); asserts the
  skill loaded AND the agent actually exercised the code in ``emacs --batch``
  rather than emitting it unchecked.
- ``test_does_not_load_on_python_task`` (no sandbox) — anti-test pinning the
  trigger: a Python task must not load the elisp skill.
"""
import os
import re
import shutil
import subprocess
import tempfile
from unittest import TestCase

from langchain_core.messages import AIMessage

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model
from assist.sandbox_manager import SandboxManager


def _skill_was_loaded(agent, skill_name: str) -> bool:
    """True iff a tool call loaded the named skill (load_skill or the
    upstream /skills/<name>/ read path). Local to this suite, mirroring the
    other skill evals so they can drift independently."""
    path_needle = f"/skills/{skill_name}/"
    for m in agent.all_messages():
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            args = tc.get("args") or {}
            if tc.get("name") == "load_skill" and args.get("name") == skill_name:
                return True
            for v in args.values():
                if isinstance(v, str) and path_needle in v:
                    return True
    return False


def _executed_commands(agent) -> list[str]:
    """Command strings from every ``execute`` tool call."""
    cmds = []
    for m in agent.all_messages():
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            if tc.get("name") == "execute":
                cmd = (tc.get("args") or {}).get("command", "")
                if cmd:
                    cmds.append(cmd)
    return cmds


def _cleanup_workspace(path: str) -> None:
    """Remove a sandbox workspace, using Docker to delete root-owned files.
    Mirrors the helper in test_calculate_skill.py / test_dev_agent.py."""
    try:
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{path}:/cleanup', 'alpine',
             'sh', '-c', 'chmod -R 777 /cleanup 2>/dev/null; rm -rf /cleanup/*'],
            check=False, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    shutil.rmtree(path, ignore_errors=True)


class TestElispSkillSandbox(TestCase):
    """The skill's contract is behavioral (run emacs to verify), so this needs
    a real sandbox with the bundled emacs."""

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.workspace = tempfile.mkdtemp(prefix="elisp_skill_eval_")
        self.sandbox = SandboxManager.get_sandbox_backend(self.workspace)
        if self.sandbox is None:
            self.skipTest(
                "Docker sandbox unavailable — is Docker running and "
                "assist-sandbox built?"
            )

    def tearDown(self):
        SandboxManager.cleanup(self.workspace)
        _cleanup_workspace(self.workspace)

    def _ran_emacs_batch(self, agent) -> bool:
        """True iff the agent exercised code in ``emacs --batch`` — byte-compile,
        ERT, checkdoc, or load-and-eval all count. The contract is that the agent
        *verified in emacs* rather than eyeballing; the exact form is the model's
        choice."""
        for cmd in _executed_commands(agent):
            if re.search(r"\bemacs\b", cmd) and ("--batch" in cmd or "-batch" in cmd):
                return True
        return False

    def test_loads_and_verifies(self):
        agent = AgentHarness(create_agent(
            self.model, self.workspace, sandbox_backend=self.sandbox))
        agent.message(
            "Write an Emacs Lisp function `demo/sum-list` that returns the sum "
            "of a list of numbers, in a file `sum.el`. Make sure it actually "
            "works."
        )
        self.assertTrue(
            _skill_was_loaded(agent, "elisp"),
            "agent did not load the elisp skill for an Emacs-Lisp authoring task",
        )
        self.assertTrue(
            self._ran_emacs_batch(agent),
            "elisp skill loaded but the agent never ran emacs to verify — it "
            "should byte-compile / run ERT, not just emit elisp and trust it",
        )

    def _sandbox_sh(self, script: str) -> subprocess.CompletedProcess:
        """Run a shell snippet in a fresh sandbox container against the
        workspace, capturing output — used to independently verify the elisp
        the agent produced (not to drive the agent)."""
        return subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workspace}:/workspace',
             '-w', '/workspace', 'assist-sandbox', 'bash', '-c', script],
            capture_output=True, text=True, timeout=120)

    def test_writes_wellformed_elisp(self):
        """The skill's payoff: the agent produces elisp that is well-formed and
        actually works. We pin the file/function names, then INDEPENDENTLY
        byte-compile the artifact and run it — the eval verifies the output, not
        just that the agent ran something."""
        agent = AgentHarness(create_agent(
            self.model, self.workspace, sandbox_backend=self.sandbox))
        agent.message(
            "Create an Emacs Lisp file `mathy.el` that provides a function "
            "`mathy-factorial` taking one non-negative integer N and returning "
            "N factorial — so (mathy-factorial 5) returns 120. Give the file a "
            "proper Commentary section. Make sure it byte-compiles cleanly."
        )
        self.assertTrue(
            _skill_was_loaded(agent, "elisp"),
            "agent did not load the elisp skill for an Emacs-Lisp authoring task",
        )
        path = os.path.join(self.workspace, "mathy.el")
        self.assertTrue(os.path.exists(path), "agent did not write mathy.el")

        # Well-formed: lexical-binding cookie present.
        src = open(path).read()
        self.assertRegex(
            src, r"lexical-binding:\s*t",
            "mathy.el is missing the `lexical-binding: t` file-local cookie",
        )

        # Well-formed + working: byte-compiles with no error...
        bc = self._sandbox_sh(
            "emacs --batch -Q -f batch-byte-compile mathy.el 2>&1; echo EXIT:$?")
        self.assertIn("EXIT:0", bc.stdout,
                      f"mathy.el did not byte-compile cleanly:\n{bc.stdout}")
        self.assertNotIn("Error", bc.stdout,
                         f"byte-compile reported an error:\n{bc.stdout}")

        # ...and computes the right answer.
        run = self._sandbox_sh(
            "emacs --batch -Q -l mathy.el --eval '(princ (mathy-factorial 5))'")
        self.assertIn(
            "120", run.stdout,
            f"(mathy-factorial 5) did not return 120:\n{run.stdout}\n{run.stderr}",
        )


class TestElispSkillAntiTrigger(TestCase):
    """No sandbox: only the load decision is under test."""

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_does_not_load_on_python_task(self):
        workspace = tempfile.mkdtemp(prefix="elisp_skill_anti_")
        self.addCleanup(shutil.rmtree, workspace, ignore_errors=True)
        agent = AgentHarness(create_agent(self.model, workspace))
        agent.message(
            "Write a Python function that returns the sum of a list of "
            "numbers, and add a quick test for it."
        )
        self.assertFalse(
            _skill_was_loaded(agent, "elisp"),
            "elisp skill loaded on a pure Python task — its trigger is too broad",
        )
