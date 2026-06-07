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

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model
from assist.sandbox_manager import SandboxManager, SANDBOX_IMAGE

from .utils import cleanup_workspace, executed_commands, skill_was_loaded


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
        cleanup_workspace(self.workspace)

    # Markers of an action that actually loads/compiles/checks a file, so
    # `target` appearing merely as a positional file argument (which Emacs does
    # not evaluate) doesn't count as verification.
    _VERIFY_ACTIONS = ("batch-byte-compile", "byte-compile-file", "checkdoc",
                       "ert-run-tests", "load-file", "-l ", "--load")

    def _verified_file_in_emacs(self, agent, target: str) -> bool:
        """True iff the agent ran an ``emacs --batch`` command that VERIFIES
        ``target`` — the file it was asked to write — via a real load/compile/
        check action (byte-compile / load / ERT / checkdoc), not some unrelated
        file, a bare positional visit, or a no-op like ``--version``. The exact
        form is the model's choice."""
        for cmd in executed_commands(agent):
            if not (re.search(r"\bemacs\b", cmd)
                    and ("--batch" in cmd or "-batch" in cmd)
                    and target in cmd):
                continue
            if any(action in cmd for action in self._VERIFY_ACTIONS):
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
            skill_was_loaded(agent, "elisp"),
            "agent did not load the elisp skill for an Emacs-Lisp authoring task",
        )
        self.assertTrue(
            self._verified_file_in_emacs(agent, "sum.el"),
            "elisp skill loaded but the agent never verified sum.el in "
            "emacs --batch (byte-compile / ERT / load) — it should not just "
            "emit elisp and trust it",
        )

    def _sandbox_sh(self, script: str) -> subprocess.CompletedProcess:
        """Run a shell snippet in a fresh sandbox container against the
        workspace, capturing output — used to independently verify the elisp
        the agent produced (not to drive the agent)."""
        return subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workspace}:/workspace',
             '-w', '/workspace', SANDBOX_IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=120)

    def test_writes_wellformed_elisp(self):
        """The skill's payoff: the agent produces elisp that is well-formed and
        actually works. We pin the file/function names, then INDEPENDENTLY
        byte-compile the artifact and run it — the eval verifies the output, not
        just that the agent ran something."""
        agent = AgentHarness(create_agent(
            self.model, self.workspace, sandbox_backend=self.sandbox))
        # No quality directions in the prompt (no "byte-compile cleanly", no
        # "add a Commentary section") — those must come from the SKILL. The
        # assertions below independently verify the artifact is well-formed and
        # working, so a pass means the skill induced that, not the prompt.
        agent.message(
            "Create an Emacs Lisp file `mathy.el` that provides a function "
            "`mathy-factorial` taking one non-negative integer N and returning "
            "N factorial — so (mathy-factorial 5) returns 120."
        )
        self.assertTrue(
            skill_was_loaded(agent, "elisp"),
            "agent did not load the elisp skill for an Emacs-Lisp authoring task",
        )
        path = os.path.join(self.workspace, "mathy.el")
        self.assertTrue(os.path.exists(path), "agent did not write mathy.el")

        # Well-formed: a real first-line `-*- lexical-binding: t; -*-` cookie
        # (not merely the string appearing somewhere like a Commentary line).
        with open(path, encoding="utf-8") as f:
            src = f.read()
        first_line = src.splitlines()[0] if src.splitlines() else ""
        self.assertRegex(
            first_line, r"-\*-.*lexical-binding:\s*t.*-\*-",
            f"mathy.el's first line lacks a valid `-*- lexical-binding: t; -*-` "
            f"file-local cookie; got: {first_line!r}",
        )

        # Well-formed + working: byte-compiles with no error...
        bc = self._sandbox_sh(
            "emacs --batch -Q --eval '(setq byte-compile-error-on-warn t)' "
            "-f batch-byte-compile mathy.el 2>&1; echo EXIT:$?")
        self.assertEqual(
            bc.returncode, 0,
            f"sandbox byte-compile run failed to launch (docker/mount issue?):\n"
            f"stdout:\n{bc.stdout}\nstderr:\n{bc.stderr}")
        self.assertIn(
            "EXIT:0", bc.stdout,
            f"mathy.el did not byte-compile cleanly (warnings count as "
            f"errors):\nstdout:\n{bc.stdout}\nstderr:\n{bc.stderr}")

        # ...and computes the right answer.
        run = self._sandbox_sh(
            "emacs --batch -Q -l mathy.el --eval '(princ (mathy-factorial 5))'")
        self.assertEqual(
            run.returncode, 0,
            f"loading/running mathy.el failed:\n{run.stdout}\n{run.stderr}",
        )
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
            skill_was_loaded(agent, "elisp"),
            "elisp skill loaded on a pure Python task — its trigger is too broad",
        )
