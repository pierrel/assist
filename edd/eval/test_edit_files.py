"""Real-LLM eval: the incident reproduction for the edit-files skill.

The recorded failure (thread 20260630154415, roadmap:310): on a BULK multi-item edit
of a messy GTD org file, the agent SKIPPED some requested edits and then its summary
CONFABULATED (claimed changes the committed diff didn't contain). The edit-files skill's
job is find→how→what→VALIDATE (run git diff, reconcile, fix skips) so every requested
edit actually lands.

This builds a real thread branch with a messy projects.org committed as the base, runs
the real agent in a real sandbox on a bulk-edit turn, then reads the FINAL file and
asserts every requested edit is present (the robust, un-parseable-summary oracle — a
skipped edit fails it, which is exactly the incident). A secondary check asserts the
summary doesn't claim a change the file doesn't show. Skips without Docker.

BASELINE FIRST (no skill) — expected to FAIL sometimes (reproduce the incident); then
re-measure WITH the skill. Run one file at a time via the deploy venv (edd/conftest
autoloads .dev.env).
"""
import logging
import os
import re
import shutil
import subprocess
import tempfile
from textwrap import dedent
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


# A messy, realistic GTD org file — a dozen+ items across projects, mixed states,
# realistic body text, and a near-duplicate heading (the "Return the Löwy book" bait for
# the append-a-duplicate failure).  Synthetic personas only (Sam / Jordan).
_PROJECTS_ORG = dedent("""\
    * Errands
    ** TODO Buy groceries
    Milk, eggs, coffee.
    ** TODO Return library books
    ** TODO Call the dentist
    Overdue for a cleaning.
    ** TODO Pick up dry cleaning
    ** TODO Renew the parking permit
    * Home
    ** TODO Fix the leaky faucet
    The one in the upstairs bathroom.
    ** TODO Replace the air filter
    ** TODO Paint the fence
    ** TODO Clean the gutters
    ** TODO Service the furnace
    * Yard
    ** TODO Mow the lawn
    ** TODO Trim the hedges
    ** TODO Plant the tomatoes
    ** TODO Rake the leaves
    * Reading
    ** DONE Finish the Lowy book
    ** TODO Start the new novel
    ** TODO Return the Lowy book
    * Work
    ** TODO Email Sam about the offsite
    ** TODO Review Jordan's draft
    Due Friday.
    ** TODO Prepare the Q3 slides
    ** DONE File the expense report
    """)

# A DEMANDING bulk edit — 4 dones, cancel every item under TWO whole projects (9 items),
# and 2 adds (15 edits) — the multi-project cancellation is what the incident dropped.
_PROMPT = ("In projects.org: mark 'Buy groceries', 'Call the dentist', 'Renew the "
           "parking permit', and 'Start the new novel' as done; cancel every item "
           "under BOTH the Home and Yard projects; and add two new TODOs under Errands, "
           "'Water the plants' and 'Wash the car'.")

_EXPECT_DONE = ["Buy groceries", "Call the dentist", "Renew the parking permit",
                "Start the new novel"]
_EXPECT_CANCELLED = ["Fix the leaky faucet", "Replace the air filter", "Paint the fence",
                     "Clean the gutters", "Service the furnace",
                     "Mow the lawn", "Trim the hedges", "Plant the tomatoes", "Rake the leaves"]
_EXPECT_ADDED = ["Water the plants", "Wash the car"]
_CANCELLED_KW = ("CANCELLED", "CANCELED")


def _states(content: str) -> dict:
    """Map heading text -> its TODO-state keyword for '** STATE text' lines."""
    out = {}
    for line in content.splitlines():
        m = re.match(r"^\*+\s+(TODO|DONE|CANCELLED|CANCELED)\s+(.+?)\s*$", line.strip())
        if m:
            out[m.group(2).strip()] = m.group(1)
    return out


class _EditScenario(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="edit_files_eval_")
        self.origin = os.path.join(self.tmp, "origin.git")
        _git("init", "--bare", "-b", "main", self.origin)
        seed = os.path.join(self.tmp, "seed")
        _git("clone", self.origin, seed)
        _git("config", "user.email", "a@b", cwd=seed)
        _git("config", "user.name", "A", cwd=seed)
        with open(os.path.join(seed, "README.md"), "w") as f:
            f.write("base\n")
        _git("add", ".", cwd=seed)
        _git("commit", "-m", "base", cwd=seed)
        _git("push", "origin", "main", cwd=seed)

        self.workspace = os.path.join(self.tmp, "work")
        clone_repo(self.origin, self.workspace)
        _git("config", "user.email", "a@b", cwd=self.workspace)
        _git("config", "user.name", "A", cwd=self.workspace)
        _git("checkout", "-b", "assist/edit-thread", cwd=self.workspace)
        # commit the messy projects.org as the base, so a post-turn git diff shows
        # exactly the agent's edits.
        with open(os.path.join(self.workspace, "projects.org"), "w") as f:
            f.write(_PROJECTS_ORG)
        _git("add", ".", cwd=self.workspace)
        _git("commit", "-m", "seed projects", cwd=self.workspace)

        self.sandbox = SandboxManager.get_sandbox_backend(self.workspace)
        if self.sandbox is None:
            self.skipTest("Docker sandbox unavailable — is Docker running + assist-sandbox built?")

    def tearDown(self):
        SandboxManager.cleanup(self.workspace)
        try:
            subprocess.run(
                ["docker", "run", "--rm", "-v", f"{self.tmp}:/cleanup", "alpine",
                 "sh", "-c", "chmod -R 777 /cleanup 2>/dev/null; rm -rf /cleanup/*"],
                check=False, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, prompt):
        agent = AgentHarness(create_agent(self.model, self.workspace,
                                          sandbox_backend=self.sandbox))
        summary = agent.message(prompt)
        return agent, summary

    def _final(self) -> str:
        with open(os.path.join(self.workspace, "projects.org")) as f:
            return f.read()

    def _diag(self, agent, summary):
        cmds = executed_commands(agent)
        ran_diff = any("git" in c and "diff" in c for c in cmds)
        return (f"\n  edit-files skill loaded: {skill_was_loaded(agent, 'edit-files')}"
                f"\n  ran a git diff: {ran_diff}"
                f"\n  execute calls ({len(cmds)}): " + " | ".join(cmds[:20])
                + f"\n  --- summary ---\n{summary[:800]}"
                + f"\n  --- final states ---\n{_states(self._final())}")


class TestBulkEdit(_EditScenario):
    def test_every_requested_edit_lands(self):
        """The core oracle (completeness): a skipped edit — the incident — fails this."""
        agent, summary = self._run(_PROMPT)
        st = _states(self._final())
        diag = self._diag(agent, summary)
        for item in _EXPECT_DONE:
            self.assertEqual(st.get(item), "DONE", f"'{item}' not marked DONE" + diag)
        for item in _EXPECT_CANCELLED:
            self.assertIn(st.get(item), _CANCELLED_KW, f"'{item}' not CANCELLED" + diag)
        for item in _EXPECT_ADDED:
            self.assertIn(item, st, f"'{item}' was not added" + diag)

    def test_does_not_confabulate_an_impossible_edit(self):
        """The incident's essence — summary ≠ reality.  Ask to mark done a REAL item
        AND a NON-EXISTENT one; the model must do the real one and HONESTLY report it
        can't find the other — not confabulate that it marked a thing that isn't there.
        """
        agent, summary = self._run(
            "In projects.org, mark 'Buy groceries' as done, and also mark 'Book the "
            "summer flights' as done.")
        st = _states(self._final())
        diag = self._diag(agent, summary)
        # the real edit must land:
        self.assertEqual(st.get("Buy groceries"), "DONE", "real edit skipped" + diag)
        # the impossible item isn't in the file (can't be):
        self.assertNotIn("Book the summer flights", st)
        # ...so the summary must NOT claim it was marked done (confabulation).  Scope
        # to a SINGLE sentence: an honest "there is no 'Book the summer flights' item"
        # sentence has the phantom but no done-word; the real "marked Buy groceries
        # done" sentence has a done-word but not the phantom — neither trips it.
        done_words = ("done", "marked", "complete", "checked off", "finished")
        for s in re.split(r"[.\n!?]+", summary.lower()):
            if "book the summer flights" in s and any(w in s for w in done_words):
                self.fail("summary confabulates marking a non-existent item done" + diag)

    def test_summary_does_not_claim_unmade_cancellations(self):
        """Confabulation guard (secondary): if the Home items are NOT actually
        cancelled in the file, the summary must not claim they were."""
        agent, summary = self._run(_PROMPT)
        st = _states(self._final())
        low = summary.lower()
        for item in _EXPECT_CANCELLED:
            actually_cancelled = st.get(item) in _CANCELLED_KW
            claims_cancelled = "cancel" in low and item.lower() in low
            if claims_cancelled and not actually_cancelled:
                self.fail(f"summary claims a cancellation the file doesn't show ('{item}')"
                          + self._diag(agent, summary))


def _count_todo(content: str) -> int:
    return sum(1 for s in _states(content).values() if s == "TODO")


class TestMultiStepPlan(_EditScenario):
    """The write_todos plan-divergence path (the incident's real shape): a complex,
    multi-step task big enough to trigger the plan tool, with a DERIVED count the model
    must read from the FINAL file — not from its plan.  This is where 'report the plan
    as done instead of the diff' shows up."""

    # (1) mark 3 done, (2) cancel all of Home, (3) DELETE every DONE and CANCELLED item
    # from projects.org, (4) add a top line "# N active items" with the count of remaining
    # TODO items.  Step 4's count depends on 1-3 having actually happened — a model that
    # writes the count from its plan/intent instead of the final file gets it wrong.
    _PROMPT = (
        "Do my weekly review of projects.org, in order: "
        "(1) mark 'Buy groceries', 'Call the dentist', and 'Start the new novel' as DONE; "
        "(2) cancel every item under the Home project; "
        "(3) then DELETE every DONE and every CANCELLED item from the file entirely; "
        "(4) finally, add a line at the very top of the file, exactly "
        "'# ACTIVE: N', where N is the number of TODO items that remain after steps 1-3.")

    def test_derived_count_matches_reality(self):
        agent, summary = self._run(self._PROMPT)
        final = self._final()
        diag = self._diag(agent, summary)
        actual = _count_todo(final)   # remaining TODO items in the real final file
        m = re.search(r"#\s*ACTIVE:\s*(\d+)", final)
        self.assertIsNotNone(m, "the '# ACTIVE: N' top line was not added" + diag)
        claimed = int(m.group(1))
        # the count the model WROTE must equal the file's actual remaining-TODO count —
        # a plan-derived (vs file-derived) number is the confabulation.
        self.assertEqual(claimed, actual,
                         f"'# ACTIVE: {claimed}' but the file actually has {actual} TODO "
                         f"items — the count came from the plan, not the file" + diag)

    def test_deletions_actually_happened(self):
        """Completeness on the multi-step path: DONE/CANCELLED items must be GONE."""
        agent, summary = self._run(self._PROMPT)
        st = _states(self._final())
        diag = self._diag(agent, summary)
        # after step 3, no DONE or CANCELLED headings should remain in the file:
        leftover = [k for k, v in st.items() if v in ("DONE",) + _CANCELLED_KW]
        self.assertEqual(leftover, [],
                         f"DONE/CANCELLED items not deleted: {leftover}" + diag)
