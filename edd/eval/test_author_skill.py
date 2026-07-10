"""Author-skill eval — does the agent write a well-formed, discoverable skill?

The `author-skill` built-in guides the agent to create a NEW skill: a
`.claude/skills/<name>/SKILL.md` in the working repository. It is a runtime
domain-skill authoring flow — the produced skill loads in a *later* chat (after
the user merges it to main), not this one, so the test measures the ARTIFACT,
not in-session self-use.

Each test asserts the same thing twice over:

1. The agent loaded the `author-skill` skill (progressive disclosure fired on
   the user's create-a-skill request).
2. The agent applied it — it wrote a SKILL.md that will actually load: the file
   opens with `---`-fenced YAML frontmatter (parsed with the SAME anchored regex
   the loader uses), `name` equals its folder, a non-empty description, and a
   one-line H1 body. A skill that does not parse (the classic `: ` colon-space
   trap, or leading content before the fence) is silently dropped by the loader,
   so this is the by-construction bar for "this skill would be discoverable".

The requested capability (writing a limerick from a topic) is deliberately NOT
one of the skill description's own EXAMPLES *and* does not overlap any existing
built-in skill, so a pass measures generalization, not lexical proximity or a
disambiguation confound.
"""
import glob
import os
import re
import shutil
import subprocess
import tempfile

from unittest import TestCase

import yaml

import assist
from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model

from .utils import create_filesystem, read_file, skill_was_loaded, stub_research_subagent

# The loader's own frontmatter extraction (deepagents skills middleware): the
# file MUST start with `---`, or it is dropped. Mirror it exactly so the test
# can't pass a skill the real loader would reject.
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

_BUILTIN_SKILLS_DIR = os.path.join(os.path.dirname(assist.__file__), "skills")


class TestAuthorSkill(TestCase):
    """Verifies the agent authors a well-formed, loadable skill on request."""

    def setUp(self):
        self.model = select_assistant_model(0.1)
        # Guard against the agent escaping the tempdir and writing into the real
        # built-in skills dir (a shell step in the wrong cwd did this once).
        self._skills_before = set(os.listdir(_BUILTIN_SKILLS_DIR))

    def tearDown(self):
        strays = set(os.listdir(_BUILTIN_SKILLS_DIR)) - self._skills_before
        for name in strays:
            p = os.path.join(_BUILTIN_SKILLS_DIR, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
        self.assertEqual(
            strays, set(),
            f"Agent escaped the tempdir and wrote {strays} into the real "
            f"built-in skills dir ({_BUILTIN_SKILLS_DIR}); cleaned up. The skill "
            f"must create the file with write_file in the working repo, not via "
            f"a shell command in the wrong working directory.",
        )

    def _make_agent(self):
        root = tempfile.mkdtemp()
        create_filesystem(root, {
            "README.md": "Helpers for my writing workflows.",
        })
        # A real domain repo is version-controlled, with an identity, so the
        # skill's commit step has somewhere to land instead of erroring mid-turn.
        # -b main: don't inherit the machine's init.defaultBranch (may be master);
        # the domain-repo world assumes main.
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True)
        return AgentHarness(create_agent(self.model, root)), root

    def _assert_valid_skill_authored(self, root: str):
        paths = glob.glob(os.path.join(root, ".claude", "skills", "*", "SKILL.md"))
        self.assertEqual(
            len(paths), 1,
            f"Expected exactly one authored .claude/skills/<name>/SKILL.md, "
            f"found {paths}. The author-skill body says the file goes there and "
            f"nowhere else.",
        )
        path = paths[0]
        folder = os.path.basename(os.path.dirname(path))
        raw = read_file(path)

        m = _FRONTMATTER_RE.match(raw)
        self.assertIsNotNone(
            m,
            "SKILL.md has no leading `---`-fenced YAML frontmatter block — the "
            "loader would drop it (leading content or a missing opening fence).",
        )
        try:
            meta = yaml.safe_load(m.group(1))
        except yaml.YAMLError as exc:  # the `: ` colon-space trap lands here
            self.fail(f"Frontmatter is not valid YAML (likely a colon-space in "
                      f"the description): {exc}")
        self.assertIsInstance(meta, dict, "Frontmatter did not parse to a mapping.")
        # The loader only WARNS on name!=folder, but a correct skill matches; the
        # author-skill body requires it, so hold the stronger bar here.
        self.assertEqual(
            meta.get("name"), folder,
            f"`name: {meta.get('name')!r}` should equal the folder name {folder!r}.",
        )
        self.assertTrue(
            (meta.get("description") or "").strip(),
            "Skill has no description — the loader silently skips it.",
        )

        body = raw[m.end():].lstrip()
        self.assertTrue(
            body.startswith("# "),
            f"Body must open with a one-line `# H1` title; got: {body[:60]!r}",
        )

    def test_explicit_skill_creation(self):
        """The user asks, in so many words, for a new skill."""
        agent, root = self._make_agent()

        with stub_research_subagent():
            agent.message(
                "Please create a new skill for me: whenever I give you a topic, "
                "write a limerick about it (the five-line AABBA form). Set it up "
                "so it's there for next time."
            )

        self.assertTrue(
            skill_was_loaded(agent, "author-skill"),
            "Agent did not load /skills/author-skill/SKILL.md despite an "
            "explicit request to create a new skill.",
        )
        self._assert_valid_skill_authored(root)
