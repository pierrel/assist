"""Small-model-friendly skills middleware.

Subclasses deepagents' SkillsMiddleware to override the system-prompt template
with concrete, imperative instructions. The default template is verbose,
includes abstract praise, and uses a fictional quantum-computing example —
fine for strong models, but the small models we run (e.g. Qwen3-Coder-30B)
miss the actionable instruction underneath the fluff.

Our template:
- States the mechanism in concrete tool terms ("call read_file with the path").
- Tells the model the descriptions don't contain the rules — the SKILL.md does.
- Removes fictional examples that don't carry weight for a small model.
"""
from deepagents.middleware.skills import SkillsMiddleware


SMALL_MODEL_SKILLS_PROMPT = """

## Skills

You have access to a list of skills. Each skill is a markdown file at the path shown below. The skill's *description* tells you when to load it; the skill's *content* (in the linked `SKILL.md`) contains the actual rules.

{skills_locations}

**Available skills:**

{skills_list}

### How to use a skill

1. **Match.** Look at the descriptions above. If one matches what you're about to do, continue to step 2.
2. **Load.** Call the `read_file` tool with the skill's `SKILL.md` path. The path is shown above next to each description (the line that begins with `Read`).
3. **Apply.** Use the rules from the loaded `SKILL.md` when you compose your action or response.

The descriptions only summarize *when* to load — they do not contain the rules. You will not know the rules until you complete step 2. You MUST complete step 2 before performing the matching action; relying on the description alone leads to incorrect outcomes.
"""


class SmallModelSkillsMiddleware(SkillsMiddleware):
    """SkillsMiddleware variant with imperative instructions for small models."""

    def __init__(self, *, backend, sources):
        super().__init__(backend=backend, sources=sources)
        self.system_prompt_template = SMALL_MODEL_SKILLS_PROMPT
