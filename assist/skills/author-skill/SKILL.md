---
name: author-skill
description: Author a new agent skill — write a well-formed SKILL.md so the assistant gains a reusable capability. EXAMPLES — "make a skill for converting between currencies"; "turn the way you just did that into a reusable skill"; "add a skill that captures our release checklist". MUST load before creating or scaffolding a new skill or a .claude/skills/ folder.
---

# Author a skill — write a well-formed SKILL.md

You are creating a NEW skill: a folder holding one `SKILL.md`, discovered from the working
repository's `.claude/skills/` directory. Create it with `write_file`, then help the user
publish it. Four phases.

## 1. WHERE
- `write_file` the file to `.claude/skills/<name>/SKILL.md` in the working repository. This
  one file IS the skill — do not edit any prompt or template to "register" it.
- `<name>` is lowercase letters, digits, and single hyphens (e.g. `convert-currency`); the
  `name:` field MUST equal the folder name.

## 2. FRONTMATTER
Two fields between `---` fences:

    ---
    name: convert-currency
    description: {one-line capability}. EXAMPLES — "…"; "…". MUST load before {trigger}.
    ---

- The description is the ONLY thing seen when deciding to load the skill — say WHEN to use it,
  not how, and keep the trigger specific so it does not fire on another skill's requests.
- Never put a colon-then-space (`: `) inside an unquoted description: YAML reads it as a new
  field and the skill is silently dropped. WRONG: `Convert money: dollars to euros`.
  RIGHT: `Convert money between currencies — dollars to euros and back`.

## 3. BODY
- The first line is a one-line `# H1` title. Then the concrete rules to follow when the skill
  applies — numbered steps, and wrong-vs-right for anything easy to get wrong. Do NOT add a
  "when to use" section; that is the description's job. The body is rules, not routing.
- Reference the tools this agent has — `read_file`, `write_file`, `edit_file`, `execute`,
  `ls` — never `Bash`, `Edit`, or `WebFetch`.

## 4. PUBLISH
- `read_file` the new file back and confirm the `---` fences, both fields, and the `# H1`
  first line are present.
- You do NOT run git. Your file changes are committed and the thread branch is pushed for you
  at the end of the turn.
- Tell the user that to start using the skill they press **Merge & Push** in the web UI —
  that lands it on `main`. You cannot push, and every new chat is cloned from `main`, so the
  skill appears only after Merge & Push, and only in a NEW chat (this conversation's skill
  list was fixed when it started).
