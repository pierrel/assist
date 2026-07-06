---
name: edit-files
description: Changing the contents of existing files — marking items done or cancelled, adding or removing entries, fixing or rewriting a passage, applying a batch of changes across a list, or any multi-step edit whose result you then summarize or count. EXAMPLES — "check off the tasks I finished on the sprint board and delete the ones we dropped"; "in packing.md tick everything I've packed and drop what I'm leaving behind"; "fix the broken link in the intro and tack on a closing line". MUST load before any tool call that edits, updates, rewrites, or removes content from a file the user named, and before you report what you changed.
---

# Editing files — find it, change it, verify it

You are about to change one or more files. Do all four phases. The last one is not optional.

## 1. WHERE — find the file and the exact spot

- `read_file` the target file FIRST, every time. The file on disk is the only source
  of truth — never edit, and never describe, from memory of what you think it says.
- Locate the exact section / heading / line your change belongs to in what you just read.

## 2. HOW — make the change

- Editing existing content: `edit_file`. Creating a brand-new file: `write_file`.
- Anchor `old_string` on ONE line that appears exactly once in the file, copied verbatim
  from what you just read — never a whole paragraph, never body text. (One unique line
  can't match the wrong place and can't split a section.)
- Inserting mid-file: anchor on the single line your new content goes right before, and
  put [new content] + that same line in `new_string`.
- If `edit_file` reports the string was not found, your anchor is wrong: re-read the file,
  copy a line that actually exists, and try again. Do NOT invent a different anchor from
  memory, and do NOT report the edit as done.
- For `.org` files the unique anchor is a heading line — load the `org-format` skill for
  the heading-body detail.

## 3. WHAT — do EVERY requested change

- First, list the requested changes as a checklist, one line each: the items to mark, the
  items to cancel, the things to add or remove, and any final step (a count, a summary line).
- Do each one. A bulk request ("mark A, B, C; cancel everything under P; add Q") is done
  only when every item on your checklist has its own edit. Skipping one because it "looked
  already done" is a failure — you will catch it in phase 4.

## 4. VERIFY — your report must come from the file, not from your plan

This is the rule, the same shape as `calculate`'s "never state a number you didn't compute":

> Every claim you make about what you changed — including any COUNT — must come from the
> file as it is NOW, read back this turn. A change you report, or a number you state, that
> the file doesn't actually show is WRONG even when you "remember" doing it or it happens
> to match.

Before you write your summary:

1. Run `execute("git -C /workspace diff")` and look at what actually changed.
2. RECONCILE against your phase-3 checklist. Every item must appear in the diff. If one
   doesn't, you SKIPPED it — go back and make that edit now. Do not report success without it.
3. If the summary or the edit will state a COUNT or a derived total ("12 active items left",
   a remaining count, a new total), that is a NUMBER — treat it exactly like the calculate
   skill: don't count by eye and don't carry it from your plan (you will be off by one).
   Run a command on the FINAL file and use its output — `grep -c`, `wc -l`, or a one-line
   `python3` count. A count you didn't run a command to get is wrong even when it matches.
4. Write the summary from the diff and the re-read file, line by line. If the diff is empty,
   say you changed nothing — never narrate edits a clean diff contradicts.

## Anti-patterns

- Reporting an edit (or a count) from your plan / your memory instead of the diff and the
  re-read file — the two disagree precisely when you skipped or half-did a step.
- Marking a checklist item "done" without its own `edit_file`.
- Inventing a new anchor from memory after a "string not found" error, or claiming that
  edit landed anyway.
- Writing a summary before running `git diff`.
