---
name: git-conflict
description: Resolving a paused git rebase conflict left by a failed merge. EXAMPLES — "fix the merge conflict the rebase left"; "the merge failed on a conflict, please resolve it"; "git status shows unmerged files, work through them". MUST load before any tool call that touches git rebase state, conflict markers, or unmerged files.
---

# Git rebase conflict — fixed loop

The web UI's "Merge to Main" runs `git rebase origin/main` on this
thread's branch. When the rebase hits a conflict, it aborts and the
working tree returns to the thread branch in a clean state. The user
will type a follow-up message asking you to fix the conflict.

Your job: edit the affected files so the underlying disagreement is
resolved, then tell the user to re-click **Merge to Main**. The web
UI will re-run the rebase, which will succeed this time.

## You cannot push

The web app blocks `git push` from your `execute` tool. Don't try
push-related commands — they will fail with an explicit error message.
The only way work reaches `origin` is the user clicking **Push to
origin** in their browser.

## The fixed loop

For each conflict you're asked to resolve, run these steps in order.
Do **not** improvise — the order matters.

### 1. See what the merge attempt found

```
execute('git -C <repo> status')
```

You'll see a list under `Unmerged paths` and possibly `Both modified`
or `Both added`. Those are the files that need work.

### 2. Read each unmerged file

```
read_file('<repo>/<unmerged-file>')
```

Inside, the file has standard conflict markers:

```
<<<<<<< HEAD
... (the version that was already on origin/main)
=======
... (your branch's version)
>>>>>>> <commit hash from your branch>
```

### 3. Resolve the markers

Use `edit_file` (or `write_file` if the rewrite is complete) to replace
the conflict block with the correct merged content. The "correct" content
depends on the work — sometimes you take the `HEAD` side, sometimes
yours, sometimes both, sometimes neither plus a fresh hand-written line.
Read both sides and decide.

Remove **all three markers** (`<<<<<<<`, `=======`, `>>>>>>>`). If any
remain, git will treat the file as still unmerged.

### 4. Mark the file resolved

```
execute('git -C <repo> add <unmerged-file>')
```

This stages the resolved version.

### 5. Confirm everything is staged

```
execute('git -C <repo> status')
```

The file should now appear under `Changes to be committed` (not
`Unmerged paths`).

### 6. Tell the user to re-click Merge to Main

Don't try to run `git rebase --continue` yourself — the rebase already
aborted, and there's nothing in-flight for `--continue` to apply to.
The web UI will re-run the rebase from scratch when the user clicks
**Merge to Main** again. Your job is to leave the thread branch in a
state where that next rebase succeeds.

A short message like *"I resolved the conflict in `<file>` by `<one
line of what you did>`. Re-click Merge to Main and the rebase should
succeed."* is enough.

## Anti-patterns

- **`git rebase --abort` / `git reset --hard`** — these throw away
  work. The rebase is already aborted; anything you "abort" past that
  is the work the user wanted you to keep. Never use them.
- **`git rebase --skip`** — silently drops the commit that conflicted.
  Same problem: the user's work disappears.
- **`git push` (any form)** — blocked by middleware; it will return
  an error you can't bypass. Don't try.
- **Leaving conflict markers in the file.** Even one stray
  `=======` line means the file is still broken, the next rebase will
  fail again, and the user comes back to ask why nothing changed.
- **Editing files outside `Unmerged paths`.** Stay focused on the
  files git flagged. A drive-by refactor mid-conflict-resolution
  expands the diff and makes the next merge attempt riskier.

## When the right answer is "ask the user"

If both sides of the conflict represent legitimate changes you can't
reconcile (e.g. the `HEAD` side renamed a function and your branch
modified its body), don't guess. Tell the user what you see in the
conflict markers and ask which side they want kept. They can answer in
the next turn and you continue from step 3.
