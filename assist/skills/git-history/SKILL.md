---
name: git-history
description: Manage this thread's local git history and branch — view what changed each turn, and roll back / undo / reset the thread branch to an earlier commit, point in time, or described change; discard uncommitted edits. Use when the user wants to undo, roll back, revert recent changes, start over from an earlier point, see the history, or discard changes. Load before any git reset, checkout, restore, or history-navigating git log.
---

# Manage this thread's local history

Every turn is committed to this thread's branch, so the history is a per-turn log you can
navigate and roll back to. You operate ONLY on this thread's branch in `/workspace` —
never `main`, never push.

## See what changed and when
- Recent turns (each commit is one turn): `git -C /workspace log --oneline -20`
- A specific file's history: `git -C /workspace log --oneline -- <file>`
- Around a time: `git -C /workspace log --since="2 days ago" --oneline` (or `--until=`)

## Roll back to an earlier point (undo work)
Pick the commit to return to from the log, then:
- `git -C /workspace reset --hard <commit>`
This moves the thread branch back to that commit; everything after it is undone. (Not
truly lost — git keeps it in the reflog — but treat it as a deliberate rollback.)
- Undo just the last turn: `git -C /workspace reset --hard HEAD~1`
- "Start over from [when]": find the commit at or just before that time in the log, then
  reset to it.
- Before a reset that discards more than the last turn, confirm with the user which point
  they mean (show them the log line), then reset.

## Discard uncommitted edits (this turn's changes, not yet committed)
- One file: `git -C /workspace restore <file>` (or `git -C /workspace checkout -- <file>`)
- Everything uncommitted: `git -C /workspace restore .`

## Already-merged work is different
If the change was already merged into `main` (Merge & Push), you cannot roll the thread
branch back to undo it — it's shared history. To undo merged work, make a NEW change that
reverses it: `git -C /workspace revert <commit>`, then Merge & Push that. Or tell the user
to fix it from a real computer.

## HARD RULES (never put these in a command)
- only ever operate on this thread's branch in `/workspace`; never `git checkout main` or
  otherwise touch the `main` branch
- never `git push` — the user publishes via the Merge & Push button
- never `git reset --hard` past a point the user asked for, and confirm before discarding
  more than the last turn
