---
name: git-sync
description: Bring this thread's branch up to date with the latest main AND resolve git rebase conflicts — rebase onto main, working through any conflict markers. Use when the user asks to sync with main, get the latest, rebase onto main, update the branch, land recent changes, OR to resolve a merge/rebase conflict (files under Unmerged paths, a Merge-to-Main conflict banner). Load before running any git rebase.
---

# Sync this thread's branch with main

## main is already current here — do NOT fetch
- The host refreshes `origin/main` for you before your turn. `git fetch` does NOT work in
  this sandbox and is not needed — rebase directly onto the local `origin/main`.
- You do not push. The user pushes manually.

## Decision tree (run in order; do not improvise)
1. Run: `git -C /workspace rev-list --count HEAD..origin/main`
   - output is `0`  → main has NOT advanced; you are up to date. STOP.
   - output is `> 0` → main advanced. Go to step 2.
2. Run: `git -C /workspace rebase origin/main`
   - exit 0               → success. Report "synced". STOP.
   - output has `CONFLICT` → go to CONFLICT LOOP.
   - any other non-zero    → go to STUCK.

## CONFLICT LOOP (repeat until the rebase reports success)
a. Run: `git -C /workspace status` — read the `Unmerged paths` list.
b. For EACH unmerged file:
   - `read_file` it.
   - Resolve every `<<<<<<<` / `=======` / `>>>>>>>` block with `edit_file`; leave ZERO
     conflict markers.
   - Run: `git -C /workspace add <file>`
c. Run: `git -C /workspace rebase --continue`
   - exit 0                → success. Report what you resolved. STOP.
   - output has `CONFLICT` → a later commit also conflicts; go back to (a).
   - any other error       → go to STUCK.

## STUCK (the escape hatch — this is how you stay un-stuck)
If you cannot resolve the conflict, run: `git -C /workspace rebase --abort`
Your committed work is preserved on the branch. Tell the user exactly which files
conflicted and ask which side to keep. Do not guess.

## Discard and restart from main (only on explicit request)
ONLY if the user explicitly asks to throw away this thread's changes and start over from
the latest main: run `git -C /workspace reset --hard origin/main`. This resets the THREAD
BRANCH (never `main`) — the thread's work is gone. Never do this as part of a normal sync.

## HARD RULES (never put these in a command)
- never `git fetch` — it fails in this sandbox; `origin/main` is already current
- never `git push`, `push --force`, `send-pack` — the user pushes manually
- never `git rebase --skip` — it drops a commit (lost work)
- never `git checkout main` or otherwise touch the `main` branch
- never `git reset --hard` except the explicit discard-and-restart above
- only ever operate on `/workspace`
