---
name: git-sync
description: Bring the current thread branch up to date with main or resolve its merge conflict without rewriting published thread history. Use when the user asks to sync with main, get the latest, update the branch, or resolve a Merge & Push conflict. Do not rebase a thread branch.
---

# Sync this thread's branch with main

## Keep the published branch's ancestry
- Git-bound turn admission refreshes `origin/main`. A legacy Merge & Push
  conflict also follows a host fetch. Other legacy turns and resumed Git-bound
  turns may have a stale tracking ref. If freshness is unconfirmed, do not
  claim the branch is current. Do not fetch in this sandbox.
- Work only on the current thread branch in `/workspace`. Do not switch to `main`.
- You do not push. The Git-bound host publishes after successful finalization;
  legacy publication is best-effort and may need operator follow-up.
- If a rebase is already in progress, stop and report it. Do not continue or rewrite it.

## Merge the current main
1. Run `git -C /workspace status`. If any tracked or untracked work is dirty,
   stop and ask how to preserve it. If a merge is already in progress, stop and
   report it; do not continue a merge this skill did not start. A legacy
   Merge & Push banner means the host already aborted its rebase, not that a
   merge is in progress.
2. Only after confirming the tracking ref is fresh, run
   `git -C /workspace rev-list --count HEAD..origin/main`.
   - `0`: main has not advanced beyond this branch at that fetch. Report that
     no sync is needed relative to the fetched main tip.
   - Greater than `0`: run `git -C /workspace -c user.name=Assist -c user.email=assist@localhost -c commit.gpgsign=false merge --no-edit --no-gpg-sign origin/main`.
3. If the merge succeeds, report what changed. If it reports `CONFLICT`, use the
   conflict loop. For any other failure, stop and report the exact error.

## Conflict loop
1. Run `git -C /workspace status` to identify unmerged files.
2. Read each conflicted file. Resolve every conflict marker while preserving the
   user's intended changes; ask the user if the correct result is unclear.
3. Stage only the resolved files with `git -C /workspace add <file>`.
4. Run `git -C /workspace -c user.name=Assist -c user.email=assist@localhost -c commit.gpgsign=false -c core.editor=true merge --continue`. If another
   conflict remains, repeat. If continuation fails for another reason and your
   merge is still in progress, follow the abort rule below.
5. Report the resolution and that the branch still contains its earlier commits.

If you cannot resolve a conflict from the merge you started here, run
`git -C /workspace merge --abort` before the turn ends, then report that the
sync did not complete. The checkout was clean before this merge, so this only
discards your attempted merge resolution. If abort fails, report that the
checkout needs operator verification; do not stage conflict markers or claim
success. Never abort a merge that was already in progress when you arrived.

## Hard rules
- Never rebase, reset, or force-push this published thread branch. Those rewrite
  its authorized history and can strand future turns.
- Never fetch, push, checkout `main`, or discard work while syncing in this sandbox.
- Never use `git merge -s ours` to hide a conflict. Resolve the actual content.
