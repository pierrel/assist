---
name: orchestrate-repeated-work
description: Use when the user wants the same source-backed check for every named item in a list, then one combined report or file update. Examples include checking each company's careers page before updating one job pipeline, verifying vendor contracts into one review, comparing locations in one accessibility report, reconciling accounts, or reviewing files into one summary.
---

# Orchestrate repeated work

## Launch

Plan the groups from the user's request. Then make the next tool call only the
planned evidence-only `delegate-agent`
`start_async_task` calls. Do not start context, research, or other tasks first.

1. Treat every named entity receiving the same substantive operation as one
   repeated workload, not an outcome per entity.
2. Partition the entities into disjoint groups of at most five. Use as few
   groups as practical and start at most eight in one wave.
3. Start exactly one evidence-only delegate per group. Do not inspect inputs or
   edit shared files on the launch turn.
4. Give each delegate its entities, authoritative inputs or URLs, the one
   operation, and a bounded per-entity return format: observed facts, source
   URL, timestamp, and unknowns. Explicitly forbid shared-file edits and final
   synthesis. Name only that group's entities in its brief, including caveats or
   questions; never mention another group or the full collection. Stop at the
   authoritative source; do not crawl, retry, or infer.

## Complete

1. On a completion wake, check only its planned group. For more than eight
   groups, start the next wave only after checking the current wave.
2. After every planned group succeeds and is checked, only the main agent
   updates shared artifacts and writes the requested synthesis. Copy only facts
   from the checked evidence; preserve unknowns and never infer or replace a
   reported role, source, or status. If a group fails or times out, leave shared
   artifacts unchanged and report its entities as incomplete.
