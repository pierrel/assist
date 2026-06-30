---
name: schedule
description: Set up, change, pause, or cancel recurring scheduled prompts for this conversation — e.g. "every morning at 7 review my inbox", "weekly weather", "when does it next run", "stop the scheduled task", "cancel the cron", "change it to fire at 5am". Load whenever the user wants something to run automatically on a timer/recurring basis in this thread.
---

# Scheduling recurring prompts

This thread can run a prompt automatically on a recurring schedule. The schedule
belongs to **this** thread and each run happens **in this thread**, so results
accumulate here. Use the schedule tools — do not try to keep time yourself.

## Tools
- `create_schedule(prompt, ...)` — start a recurring prompt.
- `list_schedules()` — show this thread's schedules with their ids and next run.
- `modify_schedule(schedule_id, ...)` — change an existing one.
- `pause_schedule(id)` / `resume_schedule(id)` — stop/restart without deleting.
- `delete_schedule(id)` — remove one permanently.

## Cadence — pick ONE shape and fill its fields
Times are in the user's local timezone. `weekdays` is `0=Mon … 6=Sun`.
- daily at a time → `hour=7, minute=0`
- specific weekdays → `hour=7, minute=0, weekdays=[0,1,2,3,4]`
- hourly at a given minute → `minute=30` (omit `hour`)
- every N minutes → `every_n_minutes=30` (omit `hour`)

Resolve vague wording to concrete fields ("morning" → a specific hour; "weekly" →
a specific weekday) — ask only if you truly can't infer it. If the user asks for a
pattern these fields can't express (e.g. "every other Tuesday", "last day of the
month"), say so plainly instead of approximating.

## Changing a schedule is a RELATIVE edit
To change one, pass **only the field(s) being changed plus the id** — everything
else stays as it is. "Change it to fire at 5am" → `modify_schedule(id, hour=5)`
(the minute and days are untouched). If you don't know the id, call
`list_schedules()` first. `modify` shifts or narrows existing fields; to **broaden**
a schedule (specific weekdays → every day, a fixed hour → hourly) or switch between a
clock schedule and an every-N-minutes one, **delete it and create a new one**.

## After any change
Relay back the schedule's cadence and **next run time** exactly as the tool
returns them, so the user can catch a misread (e.g. "Done — every day at 7:00 AM,
next run Mon Jun 15, 7:00 AM"). A thread can have at most 5 schedules. If a tool
returns an error or "couldn't", tell the user — don't claim it worked.
