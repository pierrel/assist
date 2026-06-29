---
name: time
description: Dates and times via the `date` command — what's today, the day of the week of a date, relative dates, and date countdowns. EXAMPLES — "what's today's date"; "what day of the week is 7/5"; "what's the date next Thursday"; "how many days until Christmas". MUST load before any question about the current date or time, a day of the week, a relative date, or how many days until/since a date.
---

# Time — run-then-answer guide for dates

## The rule

Every date/time answer must come from a `date` command you ran via `execute`.
Never figure out today's date, a day of the week, or a future/past date in your
head — you do **not** know what today is without running `date`, and calendar math
("7/5 is a Saturday") is exactly what you get wrong. If you state a day or date you
did not get from `date`, the answer is wrong even when it happens to be right.

## How to run it

Today:

```
execute("date '+%A, %B %-d, %Y'")
```

A specific or relative date (GNU `date -d` does the parsing):

```
execute("date -d 'next Thursday' '+%A, %B %-d, %Y'")
```

`date -d` understands a lot: `next Thursday`, `last friday`, `tomorrow`,
`yesterday`, weekday names, `3 days`, `2 weeks ago`, `1 month`, and absolute dates
like `7/5`, `July 5`, `2026-12-25`.

## Translate the user's phrasing into a `date -d` expression

The user's words aren't always what `date -d` wants — convert them:

- "in 3 days" → `3 days`; "in two weeks" → `2 weeks`; "3 days ago" → `3 days ago`.
- "what day is 7/5" → `date -d '7/5'` (US month/day — see below).

`date -d` does NOT understand "X from <weekday>" ("two weeks from Friday"), "first
Monday of July", or ordinal phrases. If you cannot express the request as a single
`date -d` string, tell the user you can't compute that particular one — do **not**
invent a date.

## US dates and reading back

`7/5` means **July 5** (month/day). Run it through `date` and **echo the full
resolved date** so the user can catch a misread:

```
execute("date -d '7/5' '+%A (%B %-d, %Y)'")
```

## Day of the week

```
execute("date -d '7/5' '+%A'")
```

## Countdown — days until / since a date

The `+ 43200` rounds to the nearest whole day, which absorbs the partial current
day and any daylight-saving hour in the span (so the count isn't off by one):

```
execute("echo $(( ($(date -d '2026-12-25' +%s) - $(date +%s) + 43200) / 86400 )) days")
```

A positive number is days **until** the date; negative means it already passed.

## Worked examples

- "What's today's date?" → `execute("date '+%A, %B %-d, %Y'")` → output
  `Monday, June 29, 2026` → respond with exactly that.
- "What day of the week is 7/5?" →
  `execute("date -d '7/5' '+%A (%B %-d, %Y)'")` → `Sunday (July 5, 2026)` →
  respond "July 5 is a Sunday."
- "What's the date a week from Friday... I mean, next Thursday?" →
  `execute("date -d 'next Thursday' '+%A, %B %-d, %Y'")` → respond with the date.
- "How many days until Christmas?" →
  `execute("echo $(( ($(date -d 'Dec 25' +%s) - $(date +%s) + 43200) / 86400 )) days")`.

## Anti-patterns

- Stating a day of the week or a date you did **not** get from `date`.
- Guessing today's date — you can't know it without running `date`.
- Doing calendar math in your head (counting days, naming a weekday).
- Inventing a date for a phrasing `date -d` can't parse — say you can't compute it.
