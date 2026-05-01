---
name: calculate
description: Numeric computation in Python — arithmetic, statistics, projections, financial planning. TRIGGER WORDS — calculate, compute, sqrt, stdev, mean, median, variance, project, projection, projecting, compound interest, savings rate, percent return, contribution. MUST load before any prompt whose answer involves a number.
---

# Calculate — run-then-answer guide

## The rule

Every numeric answer must come from a Python computation you ran via `execute`. Never eyeball, never reason it out in your head — even for "easy" cases like a square root or a stdev of nine numbers. If you state a number you did not compute, the answer is wrong even when it happens to be right.

## When the answer depends on facts you don't know

Current interest rates, named investment strategies, lookup data, current statistics — delegate to `research-agent` via the `task` tool first. Wait for the result, then run your computation on the values it returns.

## Use the user's own numbers

When the prompt is about the user's own situation (their income, savings, goals, plans) and the workspace contains a relevant file with their numbers, READ that file before computing. A retirement projection that uses the user's actual income and savings rate is useful; one that runs on generic placeholders is not.

## How to run Python

Default to inline `-c`:

```
execute("python3 -c 'import math; print(math.sqrt(2_500_000))'")
```

Heredoc is fine for multi-line:

```
execute("python3 <<'PY'\nimport statistics\nxs = [1, 2, 3]\nprint(statistics.mean(xs))\nPY")
```

Standard library only, unless the user already has scientific deps installed. `math`, `statistics`, and basic float arithmetic cover almost every prompt this skill exists for.

## When to write a `.py` file

Don't, by default. One-off questions are inline. Write a file only when both are true:

1. The script has obvious reuse value (the user can re-run it later with different inputs), AND
2. The user gave you reason to keep it (mentioned saving it, asked for a script, or the work is part of a larger project).

When you do write one, save it under `scripts/`. Never at the workspace root.

## Cleanup rule

If you wrote a `.py` file and the user did not ask to keep it, you MUST delete it before responding. Run `execute("rm <path>")` in the same turn as the answer. This applies anywhere in the workspace tree — `calc.py`, `tmp/foo.py`, `scratch/x.py` all need to go.

If you used inline `-c` or heredoc, there is nothing to clean up.

## Worked example — single value

User: "What is the cube root of 4096?"

```
execute("python3 -c 'print(4096 ** (1/3))'")
```

Output: `15.999999999999998`. Respond: "The cube root of 4096 is 16."

## Worked example — financial projection

User: "What's the future value of $20,000 at 5% annual return for 10 years, compounded monthly?"

```
execute("python3 -c 'p=20000; r=0.05/12; n=12*10; print(p*(1+r)**n)'")
```

Output: `32940.20...`. Respond with the figure rounded sensibly: "$32,940.20."

## Anti-patterns

- Stating a number you did not compute ("by inspection", "approximately", "roughly").
- Writing `calc.py` at the workspace root and leaving it.
- Running a financial projection on generic numbers when the user's own numbers are in the workspace.
- Pulling in `numpy` / `pandas` / `scipy` for problems the standard library handles.
