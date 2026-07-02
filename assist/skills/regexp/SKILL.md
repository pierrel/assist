---
name: regexp
description: Write and verify a correct Python regular expression from a plain-language matching rule — anchoring, escaping, character classes, alternation — and test it against examples before relying on it. Load when a task needs a non-trivial regexp (matching phone numbers or senders, filtering strings, extracting a pattern) and getting it slightly wrong would silently match the wrong things.
---

# Writing a correct regexp

A regexp that's subtly wrong fails silently — it matches nothing, or the wrong things, with
no error. So build it deliberately and **test it against concrete examples** before you rely
on it.

## Build it
- **Escape literals.** `.`, `+`, `?`, `*`, `(`, `)`, `[`, `]`, `{`, `}`, `|`, `\`, `^`, `$`
  are special. A literal `+` (as in a phone number) is `\+`; a literal `.` is `\.`.
- **Anchor when you mean the whole string.** `re.search` finds the pattern *anywhere*.
  `^…$` pins start and end; `^…` pins only the start (a prefix). "Numbers starting +1555" →
  `^\+1555`. "Exactly this number" → `^\+15551234567$`.
- **Character classes / ranges.** `\d` a digit, `\d{10}` ten digits, `[0-9]`, `[A-Za-z]`.
- **Alternation** for a fixed set: `^(\+15551234567|\+15559876543)$` matches either number.
- **Everything**: `.*`.

## Test it before relying on it
Run a quick check against strings that SHOULD and should NOT match — don't eyeball it:
```
python3 -c "import re; p=r'^\+1555'; \
print([(s, bool(re.search(p,s))) for s in ['+15551234567','+16505550000','1555']])"
```
Confirm the shoulds are `True` and the should-nots are `False`. Adjust until they are.

## Common traps
- Forgetting to escape `+`/`.` (so `+1555` matches far more than intended).
- Using `^…$` when you wanted a prefix (or vice versa).
- Assuming `re.search` anchors — it doesn't; add `^`/`$` yourself.

State the final pattern plainly and note what it matches (and what it deliberately doesn't).
