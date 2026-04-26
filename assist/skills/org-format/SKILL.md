---
name: org-format
description: Format and editing rules for org-mode (.org) files. Load before reading, editing, or surfacing any .org file. Covers heading levels, the heading-body relationship, and the procedure for inserting a new heading without orphaning the previous heading's body content.
---

# Org-mode format guide

## When to apply

You MUST follow this guide whenever you:

- Read or refer to a `.org` file.
- Add, edit, or append content to a `.org` file.
- Surface a `.org` file path in a response to the user.

When your response involves any `.org` file, end the response with a brief note about the heading-insertion rule (see *Inserting a new heading* below) so the caller knows how to edit the file safely.

## Headings

Lines starting with one or more asterisks are headings. The heading level is the number of asterisks:

```
* Top-level heading
** Second-level heading
*** Third-level heading
```

## Heading body — the critical rule

**A heading's body is everything from that heading down to the next heading at the same level or a higher level** (i.e. equal-or-fewer asterisks). Deeper-level subheadings are part of the parent heading's body.

```
* Heading 1
This is heading 1's body.

More body content for heading 1.
** Subsection of heading 1
This subsection is also part of heading 1's body.
* Heading 2
This is heading 2's body, NOT heading 1's.
```

## Inserting a new heading — the most common mistake

To add a new heading at a given level (for example a new `*` sibling of an existing `*` heading), insert it AFTER the preceding heading's full body — that means **immediately before the next heading at the same level or a higher level**, NOT immediately after the preceding heading's first line.

### WRONG — inserts immediately after the preceding heading

This separates `* Heading 1` from its body, orphaning the body so it appears to belong to `* New heading`:

```
* Heading 1
* New heading                    ← WRONG: inserted right after the preceding heading
New heading's body.
This is heading 1's body.        ← orphaned, now reads as if part of "New heading"

More body content for heading 1.
** Subsection of heading 1
This subsection is also part of heading 1's body.
* Heading 2
```

### RIGHT — inserts after the preceding heading's full body

```
* Heading 1
This is heading 1's body.

More body content for heading 1.
** Subsection of heading 1
This subsection is also part of heading 1's body.
* New heading                    ← RIGHT: after heading 1's full body, before * Heading 2
New heading's body.
* Heading 2
```

### Procedure for inserting a new heading

1. Identify the heading you're inserting after.
2. Scan downward through its body — including any deeper-level subsections.
3. Stop at the next line that is a heading at the SAME level or a HIGHER level (equal-or-fewer asterisks).
4. Insert your new heading on the line immediately before that next heading.
5. If no such next heading exists (the heading you're appending under is the last one at its level in the file), append at the end of the file.

### Inserting a sub-heading

To add a deeper-level heading (for example a new `**` under a `*`), the rule applies recursively:

- Place the new sub-heading inside the parent heading's body.
- If the parent already has same-level sub-headings, insert before the next one of equal-or-fewer asterisks.

## Bullets and lists

- Unordered lists use `-` or `+` followed by a space. Indent (typically 2 spaces) to nest.
- Ordered lists use `1.`, `2.`, `3.` etc., with the same indentation rule for nesting.

## Emphasis

- `*bold*`
- `/italic/`
- `_underline_`
- `+strike-through+`
- `=verbatim=` and `~code~`

## Links

- Plain URLs are auto-recognized: `https://orgmode.org`
- Labeled links use `[[URL][Label]]` syntax.

## Notes

- Leave a blank line between major blocks for clarity.
- When in doubt, read the file first to understand the existing structure, then apply the insertion procedure above to find the correct line to insert before.
