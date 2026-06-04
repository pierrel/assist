---
name: org-format
description: Editing or creating org-mode (`.org`) files without breaking heading structure. EXAMPLES — "add a new section to notes.org under the Q3 plans heading"; "tweak the second bullet under Inbox in todo.org". MUST load before any tool call that reads, edits, writes, or mentions a `.org` file.
---

# Org-mode format guide

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

### WRONG — anchored on a mid-body line (the real failure)

Sections often have several paragraphs separated by blank lines, and an org *bold* line like `*Direction.*` starts with `*` and can look like a heading. Anchoring your edit on a mid-body line — or just before such a bold line — drops the new heading inside the section and orphans the rest:

```
* Research notes
The endpoint is rate-limited and behind anti-bot — stays.

* Moonshot                       ← WRONG: dropped mid-section
Moonshot's body.
*Direction.*  Pick a real API.   ← orphaned: this still belongs to Research notes
Keep the scraped endpoint for now.
* Next section
```

`*Direction.*` is *bold text*, NOT a heading — a heading needs a space after the asterisks (`* Direction`). Never anchor your edit on a bold line, a body sentence, or a blank line.

### Procedure for inserting a new heading

Do this literally, in order:

1. Decide which heading the new one goes after (or under).
2. From there, scan DOWN past everything beneath it — every body line, every blank line, every deeper `**`/`***` subsection — until you reach the next line that is a real heading at the same or a higher level: a line that begins with asterisks then a SPACE (`* ` or `** `).
3. That heading line is your edit anchor: set your `edit_file` `old_string` to it and put your new heading immediately before it. If there is no such next heading, append at the very end of the file.
4. NEVER anchor on a body sentence, a blank line, or an org *bold* line like `*Direction.*` (bold has no space after the `*`, so it is NOT a heading). Anchoring on any of those drops your new heading mid-section and splits it.

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
