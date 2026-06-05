---
name: org-format
description: Editing or creating org-mode (`.org`) files without breaking heading structure. EXAMPLES — "add a new section to notes.org under the Q3 plans heading"; "tweak the second bullet under Inbox in todo.org". MUST load before any tool call that reads, edits, writes, or mentions a `.org` file.
---

# Org-mode format guide

When your response involves any `.org` file, end the response with a brief note about the heading-insertion rule (see *Inserting a new heading* below) so the caller knows how to edit the file safely.

## Headings

A heading is a line that starts (at column 0) with one or more asterisks **followed by a space**, then the title. The heading level is the number of asterisks:

```
* Top-level heading
** Second-level heading
*** Third-level heading
```

A line whose asterisks have **no space after them** is NOT a heading — it is *bold* body text, e.g. `*Direction.*` or `*Note.* ...`. An indented `* ` (not at column 0) is a list bullet, also not a heading.

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

## Inserting a new heading — anchor on ONE heading line

`edit_file` replaces `old_string` with `new_string`. To add a new heading WITHOUT splitting an existing section, **anchor `old_string` on a single heading line — never on body text, and never on a multi-line block.**

A heading line is asterisks **followed by a space**, then a title: `* Foo`, `** Bar`. These are NOT heading lines — never anchor on them:
- a plain sentence / body text (e.g. `The endpoint is rate-limited.`)
- an org *bold* line: `*Direction.*`, `*Note.* ...` — the asterisks have **no space** after them, so it is body text, not a heading.

If `old_string` is anything other than a real `* `/`** ` heading line, your new heading lands in the middle of a section and splits it.

Procedure:

1. **`read_file` the file first**, so you can copy a heading line that actually exists in it. Do not type an anchor from memory.
2. Pick the existing heading your new heading should go immediately **before** — the next heading at the same level or a higher level (equal-or-fewer asterisks) that should follow your new one.
3. Set `old_string` to **exactly that one heading line, copied verbatim from what you just read** — no body lines, nothing else.
4. Set `new_string` to your new heading and its body, then that same heading line, unchanged.

Because `old_string` is a single heading line, you never capture (and so never split) an existing section's body.

Example — add `* New section` before the existing `* Goals` heading.
(Both strings are flush left — headings live at column 0, no indentation.)

`old_string` is exactly:

```
* Goals
```

`new_string` is:

```
* New section
New section's body.

* Goals
```

The same rule applies to a sub-heading (a new `**`): anchor on the next `**`-or-shallower heading line and insert before it.

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
