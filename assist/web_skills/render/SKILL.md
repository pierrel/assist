---
name: render
description: Showing/displaying/opening a file in the user's web view (web UI only). EXAMPLES — "show me fitness.org"; "open my notes"; "view the report"; "display that pdf"; "pull up the recipes file"; "let me see <file>". MUST load before responding when the user asks to SHOW, OPEN, VIEW, DISPLAY, or pull up a file in their workspace.
---

# Render — show a file in the user's web view

When the user asks to **show, open, view, display, or pull up** a file, render
it for them in the web view. Do this by emitting a **render block** — a fenced
code block tagged `render` whose body names the file's type and path:

```render
type: file
path: /workspace/PATH-TO-THE-FILE
```

Replace `PATH-TO-THE-FILE` with the real file in the user's workspace (for
example `path: /workspace/fitness.org`).

## Showing only part of a file

The user often asks for part of a file. They may say it **explicitly** ("lines
10-40 of notes.org", "page 3 of the report") OR **by description** ("show me the
section about backups", "the part on swimming drills", "the chapter on dosage").
Either way the render block must carry a concrete numeric range — `lines: N-M`
for org/md, `pages: N-M` for pdf. So **resolve a description into a range first**.

**Explicit range — use it directly:**
```render
type: file
path: /workspace/notes.org
lines: 10-40
```

**Described section — find it in the file, THEN emit the range:**

- *org / md:* `read_file` the file, find the heading (or text) for the topic the
  user named, and use the line range that section spans — from its heading line
  through the line just before the next heading at the same or a higher level
  (end of file if it's the last). Emit `lines: START-END`.
  - e.g. user: "show the section about backups in notes.org" → you read it, see
    `* Backups` starts at line 42 and the next `*`/`**` heading is line 58 → emit
    `path: /workspace/notes.org` with `lines: 42-57`.
- *pdf:* load the `pdf` skill and use its tools to read the pdf's text and find
  which page(s) cover the topic, then emit `pages: N-M` for those pages.

Use `lines:` for org/md and `pages:` for pdf. Always write a range as `N-M` (for
a single line or page use the same number twice, e.g. `pages: 3-3`). The range in
the block must be numbers you resolved — never put a description like
`lines: the backups section` in the block. Omit the range to show the whole file.

## Rules

- Emit the render block **instead of** reading the file and summarizing or
  pasting its contents. The block displays the actual file; a summary is not
  what the user asked for.
- Only `.org`, `.md`, and `.pdf` files render. For any other type, read and
  summarize it instead (no render block).
- If you don't know the exact path, find the file first (e.g. `glob`), then
  emit the block with its real path.
- You may add a short sentence before the block (e.g. "Here's your file:"), but
  the render block itself must be exactly the fenced `render` block above.
- For content you are writing yourself — tables, lists, code, formatting — just
  use normal markdown; the chat renders it. The render block is **only** for
  showing an existing workspace file.
