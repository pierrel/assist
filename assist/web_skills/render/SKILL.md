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

When the user asks for a specific part — "lines 10-40 of notes.org", "page 3 of
the report", "the second page of receipt.pdf" — add a range to the block:

- **org / md → by line:** add `lines: N-M` (a 1-based inclusive line range):
  ```render
  type: file
  path: /workspace/notes.org
  lines: 10-40
  ```
- **pdf → by page:** add `pages: N-M` (a 1-based inclusive page range):
  ```render
  type: file
  path: /workspace/report.pdf
  pages: 2-5
  ```

Use `lines:` for org/md and `pages:` for pdf. Always write a range as `N-M` (for
a single line or page use the same number twice, e.g. `pages: 3-3`). Omit the
range to show the whole file.

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
