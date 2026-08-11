---
name: render
description: MUST load before reading or replying when the user asks to show, open, view, or display a named or existing workspace file in the web conversation. Also use for rendering generated or existing PNG graphs/charts and maps. EXAMPLES — "show me fitness.org"; "open the project notes"; "display the chart here"; "show that PNG again".
allowed-tools: map_data
---

# Render — show a file in the user's web view

## Creating and showing graphs

When the user asks to **show a graph or chart**, complete this exact sequence:

1. Use `execute` with Python and matplotlib to create a PNG under persistent
   `/tmp` (for example `/tmp/chart.png`). Matplotlib is already installed;
   do not install packages.
2. Verify the command succeeded and the PNG exists at that exact path.
3. In the same response, emit a file render block naming that exact PNG:

```render
type: file
path: /tmp/chart.png
```

The model is text-only, but that does not prevent it from displaying an image
file it created. Do not `read_file` a PNG, replace it with an ASCII graph, create
an HTML/markdown substitute, or merely tell the user where the PNG was saved.

If the user later says **show/render the graph here**, reuse the PNG path from
the conversation and emit the block again immediately. Do not inspect, recreate,
or re-verify an existing PNG before emitting its block.

When the user asks to **show, open, view, display, or pull up** a file, render
it for them in the web view. Do this by emitting a **render block** — a fenced
code block tagged `render` whose body names the file's type and path:

```render
type: file
path: /workspace/PATH-TO-THE-FILE
```

Replace `PATH-TO-THE-FILE` with the real file — either in the user's workspace
(e.g. `path: /workspace/fitness.org`) or under `/tmp` (e.g.
`path: /tmp/summary.md`). **`/tmp` is a persistent scratch directory** that
renders exactly like the workspace, so you can `write_file` a file there to show
it (see "Showing other file types" below).

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

## Showing other file types

`.org`, `.md`, and `.pdf` render richly (formatted), and `.png` renders as an
inline image. Other files render as plain text. So you CAN show a `.txt`, `.py`,
`.j2`, log, config, or extension-less file directly:

```render
type: file
path: /workspace/assist/templates/deepagents/sub_research.txt.j2
```

**When the file would read better converted, convert it first.** If a file is
easily turned into markdown/org — a template like `sub_research.txt.j2` (really
just prompt text), a `.txt` that's already markdown-shaped, notes you want
cleaned up — `write_file` a converted copy under `/tmp` and render THAT:

```
write_file  /tmp/sub_research.md   ← the file's content (as markdown)
```
```render
type: file
path: /tmp/sub_research.md
```

`/tmp` is a per-thread scratch dir that persists across turns and renders like the
workspace, so this is the way
to show a nicely-formatted version of a file whose own extension wouldn't format.
Use it when converting adds value; otherwise just render the original path as
plain text (above).

## Maps — show places and routes on a map

**Whenever your answer names real-world places or a route — places you recommend,
compare, or locate; "how far is A from B"; a business with an address — you show them on
a map, alongside your written answer. This is your default; don't wait to be asked.** So
a "which of these is best?" or "where should I go?" answer about real places gets a map
too, not just prose. Three steps:

**1. Get coordinates** from the **`map_data`** tool — never guess a lat/lon. Pass the
places (comma- or semicolon-separated) and any routes as `"A -> B"`:

    map_data(places="Four Barrel Coffee SF, Ritual Coffee SF, Haus Coffee SF", routes="Four Barrel SF -> Haus Coffee SF")

It returns a `lat,lon` per place and a walking polyline per route.

**2. Emit one `type: map` render block** — this is required, not optional: once
`map_data` returns coordinates you MUST emit the block (don't answer in prose alone — the
coordinates are useless to the user without it). Copy them in — one `pin:` per place; one
`path:` per route. If the user's own location is in the message context, add it as a pin
prefixed with `origin` (bare `lat,lon` — drop any `~`) so it stands out; skip it if it's
a different city:

```render
type: map
pin: <lat>,<lon> <label>
pin: origin <lat>,<lon> <label>
path: <encoded-polyline> <label>
```

Enough coordinates to place the pins is enough to emit the block — don't keep re-calling
`map_data` for perfect addresses; emit what you have, then write your answer.

**3. Give your normal written answer too** — the map is in addition to it.

If you first have to FIND the places (e.g. "good coffee shops near X"), research them
as you normally would, THEN map the ones you found. Never map a name you haven't
confirmed — `map_data` geocodes real names, so a made-up one gives a wrong/empty pin.

## Rules

- Emit the render block **instead of** reading the file and summarizing or
  pasting its contents. The block displays the actual file; a summary is not
  what the user asked for.
- `.org`, `.md`, and `.pdf` render richly; `.png` renders as an inline image;
  other files render as plain text or a converted `/tmp/*.md` copy. Either way,
  SHOW the file — don't fall back to summarizing.
- If you don't know the exact path, find the file first (e.g. `glob`), then
  emit the block with its real path.
- You may add a short sentence before the block (e.g. "Here's your file:"), but
  the render block itself must be exactly the fenced `render` block above.
- For ordinary text content you are writing yourself — tables, lists, code,
  formatting — use normal markdown. A generated PNG graph is a file artifact,
  so show it with the render block after creating it.
