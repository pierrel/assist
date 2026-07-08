---
name: render
description: Rendering things in the user's web view (web UI only) beyond plain text — files AND maps. FILES — "show me fitness.org"; "open my notes"; "view the report"; "display that pdf"; "pull up the recipes file". MAPS — whenever the discussion involves mappable, real-world locations (places, addresses, businesses, routes, "near", "walking distance", "how far", "on a map"), e.g. "coffee shops near X"; "map these places"; "show the route from A to B". MUST load before responding when the user asks to SHOW/OPEN/VIEW/DISPLAY/pull up a file, OR whenever real-world places/routes are being discussed that would be clearer on a map.
---

# Render — show a file in the user's web view

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
it (see "Showing a file that isn't .org/.md/.pdf" below).

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

## Showing a file that isn't .org / .md / .pdf

`.org`, `.md`, and `.pdf` render richly (formatted). **Any other file still
renders — as plain text.** So you CAN show a `.txt`, `.py`, `.j2`, log, config,
or extension-less file directly:

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

`/tmp` persists for the turn and renders like the workspace, so this is the way
to show a nicely-formatted version of a file whose own extension wouldn't format.
Use it when converting adds value; otherwise just render the original path as
plain text (above).

## Maps — show places and routes on a map

When the discussion involves real-world **locations or routes** — coffee shops
near a spot, "how far is A from B", "map these places", businesses with
addresses — show them on a map, **in addition to** your normal written answer.
Emit a **map render block**:

```render
type: map
pin: <lat>,<lon> <label>
path: <encoded-polyline> <label>
```

- One `pin:` line per place: its `latitude,longitude` then a short label.
- One `path:` line per route: the route's encoded polyline then a label.
- A map may have many pins and paths; emit **one** `type: map` block per reply.

**You do NOT know coordinates or routes — never make them up.** (Guessing an
address or a lat/lon is the mistake that ruins a map.) Get them from the
`map_data` tool FIRST. Call it with the places as a **semicolon-separated
string** in the `places` argument (and any routes as `"A -> B"` in `routes`),
exactly like this:

    map_data(places="Four Barrel Coffee, San Francisco; Ritual Coffee, San Francisco; Haus Coffee, San Francisco", routes="Fellow Barber SF -> Haus Coffee SF")

It returns the exact `lat,lon` for each place and an encoded polyline for each
route. Copy those into the `pin:`/`path:` lines exactly. (Always put the actual
place names in `places` — never call `map_data` with an empty `places`.)

The map plots places you've IDENTIFIED — it does NOT find them for you. If the
request means you first have to FIND the places (e.g. "good coffee shops near
X"), **do that research first, exactly as you normally would** (search and read
to find real, well-reviewed places) — the map is the LAST step. Never map a place
you haven't confirmed exists: `map_data` geocodes a real name, so a made-up name
gives a wrong or empty pin.

**Worked example.** User: "good coffee shops near Fellow Barber on Valencia, and
how far to walk?" First **research the shops as you normally would** — search and
read to find real, highly-rated coffee shops near there. THEN call `map_data`
with the shops you found + the walking routes; it returns coordinates + polylines;
then, alongside your written recommendations, you emit:

```render
type: map
pin: 37.76181,-122.42191 Fellow Barber (start)
pin: 37.75266,-122.41372 Haus Coffee
pin: 37.76702,-122.42179 Four Barrel
path: sy{foUtrk_~gA}EynB Fellow Barber to Haus (24 min walk)
```

Still give your normal written answer — the map is IN ADDITION to it, never a
replacement for the research and recommendations.

## Rules

- Emit the render block **instead of** reading the file and summarizing or
  pasting its contents. The block displays the actual file; a summary is not
  what the user asked for.
- `.org`, `.md`, and `.pdf` render richly; any other file renders as plain text,
  or convert it to a `/tmp/*.md` copy first (see "Showing a file that isn't
  .org/.md/.pdf"). Either way, SHOW the file — don't fall back to summarizing.
- If you don't know the exact path, find the file first (e.g. `glob`), then
  emit the block with its real path.
- You may add a short sentence before the block (e.g. "Here's your file:"), but
  the render block itself must be exactly the fenced `render` block above.
- For content you are writing yourself — tables, lists, code, formatting — just
  use normal markdown; the chat renders it. The render block is **only** for
  showing an existing workspace file.
