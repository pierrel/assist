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

**Worked example.** User: "good coffee shops near Fellow Barber on Valencia, and
how far to walk?" You call `map_data` for the shops + the walking routes; it
returns coordinates + polylines; then, alongside your prose recommendations, you
emit:

```render
type: map
pin: 37.76181,-122.42191 Fellow Barber (start)
pin: 37.75266,-122.41372 Haus Coffee
pin: 37.76702,-122.42179 Four Barrel
path: sy{foUtrk_~gA}EynB Fellow Barber to Haus (24 min walk)
```

Still give your normal written answer — the map is IN ADDITION to it.

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
