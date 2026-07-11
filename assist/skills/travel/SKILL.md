---
name: travel
description: Real-world travel between places — time and distance by car/bike/walk/transit, AND step-by-step directions (turns, which bus or train), across a metro area and nearby cities, PLUS which geographic areas are covered and adding new ones on request. EXAMPLES — "how long from home to the Ferry Building"; "is it faster to bike or take the train"; "drive time to the airport"; "how far is the office"; "directions to City Hall"; "which bus do I take"; "what areas do you cover"; "can you do directions in Seattle"; "download the Los Angeles area". MUST load before answering any question about travel time, distance, the fastest mode, how to get from one place to another, OR which regions are covered / adding a region.
---

# Travel & directions — real map data, two tools

Two tools, both backed by real map data. **Never estimate times, distances, turns,
or which line to take from your own knowledge** — call the tool.

- **`travel(origin, destination)`** — how LONG / how FAR / which MODE is fastest.
  Returns time + distance for all four modes (car, bike, walk, transit).
- **`directions(origin, destination, mode)`** — HOW to get there: the step-by-step
  route (turns and streets, or which bus/train + transfers) for ONE mode.

## Which tool

- Wants a **time, a distance, or a comparison** ("how long", "how far", "faster to
  bike or drive") → **`travel`**.
- Wants the **steps / the route / which bus or train / turn-by-turn** ("how do I
  get to X", "directions to Y", "which train to Z", "walk me through it") →
  **`directions`**.

## Calling `travel(origin, destination)`

Pass plain place **names/addresses as the user said them** ("home", "the Ferry
Building", "123 Main St") — the one coordinate exception is the user's current
location for "from here"/"near me" (see *Place names* below). Don't pass a mode — it
returns all four modes; you highlight the relevant one when you reply.

## Calling `directions(origin, destination, mode)`

Pass the same kind of place names, plus a **`mode`** — one of `"car"`, `"bike"`,
`"walk"`, `"transit"`. Map the user's words to one of those exactly:

- drive / driving / by car → `"car"`
- bike / biking / cycling / bicycle → `"bike"`
- walk / walking / on foot → `"walk"`
- bus / train / subway / metro / light rail / public transit → `"transit"`

If the user **names a mode**, pass it. If they **don't**, ask which mode they want
— don't guess (transit directions when they meant to drive is a wrong answer).

## Place names (both tools)

- The tool geocodes names itself — pass place NAMES, not coordinates.
- **"from here" / "near me" / "nearby":** when the user wants travel or directions
  from their current location and gives no named origin, pass their coordinates from
  the message context (the `[Message context: ... from ~<lat>, <lon>]` line) as the
  origin — e.g. `"37.77,-122.42"` (the leading `~` and spaces from the context line
  are fine to include). If there's no location in the context, fall back to "home"
  or ask.
- If the user gives only one place ("directions to the airport"), the other is
  usually "home" or the place from context; if unclear, ask.
- For a short/ambiguous name, include the city/area from context (pass "Ferry
  Building San Francisco", not just "the Ferry Building"). Both tools echo the
  **resolved place names** — if one looks wrong, retry with a more specific name or
  tell the user; don't trust the result.

## Presenting the result

- **travel:** lead with the mode asked about ("driving is ~25 min"); else compare
  briefly. Round naturally. If a mode is `unavailable`, say so — don't fill a guess.
- **directions:** relay the numbered steps. For **street** routes (car/bike/walk)
  the turn directions are **approximate** — it's fine to hedge ("roughly"), and
  never invent a turn or street the tool didn't give. For **transit**, relay the
  lines, stops, transfers, **and the board/departure times** as listed (e.g. "catch
  the 9R at 2:47 PM"). Mention the resolved place names if they might differ from
  what the user meant.

## Surfacing problems (don't hide a degraded result)

When a result is degraded, NOTICE it and tell the user — and when more information
would likely fix it, ASK for that:

- **No route / unavailable / "couldn't find a place"** → say so plainly (never
  substitute an estimate or route from memory). This often means the place is
  **outside the loaded regions** — follow *Regions & coverage* below (state what you
  cover, and offer to add the region) rather than only asking for a better address.
- **A resolved place name looks wrong** (the echoed name isn't what the user meant)
  → point it out and offer to retry with a more specific place.
- **Transit comes back `unavailable`** → don't just say "unavailable". Call
  `list_regions` to see which region the trip is in and whether it has transit; if
  that region shows **no transit**, tell the user transit data isn't loaded for
  *that named region* (car/bike/walk still work). Name the region, not a generic
  "unavailable".
- **Approximate street turns** → it's fine to hedge; don't present an approximate
  turn as exact.

## Regions & coverage

Travel/directions/geocoding only work inside the **loaded regions**. Two tools let
you answer coverage questions and add regions on request:

- **`list_regions()`** — the regions currently loaded and whether each has transit.
- **`find_regions(query)`** — resolve a place NAME to the exact downloadable region
  id(s). **You never type a region id yourself** — pass a name and use the id it
  returns.

**When the user asks what you cover** ("what areas do you cover", "can you do
Seattle") → call `list_regions()` and answer from it.

**When a place can't be found** (a `travel`/`directions` "couldn't find a place",
or a coverage question about somewhere you don't have):

1. Call `list_regions()` and tell the user what you *do* cover.
2. Work out the **US state or larger region** the place is in (e.g. Seattle → the
   state of Washington; a San Diego address → Southern California) and pass THAT name
   to `find_regions` — not the street address, not the bare city if it isn't itself a
   region. If `find_regions` returns nothing, you named the wrong area — ask the user
   which state/region it's in and try again.
3. **Offer to add** the best (smallest covering) candidate it returns: name the
   region and note that adding it takes a while (it downloads and rebuilds map data).
   Only add it if the user confirms — never start an add on your own.

Don't guess a region id, and don't claim to have added a region you haven't.

## When NOT to use

- Places that aren't real/locatable, or purely hypothetical distances.
