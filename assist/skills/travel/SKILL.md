---
name: travel
description: Real-world travel time between places by car, bike, walking, and public transit (with distance for car/bike/walk), across a metro area and nearby cities. EXAMPLES — "how long from home to the Ferry Building"; "is it faster to bike or take the train downtown"; "drive time to the airport"; "how far is the office"; "can I walk to the park". MUST load before answering any question about travel time, distance, directions, or how to get from one place to another.
---

# Travel — real-world time and distance between places

When the user asks how long or how far it is to get from one place to another —
or which way is fastest — call the `travel` tool. Do NOT estimate distances or
times from your own knowledge; the tool computes them from real map data.

## How to call it

`travel(origin, destination)` — pass plain place **names or addresses exactly as
the user said them** ("home", "the Ferry Building", "123 Main St", "the airport").

- Do NOT pass coordinates — the tool geocodes names itself.
- Do NOT pass a mode — it always returns all four (car, bike, walk, transit);
  you choose which to highlight when you reply.
- If the user gives only one place ("how far is the airport"), the other is
  usually "home" or the place from context — if it's unclear, ask.
- For a short or ambiguous place name, include the city/area from context (pass
  "Ferry Building San Francisco", not just "the Ferry Building") so the geocoder
  picks the right place. The result echoes the resolved place names — if one
  looks wrong, retry with a more specific name or tell the user, and don't trust
  those numbers.

## Presenting the result

The tool returns a per-mode summary (time and distance for car/bike/walk; time
only for transit) plus the resolved place names it routed between. When you reply:

- Lead with the mode the user asked about ("driving is ~25 min"); if they didn't
  name one, give a short comparison across modes.
- Round naturally ("about 25 minutes", not "24.7 min").
- Mention the **resolved place names** if they might differ from what the user
  meant, so they can correct you ("routing from <resolved origin>…").
- If a mode comes back `unavailable` (e.g. transit not covered there), say so
  plainly — do not fill it in with a guess.

## When NOT to use it

- For places that aren't real/locatable, or purely hypothetical distances.
- If `travel` reports routing is unavailable, tell the user you couldn't look it
  up right now — do NOT substitute an estimate from memory.
