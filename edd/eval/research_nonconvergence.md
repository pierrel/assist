# Research nonconvergence evaluation

This suite prepares bounded, source-auditable model trials. It does not change
Assist. The normal `web-main` case submits an ordinary user request through the
web Run route, lets the current main agent choose its own research path, and
requires a saved requested note or report plus a final parent reply after its
launched child work settles. The `delegate-capability`,
`research-manager-capability`, and `research-leaf` profiles enter those roles
directly and count only as capability diagnostics.

## Cases and evidence

`research_live_cases.json` freezes 13 natural prompts, initial notes, material
facts to audit, source markers, and deadlines. Bundled and component cases in
each `pair_group` start separate fresh threads with the same note and source
corpus. Music and ballot notes retain the compact historical problem shape;
the synthetic Helio family supplies the longer, mixed-confidence eight-tier
note and a three-tier completing control. `helio_three_partial` uses a recorded
HTTP 503 for one tier. The prompt never directs a tool or delegate route.

`research_live_sources.json` is a known-answer **evaluation** corpus. The
Music/ballot pages are synthetic paraphrases at source-shaped URLs, not a
claim that those sites currently serve those exact bytes. Helio pages and
names are wholly invented. `research_live_fixtures.py` substitutes responses
for the existing host `search_internet` and `read_url` tools, including PDF
text. Unmatched searches return empty results; URLs outside listed pages and
the finite Helio citation frontier return an explicit corpus miss. It prevents
browser navigation in these fixture runs. A corpus miss says nothing about a
site's real availability. Inspect `fixture.jsonl` and `trace.jsonl` for every
trial; an agent's shell/network action outside those substituted tools makes
the trial unsuitable as a frozen-source comparison.

The historical incident examples are separate controls, not results from this
new suite. The November 10 direct delegate ended at an external 510-second
cutoff with five synchronous task starts, one return, and no final. A distinct
normal-main case delivered at about 960 seconds but failed its historical
source-quality audit. Their profiles and deadlines differ, so neither proves
nesting caused the timing difference.

## Prepare and run later

Use an immutable private copy of the installed product, overlay only this
suite's fixed eight-file eval inventory, and run
`research_live_supervisor --preflight-only` first. Preflight rejects any
`assist/`, `manage/`, or `dockerfiles/` byte drift from the declared installed
source and any stale or missing eval overlay against a separate reviewed source
checkout. The source checkout must differ from the private product copy; the
executing module is not its own expected source. Keep the source JSON and every
attempt.

The supervisor CLI takes these positional arguments:

```
python -m edd.eval.research_live_supervisor CASE_ID ARTIFACT_DIR \
  PRIVATE_PRODUCT_ROOT DEPLOYED_CODE_ROOT REVIEWED_EVAL_SOURCE AGENTIC_CLI \
  CURRENT_EGRESS_PROOF ENV_FILE [--profile web-main] [--preflight-only]
```

`ENV_FILE` is the existing ignored local model/search environment file; never
put its contents in the command line or tracked artifacts. Set `PYTHONPATH` to
`PRIVATE_PRODUCT_ROOT` for both preflight and supervisor. For a real model
trial, run the supervisor in a unique transient user service with
`KillMode=control-group`, a `RuntimeMaxSec` slightly beyond the case deadline,
and the existing `CODEX_THREAD_ID` forwarded. The supervisor invokes
`tools/agentic resource run llm` itself, hard-stops its private worker before
the case deadline, reaps only its own sandbox, reconciles private Runs, and
exports its checkpoint. Use the current production-admission policy; a
production-busy override needs the explicit current user-message reference in
`RESEARCH_LIVE_USER_MESSAGE_REF` with
`RESEARCH_LIVE_ALLOW_PRODUCTION_BUSY=1`.

For a normal trial, the frozen case ceiling is 2,700 seconds including
teardown. The explicit capability profiles are capped at 600 seconds by the
supervisor. Space local-model trials into at most 60 minutes of GPU work with
a 15-minute break between blocks. One completion signal plus a bounded
backstop should inspect the unit, trace growth, Runs and cleanup; do not poll
in a sleep loop. Keep every censored, errored and successful attempt.

After completion, run:

```
python -m edd.eval.research_nonconvergence ARTIFACT_DIR > metrics.json
```

The observer distinguishes external deadline non-delivery, product Run
timeout/error, cancellation, setup failure, capability answer and normal web
delivery. An unfinished turn at the internal observation cutoff remains
non-delivery even if evaluator cancellation or TestClient shutdown later makes
a draft look final; evaluator-requested cancellation is recorded separately
from an agent's cancellation. It reports actual task/Run ancestry, pending branches, repeated
normalized search/page requests, source-marker visibility, reported tokens,
provider-message JSON byte volume, model and retrieval spans, durable Run
pending time, model-slot wait and reported active time. Summed spans can
overlap; they are not a partition of wall time. A source marker is a timing
probe, not a semantic truth check. `provider_message_json_bytes_sum` is
serialized message size, not measured wire traffic.
Older captures that stored only Run ID/status, and direct profiles with no
durable Run records, have no measured durable ancestry; their child count and
durable depth are `null`, not measured zero. Synchronous task starts/returns
remain independently observable in their raw trace.

## Outcome review after each trial

Compare the raw user prompt and original note with the saved note, report,
final parent reply, source fixture bodies, `fixture.jsonl`, provider inputs,
and the full checkpoint. Record each value separately:

| Value | Audit question |
|---|---|
| Thorough | Were material requested facts covered, good existing note facts preserved, and unresolved parts stated? |
| Focused | Did added questions or research materially serve the request, or divert work from it? Useful aligned context is allowed. |
| Correct | Does each material claim follow from the source it cites, with honest source access and uncertainty? Were model-added premises falsely attributed to the original note? |
| Timing | What were total wall time, active/model/provider/queue spans, retrieval work and token/byte costs? |

Inspect every generated `task` brief against the user request for unsupported
subquestions; task count or nesting alone is diagnostic, not a pass criterion.
A fast answer with missing or unsupported facts fails quality. A deep route
that finishes accurately can pass. A provisional parent reply or child report
without the required final artifact is non-delivery. If an external deadline
ends first, record the partial graph and censor; it does not prove the product
would never converge. No automatic string-presence score substitutes for the
source and final-artifact audit.

Begin later behavioral sampling with one normal-main bundled and component
pair per family and the separate capability controls. Keep source, prompt,
profile, model settings, and deadline fixed within each comparison. Repeat
cases with an observed gap before attributing a change or reporting a pass
rate; do not select only successful draws. The old raw historical traces are
not baseline draws under this source/profile envelope.
