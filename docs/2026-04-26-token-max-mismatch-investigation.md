# Investigation: Thread 20260426073544-71c17777 BadRequestError

**Note (2026-04-28):** `ASSIST_CONTEXT_LEN` and `ASSIST_MODEL_NAME` are
no longer environment variables — both are auto-discovered from
`${ASSIST_MODEL_URL}/models`. References to them below are preserved
as historical context. The `max_input_tokens` value the middleware
reads is now sourced from `OpenAIConfig.context_len`, populated by the
probe.

**Date:** 2026-04-26
**Thread:** `20260426073544-71c17777`
**Description:** Giants weekend season ticket packages and game time restrictions
**Status at investigation time:** `{"stage": "ready"}`
**Status of fixes (2026-04-26):** Recommendations 1, 3, and 4 applied to
`assist/agent.py`. Regression eval at
`edd/eval/test_research_multi_turn_token_regression.py` captures the failing
turn sequence. Recommendation 2 (proactive message-trim middleware) deferred —
see "Status" section at the bottom.

## The error

```
Model call failed after 1 attempt with BadRequestError: Error code: 400 - {
  'error': {
    'message': "This model's maximum context length is 53616 tokens.
                However, you requested 0 output tokens and your prompt
                contains at least 53617 input tokens, for a total of at
                least 53617 tokens. Please reduce the length of the
                input prompt or the number of requested output tokens.
                (parameter=input_tokens, value=53617)",
    'type': 'BadRequestError',
    'param': 'input_tokens',
    'code': 400
  }
}
```

The prompt sent to the model exceeded the model server's context cap by exactly one token (53,617 vs 53,616).

## The configuration is correct

The active deployment is run by systemd at `/etc/systemd/system/assist-web.service`. Relevant excerpt:

```
Environment="ASSIST_MODEL_URL=http://0.0.0.0:8000/v1"
Environment="ASSIST_MODEL_NAME=stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ"
Environment="ASSIST_CONTEXT_LEN=53616"
```

This matches the model server's reported limit exactly, so this is **not** a configuration drift. The harness has the right number — it just allowed the prompt to grow past it.

`assist/code/assist/model_manager.py:56,98` reads `ASSIST_CONTEXT_LEN` and writes it as `llm.profile["max_input_tokens"]`:

```python
context_len = int(os.getenv("ASSIST_CONTEXT_LEN", "32768"))
...
llm.profile["max_input_tokens"] = config.context_len
```

## Root cause: context-management gap

### Problem 1 — the only context middleware is reactive and tool-result-only

`ContextAwareToolEvictionMiddleware` (`assist/middleware/context_aware_tool_eviction.py`) is the **only** middleware watching context size. It is wired into the agent at three places:

- `assist/agent.py:114` — main agent: `trigger_fraction=0.75`
- `assist/agent.py:197` — context agent: `trigger_fraction=0.75`
- `assist/agent.py:245` — research agent: `trigger_fraction=0.70`
- `assist/agent.py:318` — dev agent: `trigger_fraction=0.75`

It only acts in `wrap_tool_call`, i.e. on **incoming tool results** before they're added to messages. From the docstring (`context_aware_tool_eviction.py:26-44`):

> 1. Calculates current context usage from conversation history
> 2. Calculates incoming tool result size
> 3. Checks if combined size exceeds threshold (default: 75% of max_input_tokens)
> 4. If yes, writes result to /large_tool_results/{tool_call_id} and replaces with a reference message

Once messages (AI turns, prior tool results that already passed through, user inputs) accumulate in state, **nothing trims them**. There is no `SummarizationMiddleware`, no `trim_messages`, no compaction step before `wrap_model_call`. So the conversation can grow past the cap purely through ordinary turns, and tool-result eviction never gets a chance to intervene.

### Problem 2 — `chars // 4` token estimate

`context_aware_tool_eviction.py:89-96`:

```python
def _estimate_tokens(self, content: Any) -> int:
    """Estimate token count using conservative approximation.
    Uses ~4 characters per token, which works well for most content types.
    """
    if isinstance(content, str):
        return len(content) // 4
    return len(str(content)) // 4
```

For Qwen3-Coder content — code, JSON tool arguments, non-ASCII — the real tokenizer often produces more tokens than `chars // 4` predicts. Combined with the 25% headroom (eviction at 40,212 → cap at 53,616), an eviction decision based on chars/4 can let through prompts that the server's tokenizer scores well over 53,616.

### Problem 3 — `BadRequestError` is not retried with sanitization

The wired-in retry middleware (`agent.py:104-106`) is:

```python
retry_middle = ModelRetryMiddleware(
    max_retries=3,
    retry_on=(InternalServerError, TimeoutError, ConnectionError),
    backoff_factor=2)
```

`BadRequestError` is explicitly **not** in `retry_on`. The comment immediately above (`agent.py:102-103`) says:

> BadRequestError (400) is handled by invoke_with_rollback via checkpoint rollback.

but rollback only rewinds state — it doesn't reduce token count, so on a context-overflow it just replays the same oversized prompt.

A more targeted middleware exists at `assist/middleware/bad_request_retry.py` — `BadRequestRetryMiddleware`. Its docstring (lines 1-16):

> When vLLM (or another provider) returns a 400 Bad Request — typically because messages contain control characters, malformed JSON escapes, or other unparseable content — this middleware catches the error, aggressively sanitizes the request messages, and retries.

On retries it escalates: pass 1 strips control chars and fixes JSON escapes; pass 2+ also truncates large tool-result messages (`bad_request_retry.py:96-102, 200-201`). That last step would have rescued this exact request.

It is **not in the middleware list** in `agent.py`. That's why the error message says "failed after 1 attempt."

## Why this thread specifically blew up

The thread's `domain/` directory contains:

- `giants_preliminary_calculation.md`
- `giants_ticket_analysis_report.md`
- `giants_ticket_analysis_report_final.md`
- `giants_ticketing_answer.md`
- `giants_ticketing_links.md`

This was the **research agent** producing several reports iteratively (the `_final` suffix and the multiple analysis passes are characteristic). Research agents accumulate web-search results and prior-report content in the message history. That growth path is precisely what tool-result eviction does not cover once the results have already been incorporated as messages.

The research agent runs at `trigger_fraction=0.70` (`agent.py:246`) — slightly more aggressive than the default — but still too generous given this access pattern, and still subject to Problems 1 and 2 above.

## Recommendations, ordered by impact

1. **Wire `BadRequestRetryMiddleware` into the agent middleware list.** It already exists. Aggressive truncation on the second attempt is the cheapest immediate win and would have rescued this exact request.

2. **Add a context-trim or summarization middleware** that runs in `wrap_model_call` (not `wrap_tool_call`), so it operates on the full message list right before send. This closes the "messages grew through normal turns" gap.

3. **Replace `chars // 4` with a tokenizer-aware estimate** for the eviction decision, or at minimum **lower `trigger_fraction`** to 0.60 to widen the safety margin against tokenizer disagreement.

4. **Tune the research agent specifically** (`agent.py:245`) — 0.70 is still too generous given how it accumulates intermediate reports. Consider 0.55–0.60.

## Status

Applied 2026-04-26:

- **#1 — `BadRequestRetryMiddleware` wired into general / context / research
  agents.** `assist/agent.py:111` (general), `assist/agent.py:216` (context),
  `assist/agent.py:268` (research). The dev-agent already had it. On
  `BadRequestError` the middleware sanitizes messages, then on subsequent
  retries truncates large tool-result messages to 20k chars and retries again.
- **#3 — Lowered `trigger_fraction` on `ContextAwareToolEvictionMiddleware`.**
  General/context dropped from 0.75 to 0.60; research dropped from 0.70 to
  0.55. Widens the safety margin against the `chars // 4` token underestimate.
  We did not replace `chars // 4` itself — the lower fractions are a cheaper
  approximation of the same goal.
- **#4 — Tuned research agent specifically.** `trigger_fraction=0.55` (down
  from 0.70) on the research-only middleware list — more aggressive than
  general/context because research accumulates web-search results and report
  drafts faster.
- **Regression eval added.** `edd/eval/test_research_multi_turn_token_regression.py`
  runs the three failing turns through `ThreadManager` → general agent →
  research subagent. Pass criterion: no `openai.BadRequestError` reaches the
  caller.

Deferred:

- **#2 — Proactive message-trim or summarization middleware.** Not yet wired in.
  The combination of (a) reactive `BadRequestRetryMiddleware` truncation on
  retry and (b) lower `trigger_fraction` on tool-result eviction should cover
  the regression. If the eval still surfaces overflows after the changes
  above, revisit this — the right shape is a `wrap_model_call` middleware
  that drops the oldest tool-result messages until the message list is under
  some fraction of `max_input_tokens`. Tracked in `roadmap.org` under
  Reliability.

## Files inspected

- `/etc/systemd/system/assist-web.service` — confirmed `ASSIST_CONTEXT_LEN=53616`
- `assist/code/assist/model_manager.py` — how `max_input_tokens` is set on the model profile
- `assist/code/assist/agent.py` — middleware wiring for main, context, research, and dev agents
- `assist/code/assist/middleware/context_aware_tool_eviction.py` — the only active context-management middleware
- `assist/code/assist/middleware/bad_request_retry.py` — the existing-but-unwired BadRequest sanitizer
- `assist/code/assist/middleware/local_context_middleware.py` — confirmed it's a stub (just a `UserContextMiddleware` skeleton, no context trimming)
- `assist/threads/20260426073544-71c17777/{description.txt,status.json,domain/}` — thread context
