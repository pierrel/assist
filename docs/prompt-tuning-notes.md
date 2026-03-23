# Prompt Tuning Notes

## Context

These notes capture behavioral observations from eval runs on 2026-03-22.

Model history:
- Ministral-3B (baseline, 2026-03-21): **49/57 passing**
- Qwen2.5-14B-AWQ (2026-03-22): 36/57 with prompt changes, reverted to re-baseline
- Qwen3-14B-AWQ (2026-03-22): testing in progress

**Important**: The observations in sections 1-6 below were from Qwen2.5-14B.
Qwen3-14B has a different behavioral profile — see the Qwen3-specific section.

---

## Observed Behavioral Differences vs Ministral-3B

### 1. Answers from own knowledge instead of delegating
**Symptom**: General agent does 1 model call with no tool calls, answers
research questions directly (e.g. "what are good custom pants options").
**Ministral behavior**: Reliably called task() to delegate.
**Qwen2.5 behavior**: Often answers from training data, skips tool routing.
**Implication**: Routing rules need to be more explicit or use a different
framing — "call task() immediately" may need a concrete trigger example.

### 2. Multi-intent completion (research + task write)
**Symptom**: After research completes, agent returns without writing the
TODO to the user's task file.
**Root cause**: Model treats research response as "done" for the whole request.
**Fix direction**: Plan step must enumerate both sub-tasks before starting.
The `write_todos` planning step needs to be more strictly enforced.

### 3. Context-agent gives up instead of exploring
**Symptom**: "I couldn't find any files related to..." when files clearly exist.
**Root cause**: The old context-agent `ls path="/"` rule made it look at
the system root, missing workspace files. This was partially fixed in the
session but the revert undid it.
**Fix direction**: The `ls path="/"` in the Rules section is wrong — it should
be `ls path="{{ workspace_dir }}"`. This is a safe, targeted fix.

### 4. Context-agent creates files (violates read-only)
**Symptom**: Context-agent creates `tasks.org` or `todos.org` instead of
surfacing the existing `tasks.md`.
**Root cause**: Model defaults to org format and ignores the read-only constraint.
**Fix direction**: Bold/header read-only enforcement + explicit response format
showing that the output is text, not a file operation.

### 5. File extension override
**Symptom**: General agent edits/creates `tasks.org` even when context-agent
found `tasks.md`.
**Root cause**: Model has strong org-format preference from training data.
**Fix direction**: Explicit instruction to use `edit_file` on the EXACT path
returned, including extension. The word "NEVER" with a concrete example works.

### 6. Dev-agent not running tests
**Symptom**: `test_runs_tests` fails — dev-agent doesn't call `execute`.
**Likely cause**: Model behavior difference, not prompt issue. Qwen2.5
may need a more explicit "always run tests after making changes" rule
in the dev-agent prompt.

---

## What NOT to Do (lessons from today's session)

1. **Don't add example responses to the context-agent** — a concrete example
   of the output format caused the model to rush to match the example format
   rather than thoroughly exploring the filesystem.

2. **Don't over-specify routing** — adding too many "NEVER answer from your
   own knowledge" rules caused the general agent to route everything to
   research-agent, including things better handled locally.

3. **Don't change multiple prompts in one targeted fix loop** — changes
   interact in non-obvious ways. Fix one thing at a time and run the full
   suite between changes.

4. **Don't use `write_todos` example syntax in the plan step** — showing
   a checklist template caused the model to produce that template literally
   rather than generating a contextual plan.

---

## Qwen3-14B-AWQ Specific Issues

### 7. Thinking mode breaks tool-call parsing (CRITICAL, OPERATIONAL)
**Symptom**: Model outputs `<think>...</think>` reasoning block followed by
`<tool_call>{"name": "ls", ...}</tool_call>` — but the entire thing arrives as
plain text content, not as structured `tool_calls`. The model does 1 model call
and "completes" without any tool calls being executed.
**Root cause**: Qwen3's default thinking mode generates `<think>` tokens. The
`qwen3_coder` parser in vLLM cannot extract tool calls when `<think>` blocks
precede them in the same content stream.
**Fix**: Add `--reasoning-parser deepseek_r1` to the vLLM serve command. This
strips `<think>` blocks from the content before the tool-call parser runs.
Verified working on vLLM 0.15.1.
**Note**: `--chat-template-kwargs '{"enable_thinking": false}'` does NOT exist
in vLLM 0.15.1. Do not use it.

### 8. Context window is 40960, not 128k
The research agent reported Qwen3-14B supports 128k context. This is true for
the base model with YaRN scaling, but the **AWQ variant** has
`max_position_embeddings=40960` baked into its `config.json`. Exceeding this
causes a startup error. Do not set `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` — positions
beyond 40960 will produce NaN with RoPE.

---

## Recommended Prompt Changes (conservative, one at a time)

### P1 — Fix context-agent ls path (HIGH CONFIDENCE, LOW RISK)
In `context_agent.md.j2` Rules section:
```
# Change:
always `ls` with `path="/"` and read README.* before answering
# To:
always `ls` with `path="{{ workspace_dir }}"` and read README.* before answering
```
This is clearly correct. The old rule causes the agent to list `/` (system root)
instead of the workspace, causing it to miss all workspace files.
Expected impact: fixes several context-agent "couldn't find" failures.

### P2 — Context-agent read-only enforcement (MEDIUM CONFIDENCE, LOW RISK)
Add to the top of context_agent.md.j2, after the intro paragraph:
```
YOU ARE READ-ONLY. Never call write_file, edit_file, or any tool that
creates or modifies files. Your output is always a text response.
```
Expected impact: fixes test_no_org_format_guidance and similar tests where
the context-agent creates a file instead of surfacing the existing one.

### P3 — Research agent Sources section (MEDIUM CONFIDENCE, LOW RISK)
In `research_instructions.txt.j2` citation_rules:
```
# Change:
- End with *** Sources that lists each source with corresponding numbers
# To:
- MANDATORY: Every report MUST end with a *** Sources section. A report
  without *** Sources is incomplete. Before writing the file, verify
  your draft includes this section.
```
Expected impact: fixes test_has_references_with_urls.

### P4 — Context-agent file path in response (LOW CONFIDENCE, MEDIUM RISK)
Add to Step 3 in context_agent.md.j2:
```
Always state the exact filename in your response (e.g. "tasks.md", "inbox.org").
```
Expected impact: fixes test_no_org_format_guidance assertion on file path.
Risk: May cause over-verbose responses.

### P5 — General agent file extension compliance (LOW CONFIDENCE, HIGH RISK)
The fix for using exact file paths from context-agent worked in isolation
but caused regressions. Needs to be very carefully worded.
Hold until P1-P4 are validated.

---

## Process Recommendation

1. Run full eval with Qwen3-14B + reverted prompts (establishes clean Qwen3 baseline)
2. Compare Qwen3 baseline against Ministral baseline and Qwen2.5 run
3. Apply P1 only, run full eval
4. Apply P2, run full eval
5. Apply P3, run full eval
6. Compare each step before adding the next change

Note: P1-P5 were designed for Qwen2.5-14B. They need re-validation with Qwen3.
Some may be unnecessary (Qwen3 is a better instruction follower) and some may
need different wording.
