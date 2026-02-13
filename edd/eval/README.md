# Agent Evaluations

This directory contains evaluation scripts for testing agent behavior under realistic conditions.

## Available Evals

### `eval_multi_turn_research.py`

Tests the agent's ability to handle sustained research conversations over multiple turns.

**What it tests**:
- Multi-turn conversation handling (10 turns by default)
- Research request processing
- Follow-up question handling
- Comparative analysis
- JSON validation under realistic load
- Tool call patterns and concurrency
- Memory and context management

**How it works**:
1. Uses a driving model to generate conversation prompts
2. Starts with an initial research request
3. Generates intelligent follow-ups based on conversation history
4. Tracks metrics: tool calls, tokens, errors, timing
5. Reports comprehensive statistics

**Usage**:

```bash
# Run with defaults (10 turns)
python tests/evals/eval_multi_turn_research.py

# Run with custom number of turns
python tests/evals/eval_multi_turn_research.py --turns 20

# Run quietly (summary only)
python tests/evals/eval_multi_turn_research.py --quiet

# Save metrics to JSON
python tests/evals/eval_multi_turn_research.py --output metrics.json

# Combine options
python tests/evals/eval_multi_turn_research.py --turns 15 --quiet --output results.json
```

**Expected Output**:

```
================================================================================
STARTING MULTI-TURN RESEARCH CONVERSATION EVAL
================================================================================
Turns: 10
Working Directory: /tmp/tmp123abc/xyz
================================================================================

────────────────────────────────────────────────────────────────────────────────
TURN 1/10
────────────────────────────────────────────────────────────────────────────────
USER: Research the key differences between Python asyncio and threading...

AGENT: I'll research the key differences between Python asyncio and threading...
       (response truncated, total length: 2450 chars)

────────────────────────────────────────────────────────────────────────────────
TURN 2/10
────────────────────────────────────────────────────────────────────────────────
USER: Now research the performance characteristics of each approach

AGENT: Based on the previous research, let me investigate performance...
       (response truncated, total length: 1823 chars)

...

================================================================================
EVALUATION SUMMARY
================================================================================
Total Turns: 10
Duration: 145.23 seconds
Total Tool Calls: 28
Total Tool Results: 28
JSON Errors: 0
Avg Tool Calls/Turn: 2.80

Per-Turn Breakdown:
────────────────────────────────────────────────────────────────────────────────
  Turn  1: User:   87 chars | Agent:  2450 chars | Tools: 3 calls, 3 results
  Turn  2: User:   65 chars | Agent:  1823 chars | Tools: 2 calls, 2 results
  Turn  3: User:   71 chars | Agent:  2104 chars | Tools: 3 calls, 3 results
  ...
================================================================================

✅ PASSED: No JSON errors
```

**Metrics Tracked**:

- `total_turns`: Number of conversation turns completed
- `total_tool_calls`: Total tools invoked across all turns
- `total_tool_results`: Total tool results processed
- `json_errors`: Number of JSON validation errors (should be 0)
- `duration_seconds`: Total evaluation runtime
- `avg_tool_calls_per_turn`: Average concurrent tool usage

**Success Criteria**:

✅ All 10 turns complete without errors
✅ No JSON validation errors
✅ Agent uses tools appropriately (research, file operations)
✅ Conversation maintains coherence across turns
✅ Follow-ups are contextually relevant

**Failure Modes**:

❌ JSON serialization errors (especially with large content)
❌ Tool call failures
❌ Context loss between turns
❌ Irrelevant or repetitive responses

## Running All Evals

```bash
# Run all evals with pytest
pytest tests/evals/ -v

# Or run specific eval
pytest tests/evals/eval_multi_turn_research.py -v
```

## Creating New Evals

1. Create a new Python file: `eval_<name>.py`
2. Implement the eval logic with metrics tracking
3. Use `pytest` fixture pattern or `if __name__ == "__main__"` for CLI
4. Document in this README

### Template Structure

```python
"""Evaluation: Description of what this tests."""

class EvalMetrics:
    """Track metrics for the evaluation."""
    pass

def run_eval(**kwargs) -> EvalMetrics:
    """Run the evaluation.

    Returns:
        EvalMetrics object with results
    """
    pass

def main():
    """CLI entry point."""
    metrics = run_eval()
    metrics.print_summary()

    if metrics.has_errors():
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
```

## Continuous Evaluation

These evals can be integrated into CI/CD:

```yaml
# Example GitHub Actions
- name: Run Agent Evals
  run: |
    python tests/evals/eval_multi_turn_research.py --quiet --output metrics.json

- name: Upload Metrics
  uses: actions/upload-artifact@v2
  with:
    name: eval-metrics
    path: metrics.json
```

## Interpreting Results

### Good Signs ✅

- Zero JSON errors
- Consistent tool usage across turns
- Response lengths appropriate to question complexity
- Completion within reasonable time (< 5 min for 10 turns)

### Warning Signs ⚠️

- Increasing response times per turn (memory leak?)
- Decreasing tool usage (agent giving up?)
- Very long responses (runaway generation?)
- High variance in turn duration

### Red Flags 🚩

- Any JSON errors (middleware not working)
- Tool call failures
- Crashes or timeouts
- Incoherent responses

## Debugging Failed Evals

1. **Enable verbose logging**:
   ```bash
   PYTHONUNBUFFERED=1 python tests/evals/eval_multi_turn_research.py 2>&1 | tee eval.log
   ```

2. **Check conversation file**:
   - Saved to temp directory (printed at end)
   - Review for context loss or errors

3. **Examine metrics**:
   - Look for patterns in tool usage
   - Check which turn failed
   - Review error messages

4. **Run single turn**:
   ```bash
   python tests/evals/eval_multi_turn_research.py --turns 1
   ```

## Future Evals

Planned evaluations:
- [ ] Long-form report generation (tests large content handling)
- [ ] Multi-domain research (tests knowledge synthesis)
- [ ] Error recovery (tests resilience)
- [ ] Concurrent tool usage optimization (tests parallelism)
