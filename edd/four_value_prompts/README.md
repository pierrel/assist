# Opt-in four-value calibration

`edd.four_value_codex` runs three independent assessors for Thorough, Focused,
and Correct, then a fourth call that combines their unchanged grades. It uses
the Codex CLI transport measured in the calibration records. It is an eval
entrypoint, not Assist's default judge or a production model setting.

From a source checkout with its test environment:

```sh
python -m edd.four_value_codex OBSERVATION.json PRIVATE_OUTPUT_DIR \
  --model gpt-6.1-sol --effort medium --codex-cwd PRIVATE_CODEX_WORKDIR
```

The caller must authorize sending the observation to the selected model. Use
public or approved redacted data, and run in an environment where readable files
are safe for provider access. The prompt forbids tools and the runner rejects
observed tool use afterward, but the CLI's read-only sandbox permits file reads
during the call. Rejection cannot undo a read. The output directory is new and owner-only;
the Codex CLI also retains its own local session record. Each role has a
180-second cap by default. A failed or incomplete role is recorded as a
nonverdict, with no retry or combined grade. The per-role folders retain the
prompt, answer, events, requested and observed model, usage, elapsed time,
and schema hash. The summary records the source observation hash. A grade's
evidence IDs are checked against the supplied observation; this check does not
prove the cited source entails the claim.

The measured Sol-medium results apply to this exact selected-evidence packet,
prompt and schema set. Other models, reasoning settings, longer traces, and
different evidence renderings need their own calibration. `Uncertain` is a
valid result when decisive evidence is missing. Timing and token use are
reported separately from Thorough, Focused, and Correct.

The invented fixture `edd/eval/fixtures/outcome-judge/four_value_synthetic.json`
contains twelve cases in six contrasts: effort/completeness, access-status
honesty, final correction, relevance, source pairing, and missing evidence.
The first four contrasts are tuning controls; source pairing and missing
evidence are held out. Host-authored expectations are separate from provider
inputs and are not human labels. Synthetic fixture success cannot establish
natural research reliability.
