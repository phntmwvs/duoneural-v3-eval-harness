# Hermes FC suite — System-2 `` emission probe

**Ticket:** #12 (author 40 Hermes FC cases + probe `` emission on the MBP).

**Why this exists:** the System-2 category's scoring rule (#4) requires the
model to emit a ``<thought>...</thought>`` block before its tool call.
If the v3 model never emits ``<thought>`` under any reasonable
instruction, the category always scores 0 and the suite needs a
redesign before more system2 cases are authored. This probe is the
gate.

## What it does

`probe_thought_emission.py` sends one system2 case (`fc-system2-001`) N times
to an `mlx_lm server` instance owned by the shared `ServerManager`
(ticket #16), then reports:

- how often the model emitted a ``<thought>`` block,
- how often it emitted a ``<tool_call>``,
- how often the call JSON parses,
- the full pass rate (thought + call + decodes).

Greedy decoding (temperature=0, n=1) — we want a reproducible pass-rate
signal, not sampling diversity. Exit code 0 iff the pass rate meets the
gate (default: 10/10).

## How to run it (on the M4 Pro)

The probe needs the `mlx_lm` venv (`/.venv-core/bin/python`). The
mini can't load the model — the M4 Pro holds the artifacts.

```bash
# 1. Anti-sleep: prevents the MBP from napping during long generations.
caffeinate -dims -t 1800 &

# 2. Pull the latest on the probe branch.
cd ~/workspace/duoneural-v3-eval-harness
git checkout feat/hermes-fc-thought-probe
git pull

# 3. Run on the 4-bit quant (worst case for `` emission).
./.venv-core/bin/python evals/hermes_fc/probe_thought_emission.py \
    --checkpoint checkpoints/DuoNeural-v3-4bit \
    --case evals/hermes_fc/cases/fc-system2-001.json \
    --n 10 \
    --log-file /tmp/probe-4bit.log

# 4. (Optional) Pin to BF16 (gold reference) for a second data point.
./.venv-core/bin/python evals/hermes_fc/probe_thought_emission.py \
    --checkpoint checkpoints/DuoNeural-v3-BF16 \
    --case evals/hermes_fc/cases/fc-system2-001.json \
    --n 10 \
    --log-file /tmp/probe-bf16.log
```

BF16 startup needs ~17 GB of unified memory (the M4 Pro has 48 GB, so
fine). 4-bit needs ~5 GB. Server readiness is polled for up to 300 s
(default); override with `--startup-timeout-s` if your checkpoint is on
slow disk.

## Interpreting the result

- **GATE CLEARED** (10/10): safe to author `fc-system2-002..005` and
  keep the existing scoring rule from #4.
- **GATE NOT CLEARED** with `` present < 100%: the model emits
  `` inconsistently. The scoring rule from #4 will penalize every
  miss. Two options: (a) drop the System-2 category from v1.0 and
  document the v3 model's behavior, or (b) loosen the scoring rule
  (e.g. `` optional, thought content is graded for quality not just
  presence). Either way this is a #4-amendment conversation, not a
  cases-only edit.
- **GATE NOT CLEARED** with 0/10: the model never emits `` at all
  on this case with the stock template. Re-run with the system
  override in `fc-system2-001.system` set to "Use ``<thought>...</thought>``
  to reason step-by-step before emitting your tool call" and see if it
  picks it up. If still 0/10, the category is unsalvageable for v1.0.

## Files

- `cases/fc-system2-001.json` — the probe case (will move to the
  cases branch when the system2 category batch is authored).
- `probe_thought_emission.py` — the probe script. Stdlib-only; reuses
  `evals.server.ServerManager` for serve lifecycle.
- `../tests/test_hermes_probe.py` — unit tests for the pure helpers
  (`analyze`, `build_payload`); does not exercise the network path
  (the live run is MBP-only).
