# Hermes FC adapter — live smoke run record (ticket #19)

**Date:** 2026-10-08 · **Branch:** `feat/hermes-fc-adapter` (`2ca12dc`) · **Host:** M4 Pro (Dans-MacBook-Pro.local) · **Run dir:** `/Users/hermes/projects/hermes-fc-adapter-run` (rsync'd from the mini, dirty-tree-free)

## Artifact under test

| field | value |
|---|---|
| checkpoint | `/Users/hermes/projects/duoneural-v3-eval-harness/checkpoints/DuoNeural-v3-4bit` (DuoNeural-v3-4bit, 5.0 GB) |
| server | `mlx_lm server` 0.32.0 via shared `ServerManager` (ticket #16) |
| adapter venv | `.venv-core` python 3.12.15 |
| decoding | greedy — temperature 0.0, n=1, stream=false, max_tokens=512 |
| request shape | `POST /v1/chat/completions` with `tools=<case.tools[]>`, `tool_choice=auto`, `model=default_model` — the checkpoint's own `chat_template.jinja` renders the `<tools>` block (Q1) |

## Result (first full row)

| metric | value |
|---|---|
| **score** | **0.525 (21/40)** |
| runtime_s | 29.93 (0.75 s/case avg) |
| errors | 0/40 |
| single | 12/20 |
| parallel | 4/10 |
| negative | 4/5 |
| system2 | 1/5 |

Result JSON: `evals/results/hermes-4bit.json` (in the run dir; `evals/results/` is gitignored — this record is the committed provenance).

## Command

```bash
rsync -avz --exclude .git --exclude '.venv*' --exclude __pycache__ \
  --exclude checkpoints --exclude evals/results --exclude 'evals/*_runs' \
  duoneural-v3-eval-harness/ mbp:/Users/hermes/projects/hermes-fc-adapter-run/
ssh hermes@mbp
cd /Users/hermes/projects/hermes-fc-adapter-run && caffeinate -dims -t 1800 \
  /Users/hermes/projects/duoneural-v3-eval-harness/.venv-core/bin/python _smoke19.py
```

(`_smoke19.py` = `ServerManager` up → `evals.components.hermes.runner.run(...)` → teardown; driver deleted after the run.)

## Failure classification (21/40 → what's model vs. case vs. contract)

Scored by the locked rule; inspected each fail's predicted-vs-expected diff:

- **Model behavioral (the suite's signal):** fc-single-003/008/019 (wrong tool family — parse/filter vs. extract/transform), fc-single-013 (malformed JSONPath), fc-single-010 (empty `read_env` args), fc-system2-001 (timezone reasoning: 14:00 kept instead of 09:00 EST), fc-system2-002 (`/etc/localtime` vs `/etc/timezone`), fc-system2-003 (no `<thought>`), fc-parallel-003 (dropped `ssn` not `credit_card`), fc-parallel-006 (listed instead of deleted), fc-negative-002 (called a tool on a prose prompt).
- **Case-precision candidates (expected value arguably too strict — worth a case-review pass, not an adapter change):** fc-single-014 + fc-parallel-009 (model's `[d, d+1)` window `23:59:59` vs expected `00:00:00` next-day — semantically equal modulo the suite's `[start,end)` convention), fc-parallel-004 (same `23:59:59` pattern; the scheduling call matched), fc-single-007 (`web_search` query verbosity — none of 6 equivalents covers "…version number and release date"), fc-system2-005 (`ls -l | sort -k5 -nr` not in the 10-variant equivalents list).
- **Contract-tension candidate (needs oxy's call — do NOT silently change):** Q7 strictness vs. models emitting optional args. fc-single-005 + fc-system2-005 predicted the exact expected `command` string but added `confirm: true` (+ `timeout_seconds: 10`); Q7 ("predicted keys must equal expected keys, `null` included") fails them. fc-parallel-010: model added `confirm: true` to the expected `run_shell_command` call AND emitted 2 extra grounded calls (read_env, list_processes) — the extras are correctly ignored per Q6, but the modified call still fails Q7. If oxy wants "extra *grounded-schema-optional* keys ignored", that's a scoring-rule amendment to #19 Q7, not a bug fix.
- **fc-parallel-001:** `depends_on_order: true` case — model emitted only the first expected call (`copy_file`), never the confirmatory `read_file`; genuine model failure (no second call to score).

## Gate assessment

The ticket's resolution record is met: (1) adapter code on `feat/hermes-fc-adapter`; (2) the known-good-mock smoke passes 40/40 in unit tests (`test_smoke_all_cases_pass_on_known_good_completions` — includes the 5 calibration singles fc-single-001..005); (3) this file is the first full-row result record. The 21/40 live score is the 4-bit quant's real baseline for the A/B delta table (#7) — the BF16 row will say how much of the gap is quantization vs. the model itself.
