# DuoNeural v3 MLX — eval harness

Eval harness for the DuoNeural v3 MLX release
(`DuoNeural/LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3` → MLX: BF16, 8-bit, 4-bit).

Scores the matrix — **BF16, 8-bit, 4-bit, and stock LiquidAI base** — on three components and
reports an **A/B delta vs the BF16 baseline** plus a **base-vs-published cross-check**:

1. **BFCL v3 multi-turn** (800 entries) — tool-calling anchor, official prompts + AST scoring.
2. **EvalPlus HumanEval + MBPP** (base and `+`) — pass@1, temperature 0.
3. **Custom Hermes FC suite** (~40 cases) — Hermes `<tool_call>` format, strict JSON-schema +
   exact function-name match, `<thought>` stripped before scoring.

## Layout

- `evals/` — the harness package (common runner contract + per-component adapters).
- `SETUP.md` — MBP environment setup (read this first).
- `requirements.txt` — pinned stack.

## Topology

The harness is developed via PRs from a separate build agent; **the user runs the eval matrix on the
M4 Pro MacBook Pro** (48 GB) which holds the artifacts and the HF token. See `SETUP.md`.

## Process

- **Wayfinder map:** issue #1 (destination + decisions index).
- **Never commit to `main`** — branch + PR, reviewed and merged by a maintainer.
- Prior effort (conversion + quantization): `phntmwvs/duoneural-v3-mlx`.
