# SETUP — DuoNeural v3 MLX eval harness

Environment setup for running the eval matrix on the **M4 Pro MacBook Pro**.
The harness is developed via PRs from a separate build agent; **the user runs the matrix on the MBP**
(see the wayfinder map, issue #1). All commands below run **on the MBP**.

## 0. Prerequisites

- **Apple silicon Mac with enough unified memory** for the BF16 row (~17 GB weights + runtime
  headroom). The M4 Pro (48 GB) is the target; a 16 GB machine cannot hold BF16.
- **Homebrew** on PATH (`brew --version` should succeed).
- **Python 3.12.** BFCL requires ≥ 3.10, but the eval stack's pinned deps (`tree-sitter`,
  and others) ship **no cp314 wheels** — so the modern-but-mature **3.12** is the target.
  **Do NOT use Apple's system Python** (`/usr/bin/python3` = 3.9.6 — too old), and do **not**
  use 3.14 for this stack (missing prebuilt wheels → source builds that need a C toolchain).

## 1. Interpreter: Homebrew Python 3.12 by absolute path

Install 3.12 if needed, then pin it by absolute path (never bare `python3`, which resolves
to Apple's 3.9):

```bash
brew install python@3.12
PY=/opt/homebrew/opt/python@3.12/bin/python3.12
ls -l "$PY"            # must exist
"$PY" --version        # Python 3.12.x
```

> **Why not 3.14 (which the MBP already has)?** `tree-sitter` 0.21.x / 0.22.x (required by
> bfcl_eval and evalplus) publish wheels only up to **cp312**. On 3.14 pip falls back to a
> source build that needs a working C toolchain and frequently fails. 3.12 has prebuilt
> wheels for the entire pinned stack, so installs are fast and reliable.

## 2. Three venvs — the components do NOT share an environment

`bfcl_eval` pins `tree_sitter==0.21.3` and `evalplus` needs `tree-sitter>=0.22.0` —
**they cannot coexist in one venv** (pip `ResolutionImpossible`). The harness is three
independent components behind a common runner contract (wayfinder Q2), so each gets its own
venv. The **core** venv (mlx-lm + `datasets`) is the shared generation/serving layer the
components drive.

```bash
cd duoneural-v3-eval-harness
PY=/opt/homebrew/opt/python@3.12/bin/python3.12

# Core: mlx-lm serving + datasets (drives mlx_lm serve for all components)
"$PY" -m venv .venv-core
.venv-core/bin/python -m pip install --upgrade pip
.venv-core/bin/python -m pip install -r requirements-core.txt

# BFCL component
"$PY" -m venv .venv-bfcl
.venv-bfcl/bin/python -m pip install --upgrade pip
.venv-bfcl/bin/python -m pip install -r requirements-bfcl.txt

# EvalPlus component
"$PY" -m venv .venv-evalplus
.venv-evalplus/bin/python -m pip install --upgrade pip
.venv-evalplus/bin/python -m pip install -r requirements-evalplus.txt
```

`.venv-*/` are gitignored. Activate the one you're working in (`source .venv-core/bin/activate`),
or call the venv pythons directly (the matrix runner does the latter — no activation needed).

## 3. Verify the import anchors

```bash
.venv-core/bin/python     -c "import mlx_lm; print('mlx_lm', mlx_lm.__version__)"
.venv-core/bin/python     -c "import datasets; print('datasets', datasets.__version__)"
.venv-bfcl/bin/python     -c "import bfcl_eval; print('bfcl_eval ok')"
.venv-evalplus/bin/python -c "import evalplus; print('evalplus ok')"
```

All four should print cleanly. A `tree-sitter` resolution or build error means you're on the
wrong interpreter (3.14 or system 3.9) — recheck §1.

## 4. Checkpoints

Matrix rows (wayfinder map, Q3): **BF16, 8-bit, 4-bit, and stock LiquidAI base.**

- **3 quant artifacts** — pushed to `phntmwvs/*-MLX` (private) per ticket #2 (PUBLISH GATE).
  Pull by repo ID:
  ```bash
  huggingface-cli download phntmwvs/DuoNeural-v3-BF16-MLX --local-dir checkpoints/DuoNeural-v3-BF16
  huggingface-cli download phntmwvs/DuoNeural-v3-8bit-MLX --local-dir checkpoints/DuoNeural-v3-8bit
  huggingface-cli download phntmwvs/DuoNeural-v3-4bit-MLX --local-dir checkpoints/DuoNeural-v3-4bit
  ```
- **Stock base** — `LiquidAI/LFM2.5-8B-A1B` loads in mlx-lm **directly (no conversion)**;
  `mlx_lm serve --model LiquidAI/LFM2.5-8B-A1B` pulls it on first run (~17 GB).

`checkpoints/` is gitignored.

## 5. Component notes

- **BFCL v3 multi-turn** — thin `OSSHandler` subclass (modeled on `qwen_fc.py`), registered in
  `bfcl_eval/constants/model_config.py`; run `bfcl generate --model <name> --skip-server-setup`
  against `mlx_lm serve` via `REMOTE_OPENAI_BASE_URL` + `REMOTE_OPENAI_TOKENIZER_PATH`.
  Pinned to `bfcl_eval==2026.3.23`. Runs in `.venv-bfcl`. See ticket #3 (spike) before the full build.
- **EvalPlus HumanEval+MBPP** — pass@1, temperature 0, against `mlx_lm serve`
  (`--backend openai --base-url`). Runs in `.venv-evalplus`. Sandbox generated-code execution via
  the official `ganler/evalplus` Docker image **if Docker is present** (`docker info` responds),
  else a local run with resource limits.
- **Custom Hermes FC suite** — ~40 hand-authored cases, strict JSON-schema + exact
  function-name match, `<thought>` stripped before scoring. Runs in `.venv-core` (no extra deps).
  See ticket #4.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `python3 --version` → 3.9.6 | You're hitting Apple's system Python. Use the absolute brew 3.12 path (§1). |
| `ResolutionImpossible: tree-sitter` / `tree_sitter` | You put bfcl_eval + evalplus in one venv. Use the three-venv layout (§2). |
| `tree-sitter` has no wheel / tries to build from source | You're on Python 3.14. Switch to 3.12 (§1). |
| `command not found: brew` | `eval "$(/opt/homebrew/bin/brew shellenv)"` (add to `~/.zprofile`). |
| `import bfcl_eval` fails | You're not in `.venv-bfcl`. Use the venv's python directly (§3). |
| `No module named 'datasets'` | mlx-lm has no `datasets` extra (it's under `train`). Re-run §2 core install — `requirements-core.txt` pins `datasets` explicitly. |
