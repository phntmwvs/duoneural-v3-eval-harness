# SETUP — DuoNeural v3 MLX eval harness

Environment setup for running the eval matrix on the **M4 Pro MacBook Pro**.
The Foundry agent builds this repo via PRs from the mini; **oxy runs the matrix on the MBP**
(see the wayfinder map, issue #1). All commands below run **on the MBP**.

## 0. Prerequisites

- **Apple silicon Mac with enough unified memory** for the BF16 row (~17 GB weights + runtime
  headroom). The M4 Pro (48 GB) is the target; a 16 GB machine cannot hold BF16.
- **Homebrew** on PATH (`brew --version` should succeed).
- **Python ≥ 3.10.** BFCL requires it. **Do NOT use Apple's system Python**
  (`/usr/bin/python3` = 3.9.6 — too old, and `mlx-lm`/`evalplus` will fail against it).

## 1. Interpreter: use Homebrew Python 3.14 by absolute path

The MBP has `python@3.14` (3.14.8) via brew. Bare `python3` resolves to Apple's 3.9 system
Python — **never rely on `python3` by name.** Pin the interpreter by absolute path:

```bash
PY=/opt/homebrew/opt/python@3.14/bin/python3.14
ls -l "$PY"            # must exist; if not: brew install python@3.14
"$PY" --version        # Python 3.14.x
```

> **Why 3.14?** It's what's installed and modern. It's bleeding-edge — if a pinned dep below
> fails to build a wheel on 3.14, fall back to a mature version:
> `brew install python@3.12` and set `PY=/opt/homebrew/opt/python@3.12/bin/python3.12`,
> then recreate the venv. Don't fight a broken 3.14 build; drop down.

## 2. Clone + venv

```bash
git clone https://github.com/phntmwvs/duoneural-v3-eval-harness.git
cd duoneural-v3-eval-harness
PY=/opt/homebrew/opt/python@3.14/bin/python3.14
"$PY" -m venv .venv
source .venv/bin/activate
python --version       # must read 3.14.x (NOT 3.9.x)
which python           # must be .../duoneural-v3-eval-harness/.venv/bin/python
```

The venv is **activated per-session** (`source .venv/bin/activate`), not added to PATH globally.
Inside the activated venv, `python`/`pip` are the venv's and Apple's 3.9 is unreachable.

## 3. Install the pinned stack

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then verify the three import anchors:

```bash
python -c "import mlx_lm; print('mlx_lm', mlx_lm.__version__)"
python -c "import datasets; print('datasets', datasets.__version__)"
python -c "import evalplus; print('evalplus ok')"
python -c "import bfcl_eval; print('bfcl_eval ok')"
```

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
  Pinned to `bfcl_eval==2026.3.23`. See ticket #3 (spike) before the full build.
- **EvalPlus HumanEval+MBPP** — pass@1, temperature 0, against `mlx_lm serve`
  (`--backend openai --base-url`). Sandbox via the official `ganler/evalplus` Docker image
  **if Docker is present** (`docker info` responds), else a local venv with resource limits.
- **Custom Hermes FC suite** — ~40 hand-authored cases, strict JSON-schema + exact
  function-name match, `<thought>` stripped before scoring. See ticket #4.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `python3 --version` → 3.9.6 | You're hitting Apple's system Python. Use the absolute brew path (§1). |
| `command not found: brew` | `eval "$(/opt/homebrew/bin/brew shellenv)"` (add to `~/.zprofile`). |
| `pip install` build error on 3.14 | Fall back to `python@3.12` (§1 note); recreate the venv. |
| `import bfcl_eval` fails | The pinned fork isn't installed; re-run §3 and check `requirements.txt`. |
