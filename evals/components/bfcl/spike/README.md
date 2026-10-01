# BFCL OSSHandler spike — PROTOTYPE (wayfinder ticket #3)

Throwaway prototype that answers ticket #3's question:

> Can a thin `OSSHandler` subclass serve DuoNeural v3 over `mlx_lm serve` and
> complete one BFCL multi-turn conversation end-to-end
> (prompt → `<tool_call>` decode → tool-result re-render)?

**Runs on the M4 Pro MBP** (the mini can't hold any matrix row). This is a
spike — it de-risks the BFCL integration before the full handler build (ticket
#6). Proves the plumbing; the score is irrelevant here.

## Files

- `handler.py` — `DuoNeuralV3FCHandler(QwenFCHandler)`. Same
  `<|im_start|>`/`<|im_end|>` + `<tool_call>`/`<tool_response>` template as
  QwenFC; strips `<thought>` (not `<think>`) before `<tool_call>` decode.
  Inherits the completions-API path (`client.completions.create` →
  `REMOTE_OPENAI_BASE_URL`) and remote tokenizer loading unchanged.
- `register.py` — inserts a `ModelConfig` into `MODEL_CONFIG_MAPPING` and
  appends to `SUPPORTED_MODELS` at import time. **Runtime monkey-patch — no
  fork edit.** Model key: `duoneural-v3-mlx-fc`.
- `run_spike.py` — writes the single-id file, registers, and drives
  `bfcl generate --run-ids --skip-server-setup` in-process for one
  `multi_turn_base` conversation (`multi_turn_base_0`, 4 turns).

## Prerequisites (MBP, per SETUP.md)

1. `.venv-bfcl` and `.venv-core` built (`requirements-bfcl.txt` /
   `requirements-core.txt`). `requirements-bfcl.txt` now also pins `soundfile`
   — an **undeclared transitive dep** of `bfcl_eval` (via `qwen-agent`) without
   which `import bfcl_eval.constants.model_config` dies with
   `ModuleNotFoundError: No module named 'soundfile'`. Re-run
   `.venv-bfcl/bin/pip install -r requirements-bfcl.txt` if your venv predates
   the pin.
2. A checkpoint pulled (smallest row is fine — this tests plumbing):
   ```bash
   huggingface-cli download phntmwvs/DuoNeural-v3-4bit-MLX \
       --local-dir checkpoints/DuoNeural-v3-4bit
   ```

## Run

Two shells on the MBP (or run the server in the background):

```bash
# 1. serve the checkpoint (core venv provides mlx_lm).
#    NOTE: the pip mlx-lm subcommand is `server`, not `serve`
#    (the brew formula's binary aliases `serve`; the venv console-script does not).
.venv-core/bin/mlx_lm server --model checkpoints/DuoNeural-v3-4bit --port 8080 &

# 2. wait for it to come up, then run the spike
.venv-bfcl/bin/python -m evals.components.bfcl.spike.run_spike \
    --tokenizer-path checkpoints/DuoNeural-v3-4bit \
    --base-url http://localhost:8080/v1
```

To run a different conversation: `--test-id multi_turn_base_5`.

## Go / No-go

- **GO** — the result log under `result/duoneural-v3-mlx-fc/` shows all 4 turns
  of `multi_turn_base_0` with a decoded `<tool_call>` and a re-rendered
  `<tool_response>` each step, no decode exception. → proceed to the full BFCL
  handler build (#6).
- **NO-GO** — record the exact failure on ticket #3: fork-edit mismatch,
  completions-API gap, or `<tool_call>`/`<tool_response>` rendering mismatch.

Record the outcome (link this branch) as the resolution comment on ticket #3.
