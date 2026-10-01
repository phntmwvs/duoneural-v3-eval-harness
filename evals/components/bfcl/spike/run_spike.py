"""BFCL spike runner (wayfinder ticket #3) — PROTOTYPE, throwaway.

Drives the prove-gate for the BFCL integration:

    prompt -> <tool_call> decode -> tool-result re-render

over one BFCL multi-turn conversation (``multi_turn_base_0``, 4 turns) against a
live ``mlx_lm serve`` hosting DuoNeural v3 MLX.

What it does, in order:

1. Writes ``test_case_ids_to_generate.json`` (``{"multi_turn_base":
   ["multi_turn_base_0"]}``) into the BFCL project root so ``--run-ids``
   generates exactly one conversation.
2. Registers the DuoNeural v3 FC config in BFCL's model maps (runtime
   monkey-patch — no fork edit; see ``register.py``).
3. Invokes ``bfcl generate --model duoneural-v3-mlx-fc --run-ids
   --skip-server-setup`` in-process, with ``REMOTE_OPENAI_BASE_URL`` /
   ``REMOTE_OPENAI_TOKENIZER_PATH`` pointed at the already-running
   ``mlx_lm serve``.

Prereqs (all on the MBP, per SETUP.md):
  - ``.venv-bfcl`` active (``bfcl_eval==2026.3.23`` installed).
  - ``mlx_lm serve --model <checkpoint> --port 8080`` already running
    (``.venv-core`` provides mlx-lm). Smallest row is fine for the spike —
    this tests the plumbing, not the score.

Usage:
  .venv-bfcl/bin/python -m evals.components.bfcl.spike.run_spike \
      --tokenizer-path checkpoints/DuoNeural-v3-4bit \
      [--base-url http://localhost:8080/v1] [--test-id multi_turn_base_0]

Go/no-go and the exact failure (if any) are printed at the end and recorded on
ticket #3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from bfcl_eval.constants.eval_config import (
    PROJECT_ROOT,
    TEST_IDS_TO_GENERATE_PATH,
)

from .register import register

DEFAULT_TEST_ID = "multi_turn_base_0"
DEFAULT_CATEGORY = "multi_turn_base"


def _write_ids_file(test_id: str, category: str) -> Path:
    """Write the single-id file BFCL's --run-ids reads."""
    ids = {category: [test_id]}
    TEST_IDS_TO_GENERATE_PATH.write_text(json.dumps(ids, indent=2))
    print(f"[spike] wrote {TEST_IDS_TO_GENERATE_PATH}: {ids}")
    return TEST_IDS_TO_GENERATE_PATH


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="BFCL OSSHandler spike (ticket #3)")
    p.add_argument(
        "--base-url",
        default=os.getenv("REMOTE_OPENAI_BASE_URL", "http://localhost:8080/v1"),
        help="mlx_lm serve OpenAI base URL (default %(default)s)",
    )
    p.add_argument(
        "--tokenizer-path",
        default=os.getenv("REMOTE_OPENAI_TOKENIZER_PATH"),
        required=os.getenv("REMOTE_OPENAI_TOKENIZER_PATH") is None,
        help="Local checkpoint dir holding config.json/tokenizer (required unless "
        "REMOTE_OPENAI_TOKENIZER_PATH is set)",
    )
    p.add_argument("--test-id", default=DEFAULT_TEST_ID)
    p.add_argument("--category", default=DEFAULT_CATEGORY)
    p.add_argument("--port-note", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    # --- point BFCL at the already-running mlx_lm serve ----------------------
    os.environ["REMOTE_OPENAI_BASE_URL"] = args.base_url
    os.environ["REMOTE_OPENAI_TOKENIZER_PATH"] = str(args.tokenizer_path)
    os.environ.setdefault("REMOTE_OPENAI_API_KEY", "EMPTY")

    model_key = register()
    print(f"[spike] registered model key: {model_key}")
    print(f"[spike] REMOTE_OPENAI_BASE_URL={args.base_url}")
    print(f"[spike] REMOTE_OPENAI_TOKENIZER_PATH={args.tokenizer_path}")
    print(f"[spike] BFCL PROJECT_ROOT={PROJECT_ROOT}")

    _write_ids_file(args.test_id, args.category)

    # --- invoke `bfcl generate` in-process ----------------------------------
    # `cli` is a typer.Typer() (console_scripts: bfcl = bfcl_eval.__main__:cli).
    # A typer app isn't callable with args=; build the click command and invoke it.
    import typer
    from bfcl_eval.__main__ import cli

    click_cmd = typer.main.get_command(cli)
    cli_args = [
        "generate",
        "--model",
        model_key,
        "--test-category",
        args.category,
        "--run-ids",
        "--skip-server-setup",
        "--allow-overwrite",
        "--include-input-log",
    ]
    print(f"[spike] invoking: bfcl {' '.join(cli_args)}")
    try:
        click_cmd(args=cli_args, standalone_mode=False)
    except SystemExit as e:  # click/typer may sys.exit(0)
        code = e.code if isinstance(e.code, int) else 0
        if code != 0:
            raise

    result_dir = PROJECT_ROOT / "result" / model_key.replace("/", "_")
    print("\n[spike] DONE. Result tree:", result_dir)
    print(f"[spike] Inspect the multi-turn log under: {result_dir}")
    print("[spike] GO if the log shows all 4 turns with decoded <tool_call> and "
          "re-rendered <tool_response>; otherwise record the exact failure on #3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
