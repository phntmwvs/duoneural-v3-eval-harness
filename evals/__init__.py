"""DuoNeural v3 MLX eval harness.

One ``evals`` package, three components behind a common runner contract
(wayfinder map, issue #1 / Q2). The per-component logic lives in adapters;
``run_eval`` drives a single component, ``run_matrix`` loops the artifact set.

This module only defines the package surface and the runner-contract docstring.
The adapters, matrix orchestration, and reporting are implemented in the build
tickets (#6 harness architecture, then #3/#4/#5 component work).
"""

__version__ = "0.1.0"

#: The matrix of checkpoints every component runs over (wayfinder Q3).
#: BF16 is the A/B baseline; base is the stock LiquidAI reference row.
MATRIX_ROWS = ("bf16", "8bit", "4bit", "base")

#: The three scored components (wayfinder duoneural-v3-mlx #7).
COMPONENTS = ("bfcl", "evalplus", "hermes")
