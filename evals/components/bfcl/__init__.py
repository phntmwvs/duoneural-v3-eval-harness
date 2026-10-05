"""BFCL v3 multi-turn component adapter (ticket #17).

Folds the #3 spike (``spike/bfcl-osshandler``, PR #14) forward into the full
component adapter behind the ``ComponentRunner`` contract
(``evals/runner.py``): take a checkpoint, receive a live ``base_url`` from the
shared ``ServerManager`` (it never owns serve), drive the BFCL v3 multi-turn
categories, and emit the normalized result JSON (``evals/results.py``).

Layout (per ticket #6, resolution 1a — this component runs in ``.venv-bfcl``):

- ``handler.py`` — ``DuoNeuralV3FCHandler(QwenFCHandler)``, the spike's
  validated seam (``<thought>`` strip + served-id resolution from
  ``/v1/models``). Imports ``bfcl_eval`` — only importable inside
  ``.venv-bfcl``.
- ``register.py`` — runtime ``ModelConfig`` registration (no fork edit).
- ``runner.py`` — stdlib-only orchestration: serve lifecycle, ``bfcl
  generate`` + ``bfcl evaluate`` invocation, ``--resume`` skip, thrash-guard
  turn counting. Imports nothing from ``bfcl_eval`` so its pure helpers are
  testable in any interpreter.
- ``normalize.py`` — stdlib-only normalization of BFCL's score/result files
  into the harness result schema (``evals/results.py``).
- ``__main__.py`` — the entry point ``run_eval.py`` subprocesses into
  (``python -m evals.components.bfcl``).

The BFCL categories driven are the four multi-turn categories:
``multi_turn_base``, ``multi_turn_miss_func``, ``multi_turn_miss_param``,
``multi_turn_long_context``. Score = unweighted mean of their accuracies,
matching BFCL's own multi-turn aggregation.
"""

from __future__ import annotations

# bfcl_eval is NOT imported here — this package must be importable by tests in
# the core venv (no bfcl_eval). The handler/register modules import bfcl_eval
# lazily and are only exercised inside .venv-bfcl.
