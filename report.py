#!/usr/bin/env python3
"""``report.py`` — DuoNeural v3 MLX eval report CLI (wayfinder ticket #7).

Thin wrapper over ``evals.report``; lets the user run::

    ./report.py --summary-json evals/results/summary.json

and get the rendered Markdown on stdout (or written via ``--output``).
The report consumes ``run_matrix.py --summary-json`` as the source of
truth — no re-derivation.

Stdlib-only and 3.9-compatible (matches the matrix's constraints).
"""

from __future__ import annotations

import os
import sys

# Repo-root import for ``evals``.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evals import report as _report  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(_report.main())