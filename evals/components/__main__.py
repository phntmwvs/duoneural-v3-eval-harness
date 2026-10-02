"""Component entry point for the scaffold (ticket #16).

``run_eval.py`` dispatches to ``python -m evals.components.<component>``. The
real adapters are ticketed separately (#17 BFCL, #18 EvalPlus, #19 Hermes);
until they land this entry point emits a contract-valid stub result so the
per-venv subprocess wiring and the result schema are proven end-to-end.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running both as ``python -m evals.components`` and, in tests, directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evals import results as results_mod  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Component adapter stub (ticket #16)")
    p.add_argument("--component", required=True, choices=list(results_mod.COMPONENTS))
    p.add_argument("--model", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)

    result = results_mod.new_result(
        component=args.component,
        model=args.model,
        checkpoint=args.model,
        score=0.0,
        subscores={
            "stub": True,
            "resume": bool(args.resume),
            "note": "placeholder from the component stub; not a real score",
        },
        runtime_s=0.0,
        artifact_versions={"component_stub": "scaffold", "python": sys.version.split()[0]},
    )
    path = results_mod.write_result(result, os.path.join(_REPO_ROOT, "evals", "results"))
    print("[component:{0}] stub result written: {1}".format(args.component, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
