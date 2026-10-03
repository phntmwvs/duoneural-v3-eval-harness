"""BFCL component entry point (ticket #17).

``run_eval.py`` dispatches to ``python -m evals.components.bfcl`` inside
``.venv-bfcl``. This parses the shared adapter flags (``--model`` /
``--model-name`` / ``--resume``) and drives the full BFCL v3 multi-turn
pipeline: serve (if no ``--base-url``) → generate → evaluate → normalize →
write the harness result JSON.

    .venv-bfcl/bin/python -m evals.components.bfcl \
        --model checkpoints/DuoNeural-v3-4bit --model-name 4bit [--resume]

``--base-url`` lets the matrix (ticket #20) hand in an already-running server
instead of this component owning one.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evals.components.bfcl import normalize, runner  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="BFCL v3 multi-turn adapter (ticket #17)")
    p.add_argument("--model", required=True,
                   help="checkpoint ref (local dir for the quants, HF id for base)")
    p.add_argument("--model-name", default=None,
                   help="matrix row name for the result model field / filename; "
                        "defaults to --model")
    p.add_argument("--base-url", default=None,
                   help="use an already-running mlx_lm server (matrix mode); "
                        "when omitted this component serves --model itself")
    p.add_argument("--resume", action="store_true",
                   help="skip categories whose result/<model>/ file is complete "
                        "(ticket #6, decision 7a)")
    p.add_argument("--categories", default=",".join(normalize.MULTI_TURN_CATEGORIES),
                   help="comma-separated BFCL categories (default: the 4 multi-turn)")
    args = p.parse_args(argv)

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    result = runner.run(
        args.model,
        base_url=args.base_url,
        model_name=args.model_name,
        resume=args.resume,
        categories=tuple(categories),
    )
    out = os.path.join(
        _REPO_ROOT, "evals", "results",
        "bfcl-{0}.json".format(args.model_name or args.model),
    )
    print("[component:bfcl] score={0:.4f} thrash_flag={1} missing={2}".format(
        result["score"],
        result["subscores"]["thrash"]["flag"],
        result["subscores"]["missing_categories"],
    ))
    print("[component:bfcl] result written under evals/results/ ({0})".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
