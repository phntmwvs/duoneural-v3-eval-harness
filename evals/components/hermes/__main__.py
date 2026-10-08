"""Hermes FC suite component entry point (ticket #19).

``run_eval.py`` dispatches to ``python -m evals.components.hermes`` inside
``.venv-core``. This parses the shared adapter flags (``--model`` /
``--model-name`` / ``--base-url`` / ``--resume``) and runs the 40 hand-
authored cases against the live ``mlx_lm server`` handed to it via
``--base-url``: one greedy chat+tools call per case, one-set subset
scoring, then the normalized result JSON.

Per the runner contract (``evals/runner.py``) the adapter never owns serve:
the shared ``ServerManager`` (ticket #16) / matrix (ticket #20) brings the
server up and passes its ``--base-url``.

    .venv-core/bin/python -m evals.components.hermes \
        --model checkpoints/DuoNeural-v3-4bit --model-name 4bit \
        --base-url http://127.0.0.1:8080/v1 [--resume]
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

from evals import results as results_mod  # noqa: E402
from evals.components.hermes import runner  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Hermes FC suite adapter (ticket #19)")
    p.add_argument("--model", required=True,
                   help="checkpoint ref (local dir for the quants, HF id for base)")
    p.add_argument("--model-name", default=None,
                   help="matrix row name for the result model field / filename; "
                        "defaults to --model")
    p.add_argument("--base-url", required=True,
                   help="live mlx_lm server OpenAI base URL handed in by the "
                        "shared ServerManager / matrix (the adapter never serves)")
    p.add_argument("--resume", action="store_true",
                   help="keep per-case results already in state.json and run "
                        "only the missing cases (ticket #6, decision 7a)")
    args = p.parse_args(argv)

    result = runner.run(
        args.model,
        base_url=args.base_url,
        model_name=args.model_name,
        resume=args.resume,
    )
    path = results_mod.result_path(
        os.path.join(_REPO_ROOT, "evals", "results"),
        result["component"],
        result["model"],
    )
    per_cat = result["subscores"]["per_category"]
    print("[component:hermes] score={0:.4f} ({1}/{2} passed, {3} errors)".format(
        result["score"],
        result["subscores"]["n_passed"],
        result["subscores"]["n_cases"],
        result["subscores"]["n_errors"],
    ))
    for cat in sorted(per_cat):
        print("[component:hermes]   {0}: {1}/{2}".format(
            cat, per_cat[cat]["passed"], per_cat[cat]["n"]))
    print("[component:hermes] result written: {0}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
