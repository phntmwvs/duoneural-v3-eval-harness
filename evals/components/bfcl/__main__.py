"""BFCL component entry point (ticket #17).

``run_eval.py`` dispatches to ``python -m evals.components.bfcl`` inside
``.venv-bfcl``. This parses the shared adapter flags (``--model`` /
``--model-name`` / ``--resume``) and drives the full BFCL v3 multi-turn
pipeline against the live ``mlx_lm server`` handed to it via ``--base-url``:
generate → evaluate → normalize → write the harness result JSON.

Per the runner contract (``evals/runner.py``) the adapter never owns serve:
the shared ``ServerManager`` (ticket #16) / matrix (ticket #20) brings the
server up and passes its ``--base-url``.

    .venv-bfcl/bin/python -m evals.components.bfcl \
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
from evals.components.bfcl import normalize, runner  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="BFCL v3 multi-turn adapter (ticket #17)")
    p.add_argument("--model", required=True,
                   help="checkpoint ref (local dir for the quants, HF id for base)")
    p.add_argument("--model-name", default=None,
                   help="matrix row name for the result model field / filename; "
                        "defaults to --model")
    p.add_argument("--base-url", required=True,
                   help="live mlx_lm server OpenAI base URL handed in by the "
                        "shared ServerManager / matrix (the adapter never serves)")
    p.add_argument("--resume", action="store_true",
                   help="skip ids already complete in result/<model>/ "
                        "(BFCL native per-id resume; ticket #6, decision 7a)")
    p.add_argument("--categories", default=",".join(normalize.MULTI_TURN_CATEGORIES),
                   help="comma-separated BFCL categories (default: the 4 multi-turn)")
    p.add_argument("--num-threads", type=int, default=runner.DEFAULT_NUM_THREADS,
                   help="concurrent inference threads for bfcl generate "
                        "(default %(default)s = serial; BFCL's own default is "
                        "100, which the single-threaded mlx_lm server cannot absorb)")
    args = p.parse_args(argv)

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    result = runner.run(
        args.model,
        base_url=args.base_url,
        model_name=args.model_name,
        resume=args.resume,
        categories=tuple(categories),
        num_threads=args.num_threads,
    )
    path = results_mod.result_path(
        os.path.join(_REPO_ROOT, "evals", "results"),
        result["component"],
        result["model"],
    )
    print("[component:bfcl] score={0:.4f} complete={1} thrash_flag={2} "
          "missing={3} partial={4}".format(
        result["score"],
        result["subscores"]["complete"],
        result["subscores"]["thrash"]["flag"],
        result["subscores"]["missing_categories"],
        result["subscores"]["partial_categories"],
    ))
    print("[component:bfcl] result written: {0}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
