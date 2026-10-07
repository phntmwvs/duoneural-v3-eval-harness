"""EvalPlus component entry point (ticket #18).

``run_eval.py`` dispatches to ``python -m evals.components.evalplus`` inside
``.venv-evalplus``. This parses the shared adapter flags (``--model`` /
``--model-name`` / ``--resume``) and drives the full EvalPlus pipeline against
the live ``mlx_lm server`` handed to it via ``--base-url``: generate (host) →
sandboxed evaluate (Docker) → normalize → write the harness result JSON.

Per the runner contract (``evals/runner.py``) the adapter never owns serve:
the shared ``ServerManager`` (ticket #16) / matrix (ticket #20) brings the
server up and passes its ``--base-url``.

    .venv-evalplus/bin/python -m evals.components.evalplus \
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
from evals.components.evalplus import runner  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="EvalPlus HumanEval+MBPP adapter (ticket #18)")
    p.add_argument("--model", required=True,
                   help="checkpoint ref (local dir for the quants, HF id for base)")
    p.add_argument("--model-name", default=None,
                   help="matrix row name for the result model field / filename; "
                        "defaults to --model")
    p.add_argument("--base-url", required=True,
                   help="live mlx_lm server OpenAI base URL handed in by the "
                        "shared ServerManager / matrix (the adapter never serves)")
    p.add_argument("--resume", action="store_true",
                   help="append to a partial sample file (codegen --resume) "
                        "instead of regenerating (ticket #6, decision 7a)")
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
    print("[component:evalplus] score={0:.4f} complete={1} missing={2} partial={3}".format(
        result["score"],
        result["subscores"]["complete"],
        result["subscores"]["missing_datasets"],
        result["subscores"]["partial_datasets"],
    ))
    print("[component:evalplus] result written: {0}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
