#!/usr/bin/env python3
"""Per-venv eval orchestrator (wayfinder tickets #6 / #16).

Thin entry point that runs ONE component against ONE checkpoint:

    run_eval.py --component bfcl|evalplus|hermes --model <ref> [--resume]

Per ticket #6 (resolution 1a) the three components cannot share an interpreter
(``bfcl_eval`` pins ``tree_sitter==0.21.3``, ``evalplus`` needs ``>=0.22.0``),
so ``run_eval.py`` does NOT import any component in-process. It resolves the
component's per-venv python (``.venv-bfcl`` / ``.venv-evalplus`` / ``.venv-core``)
and subprocesses into that component's adapter entry point
(``python -m evals.components.<component>``). The adapter implements the
``ComponentRunner`` contract (``evals/runner.py``) and emits the normalized
result JSON (``evals/results.py``).

The components are ticketed separately (#17 BFCL, #18 EvalPlus, #19 Hermes):
this scaffold lands the per-venv subprocess wiring plus a contract-valid stub
emitter (``--emit-stub``) so the orchestration + result schema are proven
end-to-end before any adapter exists.

Stdlib-only and 3.9-compatible: it must run from any interpreter on the box.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evals import results as results_mod  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Component -> its dedicated venv (ticket #6, resolution 1a; SETUP.md §2).
# Hermes runs in the shared core venv (no extra deps — SETUP.md §5).
COMPONENT_VENV = {
    "bfcl": ".venv-bfcl",
    "evalplus": ".venv-evalplus",
    "hermes": ".venv-core",
}

# Component -> the ``-m`` module its adapter entry point is dispatched to.
# BFCL's real adapter (#17) is live; evalplus (#18) and hermes (#19) still run
# the scaffold stub (``evals.components``) until they land.
COMPONENT_ENTRY = {
    "bfcl": "evals.components.bfcl",
    "evalplus": "evals.components",
    "hermes": "evals.components",
}

# Entry points that take a leading --component selector (the scaffold stub
# dispatches on it; the real per-component adapters do not).
_NEEDS_COMPONENT_FLAG = {"evals.components"}


def venv_python(component: str, repo_root: str = REPO_ROOT) -> str:
    """Absolute path to the component's venv python."""
    return os.path.join(repo_root, COMPONENT_VENV[component], "bin", "python")


def build_component_command(
    component: str,
    model: str,
    *,
    model_name: str,
    resume: bool = False,
    repo_root: str = REPO_ROOT,
) -> list[str]:
    """Argv that runs the component's adapter in its own venv.

    Real adapters (#17/#18/#19) expose their own ``python -m
    evals.components.<component>`` ``__main__`` and parse ``--model`` /
    ``--model-name`` / ``--resume``. Components without a real adapter yet
    dispatch to the scaffold stub (``evals.components``) which takes a leading
    ``--component`` selector. ``model`` is the checkpoint ref (local dir or HF
    id); ``model_name`` is the distinct matrix row name that the result
    ``model`` field / filename uses.
    """
    entry = COMPONENT_ENTRY[component]
    argv = [venv_python(component, repo_root)]
    argv += ["-m", entry]
    if entry in _NEEDS_COMPONENT_FLAG:
        argv += ["--component", component]
    argv += ["--model", model, "--model-name", model_name]
    if resume:
        argv.append("--resume")
    return argv


def run_component(
    component: str,
    model: str,
    *,
    model_name: str | None = None,
    resume: bool = False,
    emit_stub: bool = False,
    dry_run: bool = False,
    repo_root: str = REPO_ROOT,
) -> int:
    """Dispatch one component in its venv; return its exit code.

    ``model`` is the checkpoint ref (``--model``); ``model_name`` is the matrix
    row name used for the result ``model`` field and filename — it defaults to
    ``model`` so callers that pass a row name as ``--model`` are unaffected,
    but an HF-id checkpoint should pass a slug-safe ``--model-name``.
    """
    if model_name is None:
        model_name = model
    if dry_run:
        argv = build_component_command(
            component, model, model_name=model_name, resume=resume, repo_root=repo_root
        )
        print("[run_eval] dry-run: {0}".format(" ".join(argv)))
        return 0
    # --emit-stub is a scaffold prove-out: emit in-process, no venv required.
    if emit_stub:
        return _emit_stub(component, model, model_name=model_name, repo_root=repo_root)
    argv = build_component_command(
        component, model, model_name=model_name, resume=resume, repo_root=repo_root
    )
    python = argv[0]
    if not os.path.exists(python):
        print(
            "[run_eval] ERROR: venv python not found: {0}\n"
            "  Create the {1} venv first — see SETUP.md §2.".format(python, COMPONENT_VENV[component]),
            file=sys.stderr,
        )
        return 2
    env = dict(os.environ, PYTHONPATH=repo_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(argv, cwd=repo_root, env=env)
    return proc.returncode


def _emit_stub(
    component: str, checkpoint: str, *, model_name: str, repo_root: str = REPO_ROOT
) -> int:
    """Write a contract-valid placeholder result (scaffold prove-out only).

    Stands in for a real adapter until #17/#18/#19 land, so the result schema
    and the write path are exercised without a component. ``score`` is a
    placeholder, NOT a measurement — real scores come from the adapters.
    ``model`` (the row name) and ``checkpoint`` (the ref) are kept distinct.
    """
    result = results_mod.new_result(
        component=component,
        model=model_name,
        checkpoint=checkpoint,
        score=0.0,
        subscores={"stub": True, "note": "placeholder from run_eval --emit-stub; not a real score"},
        runtime_s=0.0,
        artifact_versions={"run_eval": "scaffold", "python": sys.version.split()[0]},
    )
    out_dir = os.path.join(repo_root, "evals", "results")
    path = results_mod.write_result(result, out_dir)
    print("[run_eval] stub result written: {0}".format(path))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run one eval component against one checkpoint (ticket #16)."
    )
    p.add_argument(
        "--component",
        required=True,
        choices=list(COMPONENT_VENV),
        help="which component to run",
    )
    p.add_argument(
        "--model",
        required=True,
        help="checkpoint ref: local dir for the quants, HF id for the base row",
    )
    p.add_argument(
        "--model-name",
        default=None,
        help="matrix row name for the result model field / filename (e.g. "
        "'bf16', 'base'). Defaults to --model; pass a slug-safe name when "
        "--model is an HF id containing '/'.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run from persisted state (BFCL per-id resume)",
    )
    p.add_argument(
        "--emit-stub",
        action="store_true",
        help="write a contract-valid placeholder result (scaffold prove-out; not a real score)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the per-venv command that would run, without running it",
    )
    args = p.parse_args(argv)
    return run_component(
        args.component,
        args.model,
        model_name=args.model_name,
        resume=args.resume,
        emit_stub=args.emit_stub,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
