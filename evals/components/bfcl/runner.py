"""BFCL v3 multi-turn runner — orchestration (ticket #17).

Stdlib-only: this module is importable and unit-testable in any interpreter
(it shells out to ``bfcl``; it never imports ``bfcl_eval``). The
``bfcl_eval``-dependent seam lives in ``handler.py`` / ``register.py`` and is
only exercised inside ``.venv-bfcl``.

Responsibilities:

1. Receive the live ``mlx_lm server`` ``base_url`` from the shared
   ``ServerManager`` (ticket #16) — the adapter never owns serve — and hand it
   to BFCL through ``REMOTE_OPENAI_BASE_URL``.
2. Register ``DuoNeuralV3FCHandler`` at runtime and drive ``bfcl generate``
   for the four multi-turn categories, then ``bfcl evaluate``.
3. ``--resume`` (ticket #6, decision 7a): omit ``--allow-overwrite`` so BFCL's
   native per-id skip regenerates only the ids missing from each category's
   ``result/<model>/`` file.
4. Thrash-loop guard (#3 hardening): read per-turn step counts from the
   multi-turn result logs and surface them in ``subscores.thrash``; a turn
   that exceeds BFCL's step cap (never produced a terminal answer) is flagged,
   not aborted — the matrix/report decides.

BFCL is invoked in-process (``typer.main.get_command`` on
``bfcl_eval.__main__:cli``) exactly as the #3 spike did, but from this
component's ``.venv-bfcl`` interpreter via ``__main__.py``.
"""

from __future__ import annotations

import os
import sys
import time

# Allow running as ``python -m evals.components.bfcl`` and under tests.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evals import results as results_mod  # noqa: E402

from . import normalize  # noqa: E402

COMPONENT = "bfcl"
#: Generation steps within one conversational turn at which BFCL force-quits
#: the turn ("Model has been forced to quit after 20 steps"). A turn whose
#: step count exceeds this (21 = 20 attempts + the forced terminal message)
#: never produced a terminal answer — the thrash signal. Flag, not abort.
THRASH_TURN_CAP = 20


def build_generate_argv(model_key: str, categories, *,
                        allow_overwrite=False, result_dir=None):
    """Argv (after ``bfcl``) for one ``bfcl generate`` invocation.

    ``categories`` is comma-joined (BFCL's ``handle_multiple_input`` splits on
    commas). ``--skip-server-setup`` keeps BFCL from spawning its own server —
    the shared ``ServerManager`` owns it. ``--allow-overwrite`` regenerates all
    ids (fresh run); omitting it lets BFCL's native per-id resume regenerate
    only missing ids (ticket #6, decision 7a).
    """
    argv = [
        "generate",
        "--model", model_key,
        "--test-category", ",".join(categories),
        "--skip-server-setup",
        "--include-input-log",
    ]
    if allow_overwrite:
        argv.append("--allow-overwrite")
    if result_dir is not None:
        argv += ["--result-dir", str(result_dir)]
    return argv


def build_evaluate_argv(model_key: str, categories, *, result_dir=None,
                        score_dir=None, partial_eval=True):
    """Argv (after ``bfcl``) for one ``bfcl evaluate`` invocation.

    ``--partial-eval`` is always passed: the adapter legitimately runs
    id-subsets (``--resume`` skips complete ids), and BFCL's checker otherwise
    raises "Length of model result (N) does not match length of test entries"
    when a category's result file holds fewer entries than the full prompt set.
    """
    argv = [
        "evaluate",
        "--model", model_key,
        "--test-category", ",".join(categories),
    ]
    if result_dir is not None:
        argv += ["--result-dir", str(result_dir)]
    if score_dir is not None:
        argv += ["--score-dir", str(score_dir)]
    if partial_eval:
        argv.append("--partial-eval")
    return argv


def compute_thrash(turn_counts, cap=THRASH_TURN_CAP):
    """Summarize per-id step counts into the thrash-guard subscore.

    ``turn_counts`` maps id -> ``{"turns", "max_steps", "steps_per_turn"}``
    (from ``normalize.count_turns_per_id``). A conversation thrashes when any
    of its turns reaches ``cap`` steps — BFCL force-quits a turn after 20
    steps, so ``max_steps > cap`` (21 = 20 attempts + the forced terminal
    message) means the model never answered within that turn. Returns
    ``{"flag", "max_steps_observed", "cap", "cap_hit_ids", "step_counts"}``.
    """
    counts = turn_counts or {}
    max_steps = {tid: int(rec.get("max_steps", 0))
                 for tid, rec in counts.items() if isinstance(rec, dict)}
    cap_hits = sorted([tid for tid, m in max_steps.items() if m > cap])
    return {
        "flag": bool(cap_hits),
        "max_steps_observed": max(max_steps.values()) if max_steps else 0,
        "cap": int(cap),
        "cap_hit_ids": cap_hits,
        "step_counts": {tid: rec for tid, rec in counts.items()},
    }


def run_bfcl_cli(argv, *, project_root):
    """Invoke ``bfcl <argv>`` in-process via typer/click.

    Imported here (not at module top) so this module stays bfcl_eval-free for
    unit tests. ``BFCL_PROJECT_ROOT`` is set so BFCL's ``result/`` and
    ``score/`` land under the harness's run directory, not inside the venv's
    site-packages.
    """
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    import typer  # noqa: PLC0415
    from bfcl_eval.__main__ import cli  # noqa: PLC0415

    click_cmd = typer.main.get_command(cli)
    try:
        click_cmd(args=list(argv), standalone_mode=False)
    except SystemExit as exc:  # click/typer may sys.exit(0)
        code = exc.code if isinstance(exc.code, int) else 0
        if code != 0:
            raise
    return 0


def run(checkpoint: str, *, base_url, model_name=None, resume=False,
        categories=normalize.MULTI_TURN_CATEGORIES,
        run_root=None, results_dir=None):
    """Run the BFCL component and write the normalized result JSON.

    ``checkpoint`` is the checkpoint ref (local dir or HF id); ``base_url`` is
    the live ``mlx_lm server`` handed in by the shared ``ServerManager``
    (ticket #16) — per the runner contract the adapter never owns serve.
    ``model_name`` is the matrix row name used for the result ``model`` field /
    filename (defaults to ``checkpoint``).

    ``--resume`` (ticket #6, decision 7a): BFCL resumes per-id natively. When
    ``resume`` is set we simply do NOT pass ``--allow-overwrite``; BFCL then
    loads each category's existing ``result/<model>/`` file, collects the ids
    already present (``_llm_response_generation.collect_test_cases``), and
    generates only the ids that are missing. A fresh run passes
    ``--allow-overwrite`` and regenerates everything.

    Returns the normalized result dict (also written to ``results_dir``).
    """
    # Imported here so the module stays bfcl_eval-free for tests.
    from .register import register  # noqa: PLC0415

    model_name = model_name if model_name is not None else checkpoint
    run_root = run_root or os.path.join(_REPO_ROOT, "evals", "bfcl_runs")
    results_dir = results_dir or os.path.join(_REPO_ROOT, "evals", "results")
    os.makedirs(run_root, exist_ok=True)
    result_root = os.path.join(run_root, "result")
    score_root = os.path.join(run_root, "score")

    model_key = register()
    start = time.monotonic()

    # Point BFCL at the live server + the checkpoint's tokenizer.
    os.environ["REMOTE_OPENAI_BASE_URL"] = base_url
    os.environ["REMOTE_OPENAI_TOKENIZER_PATH"] = str(checkpoint)
    os.environ.setdefault("REMOTE_OPENAI_API_KEY", "EMPTY")

    # --- generate -----------------------------------------------------------
    # allow_overwrite=False on resume -> BFCL's native per-id skip; True on a
    # fresh run -> regenerate all ids.
    run_bfcl_cli(
        build_generate_argv(
            model_key,
            categories,
            allow_overwrite=not resume,
            result_dir=result_root,
        ),
        project_root=run_root,
    )

    # --- evaluate -----------------------------------------------------------
    run_bfcl_cli(
        build_evaluate_argv(model_key, categories,
                            result_dir=result_root, score_dir=score_root),
        project_root=run_root,
    )

    # --- normalize + thrash guard -------------------------------------------
    score, per_category, missing = normalize.normalize(
        result_root, score_root, model_key, categories
    )
    turn_counts = {}
    for category in categories:
        result_file = normalize.find_result_file(result_root, model_key,
                                                 category)
        if result_file:
            turn_counts.update(normalize.count_turns_per_id(result_file))
    thrash = compute_thrash(turn_counts)
    runtime_s = time.monotonic() - start

    if score is None:
        raise RuntimeError(
            "BFCL produced no usable score for {0} (missing categories: {1})".format(
                model_name, ", ".join(missing) or "none parsed"
            )
        )

    result = results_mod.new_result(
        component=COMPONENT,
        model=model_name,
        checkpoint=checkpoint,
        score=score,
        subscores={
            "per_category": per_category,
            "missing_categories": missing,
            "resume": bool(resume),
            "thrash": thrash,
        },
        runtime_s=runtime_s,
        artifact_versions=_artifact_versions(),
    )
    errors = results_mod.validate_result(result)
    if errors:
        raise ValueError("invalid result: {0}".format("; ".join(errors)))
    results_mod.write_result(result, results_dir)
    return result


def _artifact_versions():
    versions = {"component": "bfcl", "python": sys.version.split()[0]}
    try:
        import importlib.metadata as md  # noqa: PLC0415
        versions["bfcl_eval"] = md.version("bfcl-eval")
    except Exception:  # pragma: no cover - best effort
        pass
    try:
        import mlx_lm  # noqa: PLC0415
        versions["mlx_lm"] = getattr(mlx_lm, "__version__", "unknown")
    except Exception:  # pragma: no cover - mlx_lm lives in .venv-core
        pass
    return versions
