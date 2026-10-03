"""BFCL v3 multi-turn runner — orchestration (ticket #17).

Stdlib-only: this module is importable and unit-testable in any interpreter
(it shells out to ``bfcl`` and ``mlx_lm server``; it never imports
``bfcl_eval``). The ``bfcl_eval``-dependent seam lives in ``handler.py`` /
``register.py`` and is only exercised inside ``.venv-bfcl``.

Responsibilities:

1. Bring up one ``mlx_lm server`` for the checkpoint via the shared
   ``ServerManager`` (ticket #16) and hand BFCL its ``base_url`` through
   ``REMOTE_OPENAI_BASE_URL``.
2. Register ``DuoNeuralV3FCHandler`` at runtime and drive ``bfcl generate``
   for the four multi-turn categories, then ``bfcl evaluate``.
3. ``--resume``: skip categories whose ``result/<model>/`` file already holds
   a complete record for every requested id (ticket #6, decision 7a).
4. Thrash-loop guard (#3 hardening): count turns per conversation from the
   multi-turn result logs and surface the counts in ``subscores``; a
   conversation that hits the model's max-turn cap (never produced a terminal
   answer) is flagged, not aborted — the matrix/report decides.

BFCL is invoked in-process (``typer.main.get_command`` on
``bfcl_eval.__main__:cli``) exactly as the #3 spike did, but from this
component's ``.venv-bfcl`` interpreter via ``__main__.py``.
"""

from __future__ import annotations

import json
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
from evals import server as server_mod  # noqa: E402

from . import normalize  # noqa: E402

COMPONENT = "bfcl"
#: Consecutive turns with no terminal answer that marks a conversation as
#: thrashing. BFCL multi-turn caps turns per category; a conversation at the
#: cap never answered. Surfaced as a flag, not an abort (per ticket #17).
THRASH_TURN_CAP = 20


def build_generate_argv(model_key: str, categories, *, run_ids=None,
                        allow_overwrite=False, result_dir=None):
    """Argv (after ``bfcl``) for one ``bfcl generate`` invocation.

    ``categories`` and ``run_ids`` are comma-joined (BFCL's
    ``handle_multiple_input`` splits on commas). ``--skip-server-setup`` keeps
    BFCL from spawning its own server — the shared ``ServerManager`` owns it.
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
    if run_ids is not None:
        argv.append("--run-ids")
    if result_dir is not None:
        argv += ["--result-dir", str(result_dir)]
    return argv


def build_evaluate_argv(model_key: str, categories, *, result_dir=None,
                        score_dir=None):
    """Argv (after ``bfcl``) for one ``bfcl evaluate`` invocation."""
    argv = [
        "evaluate",
        "--model", model_key,
        "--test-category", ",".join(categories),
    ]
    if result_dir is not None:
        argv += ["--result-dir", str(result_dir)]
    if score_dir is not None:
        argv += ["--score-dir", str(score_dir)]
    return argv


def compute_thrash(turn_counts, cap=THRASH_TURN_CAP):
    """Summarize per-id turn counts into the thrash-guard subscore.

    Returns ``{"flag": bool, "max_turns_observed": int, "cap": int,
    "cap_hit_ids": [...], "turn_counts": {id: n}}``. ``flag`` is True when any
    conversation reached ``cap`` turns (never produced a terminal answer).
    """
    counts = {str(k): int(v) for k, v in (turn_counts or {}).items()}
    cap_hits = sorted([tid for tid, n in counts.items() if n >= cap])
    return {
        "flag": bool(cap_hits),
        "max_turns_observed": max(counts.values()) if counts else 0,
        "cap": int(cap),
        "cap_hit_ids": cap_hits,
        "turn_counts": counts,
    }


def ids_to_run(result_root, model_key, categories, requested_ids=None):
    """Per-category ids still needing generation (``--resume``).

    ``requested_ids`` maps category -> list of ids; when None, the full
    category is (re)generated unless its result file already has complete
    records. Returns ``{category: [ids...]}``; an empty list means the
    category is fully complete and is skipped.
    """
    plan = {}
    for category in categories:
        result_file = normalize.find_result_file(result_root, model_key, category)
        done = normalize.completed_ids(result_file) if result_file else set()
        want = requested_ids.get(category) if requested_ids else None
        if want is None:
            # No explicit id list: regenerate the category only if it has no
            # complete result file at all.
            plan[category] = [] if done else ["__all__"]
        else:
            plan[category] = [i for i in want if i not in done]
    return plan


def write_ids_file(path, plan):
    """Write ``test_case_ids_to_generate.json`` for BFCL's ``--run-ids``.

    Only categories with concrete (non-wildcard) id lists are written; a
    category mapped to ``["__all__"]`` or ``[]`` is omitted (BFCL generates
    the full category when it is not in this file).
    """
    payload = {
        category: ids
        for category, ids in plan.items()
        if ids and ids != ["__all__"]
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


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


def run(checkpoint: str, *, base_url=None, model_name=None, resume=False,
        categories=normalize.MULTI_TURN_CATEGORIES, requested_ids=None,
        run_root=None, results_dir=None, server_manager=None,
        startup_timeout_s=server_mod.DEFAULT_STARTUP_TIMEOUT_S):
    """Run the BFCL component and write the normalized result JSON.

    ``checkpoint`` is the checkpoint ref (local dir or HF id). ``model_name``
    is the matrix row name used for the result ``model`` field / filename
    (defaults to ``checkpoint``). ``base_url`` may be supplied by the matrix
    (which already owns a server); when None, this function brings up its own
    ``ServerManager`` for ``checkpoint`` and tears it down on exit.

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

    own_server = None
    if base_url is None:
        if server_manager is not None:
            own_server = server_manager
        else:
            own_server = server_mod.ServerManager(
                checkpoint, startup_timeout_s=startup_timeout_s
            )
        base_url = own_server.start()

    try:
        # Point BFCL at the live server + the checkpoint's tokenizer.
        os.environ["REMOTE_OPENAI_BASE_URL"] = base_url
        os.environ["REMOTE_OPENAI_TOKENIZER_PATH"] = str(checkpoint)
        os.environ.setdefault("REMOTE_OPENAI_API_KEY", "EMPTY")

        # --- generate -------------------------------------------------------
        plan = ids_to_run(result_root, model_key, categories, requested_ids) \
            if resume else {c: ["__all__"] for c in categories}
        to_generate = [c for c, ids in plan.items() if ids]
        if to_generate:
            ids_file = os.path.join(run_root, "test_case_ids_to_generate.json")
            concrete = {c: i for c, i in plan.items() if i and i != ["__all__"]}
            use_run_ids = bool(concrete) and not any(
                plan[c] == ["__all__"] for c in to_generate
            )
            if concrete:
                write_ids_file(ids_file, plan)
            argv = build_generate_argv(
                model_key,
                to_generate,
                run_ids=use_run_ids,
                allow_overwrite=not resume,
                result_dir=result_root,
            )
            run_bfcl_cli(argv, project_root=run_root)

        # --- evaluate -------------------------------------------------------
        run_bfcl_cli(
            build_evaluate_argv(model_key, categories,
                                result_dir=result_root, score_dir=score_root),
            project_root=run_root,
        )

        # --- normalize + thrash guard ---------------------------------------
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
    finally:
        if own_server is not None:
            own_server.stop()

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
