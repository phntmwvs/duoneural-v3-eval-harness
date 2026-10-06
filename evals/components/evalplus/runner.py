"""EvalPlus HumanEval+MBPP runner — orchestration (ticket #18).

Stdlib-only: this module shells out to the ``evalplus.codegen`` /
``evalplus.evaluate`` console scripts (generation on the host, sandboxed
evaluation in Docker) and never imports ``evalplus``. The parser/normalizer
lives in ``normalize.py``.

Responsibilities:

1. Receive the live ``mlx_lm server`` ``base_url`` from the shared
   ``ServerManager`` (ticket #16) — the adapter never owns serve.
2. Generate on the host: ``evalplus.codegen <model_key> <dataset>
   --backend openai --base-url <url> --greedy`` for humaneval + mbpp
   (chat-only backend; ``--greedy`` forces n=1/temp=0 — ticket #5).
3. Evaluate sandboxed: ``docker run -v <root>:/app ganler/evalplus:latest
   evalplus.evaluate <dataset> --samples /app/<dataset>/<id>.jsonl``
   (untrusted generated code execs inside the container).
4. ``--resume`` (ticket #6, decision 7a): preserve the sample file so
   ``evalplus.codegen --resume`` appends only the missing tasks; a fresh run
   clears the sample file first (codegen never truncates).
5. Map the two ``*_eval_results.json`` into the normalized result
   (``score`` = mean of the two ``+`` pass@1s; four subscores + dataset
   versions/hashes).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

# Allow running as ``python -m evals.components.evalplus`` and under tests.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evals import results as results_mod  # noqa: E402

from . import normalize  # noqa: E402

COMPONENT = "evalplus"

#: ``mlx_lm server`` registers its loaded checkpoint under the literal key
#: ``default_model`` (``server.py: _model_map["default_model"] = cli_args.model``)
#: and resolves any request ``model`` through ``_model_map.get(model, model)``.
#: Sending ``default_model`` therefore always selects the served checkpoint —
#: a local quant path or an HF id — with no served-id resolution (whose
#: HF-id edge case returns an empty ``/v1/models``). Verified against
#: mlx-lm 0.32.0 ``server.py``.
DEFAULT_MODEL_KEY = "default_model"

#: Docker image that runs the sandboxed evaluation (ticket #5 §3).
DOCKER_IMAGE = "ganler/evalplus:latest"


def resolve_evalplus_command(script_name, python=None):
    """Argv prefix that runs an ``evalplus.*`` console script.

    ``script_name`` is ``"evalplus.codegen"`` or ``"evalplus.evaluate"``.
    Resolution order mirrors ``ServerManager.resolve_server_command``: the
    console script next to the current interpreter (venv install), then a
    PATH hit, then ``python -m <script_name>`` as a fallback.
    """
    if python is None:
        python = sys.executable
    script = os.path.join(os.path.dirname(python), script_name)
    if os.path.exists(script):
        return [script]
    on_path = shutil.which(script_name)
    if on_path:
        return [on_path]
    return [python, "-m", script_name]


def resolve_docker_command():
    """Argv prefix for ``docker``, tolerant of Docker Desktop's CLI location.

    On macOS the Desktop app's CLI lives at ``/Applications/Docker.app/
    Contents/Resources/bin/docker`` and is symlinked into ``/usr/local/bin``,
    which is often not on a non-interactive SSH ``PATH``. Check the common
    locations before falling back to ``docker`` on ``PATH``.
    """
    candidates = [
        "/usr/local/bin/docker",
        "/Applications/Docker.app/Contents/Resources/bin/docker",
    ]
    for c in candidates:
        if os.path.exists(c):
            return [c]
    return ["docker"]


def build_codegen_argv(model_key, dataset, *, base_url, root, resume=False):
    """Argv (after the ``evalplus.codegen`` prefix) for one generation run.

    ``model_key`` is sent verbatim as the OpenAI ``model`` field AND used in
    the output filename. ``--greedy`` forces n=1/temp=0 (ticket #5 §2.2).
    ``--noresume`` regenerates from scratch on a fresh run; ``--resume``
    appends to a partial file (ticket #6, decision 7a).
    """
    argv = [
        model_key, dataset,
        "--backend", "openai",
        "--base-url", base_url,
        "--greedy",
        "--root", root,
    ]
    argv.append("--resume" if resume else "--noresume")
    return argv


def build_docker_eval_argv(host_root, dataset, samples_filename):
    """Argv for the sandboxed evaluation (one ``docker run``).

    Mounts ``host_root`` at ``/app`` (the image's ``WORKDIR``) and runs
    ``evalplus.evaluate <dataset> --samples /app/<dataset>/<file>``. The
    ``*_eval_results.json`` lands next to the samples file on the host mount.
    """
    return [
        "run", "--rm",
        "-v", "{0}:/app".format(host_root),
        DOCKER_IMAGE,
        "evalplus.evaluate", dataset,
        "--samples", "/app/{0}/{1}".format(dataset, samples_filename),
    ]


def run_subprocess(argv):
    """Invoke a subprocess (codegen or docker); the seam tests patch."""
    subprocess.run(argv, check=True)


def run(checkpoint, *, base_url, model_name=None, resume=False,
        run_root=None, results_dir=None):
    """Run the EvalPlus component and write the normalized result JSON.

    ``checkpoint`` is the checkpoint ref (local dir or HF id); ``base_url`` is
    the live ``mlx_lm server`` handed in by the shared ``ServerManager``
    (ticket #16) — per the runner contract the adapter never owns serve.
    ``model_name`` is the matrix row name used for the result ``model`` field /
    filename (defaults to ``checkpoint``).

    ``--resume`` (ticket #6, decision 7a): preserves the sample file so
    ``evalplus.codegen --resume`` appends only missing tasks. A fresh run
    clears the sample file first (codegen never truncates — it appends), then
    regenerates everything. In both cases the stale ``*_eval_results.json``
    is cleared before the Docker evaluation so the full sample set is scored.

    Returns the normalized result dict (also written to ``results_dir``).
    """
    model_name = model_name if model_name is not None else checkpoint
    run_root = run_root or os.path.join(
        _REPO_ROOT, "evals", "evalplus_runs", results_mod.slug_filename(model_name)
    )
    results_dir = results_dir or os.path.join(_REPO_ROOT, "evals", "results")
    os.makedirs(run_root, exist_ok=True)

    start = time.monotonic()
    model_key = DEFAULT_MODEL_KEY
    codegen_prefix = resolve_evalplus_command("evalplus.codegen")
    docker_prefix = resolve_docker_command()

    # --- generate on host (chat-only backend; greedy n=1/temp=0) ------------
    for dataset in normalize.DATASETS:
        if not resume:
            normalize.clear_samples(run_root, dataset)
        run_subprocess(
            codegen_prefix
            + build_codegen_argv(
                model_key, dataset, base_url=base_url, root=run_root, resume=resume
            )
        )

    # --- evaluate sandboxed in Docker --------------------------------------
    for dataset in normalize.DATASETS:
        normalize.clear_eval_results(run_root, dataset)
        samples_file = normalize.find_samples_file(run_root, dataset)
        if samples_file is None:
            raise RuntimeError(
                "EvalPlus generation produced no sample file for {0} under {1}".format(
                    dataset, run_root
                )
            )
        run_subprocess(
            docker_prefix
            + build_docker_eval_argv(run_root, dataset, os.path.basename(samples_file))
        )

    # --- normalize ----------------------------------------------------------
    score, per_dataset, missing, partial = normalize.normalize(run_root)
    runtime_s = time.monotonic() - start

    if score is None:
        raise RuntimeError(
            "EvalPlus produced no usable score for {0} (missing datasets: {1})".format(
                model_name, ", ".join(missing) or "none parsed"
            )
        )

    result = results_mod.new_result(
        component=COMPONENT,
        model=model_name,
        checkpoint=checkpoint,
        score=score,
        subscores=_build_subscores(per_dataset, missing, partial, resume),
        runtime_s=runtime_s,
        artifact_versions=_artifact_versions(),
    )
    errors = results_mod.validate_result(result)
    if errors:
        raise ValueError("invalid result: {0}".format("; ".join(errors)))
    results_mod.write_result(result, results_dir)
    return result


def _build_subscores(per_dataset, missing, partial, resume):
    """Assemble the component-specific ``subscores`` (ticket #5 §4.3)."""
    def pick(dataset, key):
        entry = per_dataset.get(dataset) or {}
        return entry.get(key)

    return {
        "humaneval_base_pass1": pick("humaneval", "base_pass1"),
        "humaneval_plus_pass1": pick("humaneval", "plus_pass1"),
        "mbpp_base_pass1": pick("mbpp", "base_pass1"),
        "mbpp_plus_pass1": pick("mbpp", "plus_pass1"),
        "n_tasks": {d: pick(d, "n_tasks") for d in normalize.DATASETS},
        "dataset_versions": normalize.DATASET_VERSIONS,
        "dataset_hashes": {d: pick(d, "hash") for d in normalize.DATASETS},
        "decoding": {"greedy": True, "temperature": 0.0, "n_samples": 1},
        "sandbox": "docker:" + DOCKER_IMAGE,
        "missing_datasets": missing,
        "partial_datasets": partial,
        "complete": not missing and not partial,
        "resume": bool(resume),
    }


def _artifact_versions():
    """Best-effort versions of the evalplus/openai/mlx_lm/docker stack."""
    versions = {"component": "evalplus", "python": sys.version.split()[0]}
    import importlib.metadata as md  # noqa: PLC0415
    for dist, key in (("evalplus", "evalplus"), ("openai", "openai_client")):
        try:
            versions[key] = md.version(dist)
        except Exception:  # pragma: no cover - best effort
            pass
    try:
        import mlx_lm  # noqa: PLC0415
        versions["mlx_lm"] = getattr(mlx_lm, "__version__", "unknown")
    except Exception:  # pragma: no cover - mlx_lm lives in .venv-core
        pass
    try:
        out = subprocess.run(
            resolve_docker_command()
            + ["image", "inspect", DOCKER_IMAGE, "--format", "{{.RepoDigests}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            versions["docker_image"] = out.stdout.strip()
    except Exception:  # pragma: no cover - best effort
        pass
    return versions
