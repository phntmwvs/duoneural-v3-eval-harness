#!/usr/bin/env python3
"""Row-major 4×3 matrix orchestration (wayfinder ticket #20).

For each matrix row (BF16, 8-bit, 4-bit, base) — bring up that row's
``mlx_lm server`` once via the shared ``ServerManager`` (ticket #16),
run all three component adapters (``bfcl`` / ``evalplus`` / ``hermes``)
against the live ``base_url``, tear the server down, advance to the next
row → **4 server cycles** for the full 4×3 = 12-cell matrix.

Per ticket #6 (resolution 6a, row-major; resolution 5c, mixed
checkpoint refs; resolution 7a, per-cell resume):

* one server per row, torn down before the next row starts — the BF16
  row alone is ~17 GB of weights, so only one row can be resident on
  the 48 GB MBP at a time.
* aggregate per-cell results land in ``evals/results/`` (gitignored,
  ``<component>-<model>.json``) via the existing
  ``evals.results.write_result`` path — the matrix does **not** invent
  a new result schema.
* ``--resume`` skips cells whose result file is already complete
  (per-component ``complete`` flag — the per-component ``--resume``
  adapter flag is *also* forwarded so the adapter can resume its own
  internal state, e.g. BFCL's per-id skip).
* the matrix is a *thin* orchestrator: it composes
  ``ServerManager`` (ticket #16) + ``run_eval.run_component``
  (ticket #16) + ``evals.results`` (ticket #16). It does not import
  any per-component adapter in-process; each cell is a subprocess into
  the component's per-venv python (cross-venv execution — #6 1a).

Stdlib-only and 3.9-compatible: it must run from any interpreter on the
MBP and under the test venvs.

Usage (on the MBP):

    ./run_matrix.py                       # full 4×3 matrix, fresh
    ./run_matrix.py --resume              # skip already-complete cells
    ./run_matrix.py --rows bf16,8bit      # subset of rows
    ./run_matrix.py --components hermes   # subset of components
    ./run_matrix.py --dry-run             # print the plan, do nothing
    ./run_matrix.py --no-caffeinate       # don't pin the MBP awake
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Callable, Iterable

# Re-use the scaffold's per-venv dispatch (ticket #16) — the matrix is
# just a loop on top of it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evals import MATRIX_ROWS  # noqa: E402
from evals import results as results_mod  # noqa: E402
from evals import server as server_mod  # noqa: E402
import run_eval  # noqa: E402

# The single source of truth for row→checkpoint (ticket #6, resolution 5c).
# Three local quant dirs + the stock LiquidAI HF id for the base row.
DEFAULT_ROW_CHECKPOINTS = {
    "bf16":  "checkpoints/DuoNeural-v3-BF16",
    "8bit":  "checkpoints/DuoNeural-v3-8bit",
    "4bit":  "checkpoints/DuoNeural-v3-4bit",
    "base":  "LiquidAI/LFM2.5-8B-A1B",
}

# The order components are run within a row. Defaults to the canonical
# order from ``evals.results.COMPONENTS``; overridable via --components.
DEFAULT_COMPONENT_ORDER = tuple(results_mod.COMPONENTS)

# Where per-cell result files land (gitignored; the matrix never commits here).
DEFAULT_RESULTS_DIR = os.path.join("evals", "results")

# ``.venv-core`` is the shared mlx-lm venv (SETUP.md §2). All three
# components drive ``mlx_lm server`` from there.
DEFAULT_SERVER_VENV = ".venv-core"

# When the matrix is run inside a test (or a debug dry-run) we don't
# want to actually spawn ``mlx_lm server`` — this env knob short-
# circuits the ServerManager and hands the adapters a sentinel URL.
# The real components won't accept it; the test fakes do.
_TEST_BASE_URL_OVERRIDE_ENV = "RUN_MATRIX_TEST_BASE_URL_OVERRIDE"

# A reasonable default: BF16 ~17 GB weights + warmup + multi-turn gen
# can easily need several minutes. The ServerManager's own default is
# 300 s; we set the matrix default a touch higher to absorb a cold
# load + 3 components + 4 rows.
DEFAULT_SERVER_STARTUP_TIMEOUT_S = 600.0


# ---------------------------------------------------------------------------
# Pure helpers (no I/O, no subprocess) — easy to unit-test
# ---------------------------------------------------------------------------


def resolve_checkpoint(row: str,
                       row_checkpoints: dict = DEFAULT_ROW_CHECKPOINTS) -> str:
    """Return the checkpoint ref (local dir or HF id) for ``row``."""
    try:
        return row_checkpoints[row]
    except KeyError:
        raise ValueError(
            "unknown matrix row {0!r}; known rows: {1}".format(
                row, sorted(row_checkpoints)
            )
        )


def resolve_server_python(repo_root: str, python: str | None = None) -> str:
    """Path to the python interpreter ``mlx_lm server`` should run under.

    The matrix is invoked from any interpreter (the user's shell
    python, system python, a test venv). ``mlx_lm`` itself is installed
    in ``.venv-core`` (SETUP.md §2) — that's the interpreter whose
    console script ``ServerManager`` resolves. Point it explicitly
    rather than letting it fall back to ``sys.executable`` of the caller
    (which on the MBP is whichever ad-hoc interpreter the user ran the
    matrix from).
    """
    if python is not None:
        return python
    return os.path.join(repo_root, DEFAULT_SERVER_VENV, "bin", "python")


def is_cell_complete(path: str, *, component: str) -> bool:
    """Decide whether an existing result JSON represents a *complete* cell.

    The matrix's ``--resume`` skips cells that are already complete;
    the per-component ``--resume`` flag is forwarded independently. The
    per-component rule (what "complete" means) is read from the
    component's own ``subscores`` — the adapter is the source of truth.

    * bfcl: ``subscores.complete`` is True AND no missing/partial cats.
    * evalplus: ``subscores.complete`` is True AND no missing/partial datasets.
    * hermes: ``n_cases`` is the suite size (40) AND no per-case errors.
    * unknown / malformed: not complete (re-run on resume is the safe
      default — a partial file is indistinguishable from a crash).
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return False

    sub = data.get("subscores") or {}
    if component in ("bfcl", "evalplus"):
        if not sub.get("complete"):
            return False
        if sub.get("missing_categories") or sub.get("missing_datasets"):
            return False
        if sub.get("partial_categories") or sub.get("partial_datasets"):
            return False
        return True
    if component == "hermes":
        # 40 hand-authored cases (ticket #12). The hermes adapter
        # records every case it ran, errors and all; a complete run
        # has 40 cases and 0 errors.
        return sub.get("n_cases", 0) >= 40 and sub.get("n_errors", 0) == 0
    return False


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _build_server_manager(
    checkpoint: str,
    *,
    repo_root: str,
    server_command_factory: Callable | None = None,
    server_startup_timeout_s: float = DEFAULT_SERVER_STARTUP_TIMEOUT_S,
    server_stop_timeout_s: float = server_mod.DEFAULT_STOP_TIMEOUT_S,
    server_python: str | None = None,
    log_file=None,
) -> server_mod.ServerManager:
    """Construct a ``ServerManager`` for ``checkpoint``.

    ``server_command_factory(ckpt, host, port, python) -> list[str]``
    overrides the default ``mlx_lm server`` argv — tests use it to
    inject ``fake_mlx_lm_server.py``; production leaves it None and the
    real argv is resolved from ``.venv-core``.
    """
    python = server_python or resolve_server_python(repo_root)
    if server_command_factory is not None:
        # We need a port; the factory builds the argv. We pass the
        # port after ServerManager allocates it (see ``start``).
        return _FactoryServerManager(
            checkpoint,
            python=python,
            factory=server_command_factory,
            startup_timeout_s=server_startup_timeout_s,
            stop_timeout_s=server_stop_timeout_s,
            log_file=log_file,
        )
    return server_mod.ServerManager(
        checkpoint,
        python=python,
        startup_timeout_s=server_startup_timeout_s,
        stop_timeout_s=server_stop_timeout_s,
        log_file=log_file,
    )


class _FactoryServerManager:
    """A ServerManager that builds its argv from a factory (test seam).

    Production uses ``ServerManager`` directly. Tests inject this so the
    fake server can be spawned instead of ``mlx_lm server``.
    """

    def __init__(self, checkpoint, *, python, factory,
                 startup_timeout_s, stop_timeout_s, log_file=None):
        self.checkpoint = checkpoint
        self._python = python
        self._factory = factory
        self._startup_timeout_s = startup_timeout_s
        self._stop_timeout_s = stop_timeout_s
        self._log_file = log_file
        # Reuse ServerManager for port allocation + readiness poll, but
        # build a fresh ServerManager per start() so each row gets a
        # clean port.
        self._inner: server_mod.ServerManager | None = None

    @property
    def base_url(self) -> str:
        return self._inner.base_url  # type: ignore[union-attr]

    def start(self) -> str:
        port = server_mod.find_free_port()
        argv = self._factory(
            self.checkpoint, server_mod.DEFAULT_HOST, port, self._python
        )
        self._inner = server_mod.ServerManager(
            self.checkpoint,
            port=port,
            command=argv,
            startup_timeout_s=self._startup_timeout_s,
            stop_timeout_s=self._stop_timeout_s,
            log_file=self._log_file,
        )
        return self._inner.start()

    def stop(self) -> None:
        if self._inner is not None:
            self._inner.stop()

    def is_running(self) -> bool:
        return self._inner is not None and self._inner.is_running()

    def __enter__(self): self.start(); return self
    def __exit__(self, *a): self.stop()


# ---------------------------------------------------------------------------
# Cell invocation
# ---------------------------------------------------------------------------


def _invoke_component(
    component: str,
    checkpoint: str,
    *,
    model_name: str,
    base_url: str,
    resume: bool,
    env: dict,
    repo_root: str,
) -> int:
    """Run one (component, checkpoint) cell and return its exit code.

    Uses the shared ``run_eval.run_component`` so the per-venv dispatch,
    the argv shape, and the PYTHONPATH wiring are all reused — the
    matrix does not reinvent the dispatch.
    """
    # Echo the cell plan first so the run log shows what's happening
    # even if the adapter's own print is delayed.
    sys.stdout.write(
        "[matrix] row={0} component={1} base_url={2} resume={3}\n".format(
            model_name, component, base_url, bool(resume)
        )
    )
    sys.stdout.flush()

    return run_eval.run_component(
        component,
        checkpoint,
        model_name=model_name,
        resume=resume,
        base_url=base_url,
        repo_root=repo_root,
        env=env,
    )


# ---------------------------------------------------------------------------
# The matrix loop
# ---------------------------------------------------------------------------


def _parse_rows_arg(value: str) -> tuple:
    """Parse a --rows value (``"bf16,8bit"``) into a tuple of row names."""
    rows = tuple(r.strip() for r in value.split(",") if r.strip())
    if not rows:
        raise argparse.ArgumentTypeError("--rows must list at least one row")
    for r in rows:
        if r not in DEFAULT_ROW_CHECKPOINTS:
            raise argparse.ArgumentTypeError(
                "unknown row {0!r}; known: {1}".format(
                    r, sorted(DEFAULT_ROW_CHECKPOINTS)
                )
            )
    return rows


def _parse_components_arg(value: str) -> tuple:
    """Parse a --components value (``"bfcl,hermes"``) into a tuple."""
    comps = tuple(c.strip() for c in value.split(",") if c.strip())
    if not comps:
        raise argparse.ArgumentTypeError(
            "--components must list at least one component"
        )
    for c in comps:
        if c not in results_mod.COMPONENTS:
            raise argparse.ArgumentTypeError(
                "unknown component {0!r}; known: {1}".format(
                    c, list(results_mod.COMPONENTS)
                )
            )
    return comps


def _print_plan(rows, components, checkpoints) -> None:
    print("[matrix] plan: {0} rows x {1} components = {2} cells".format(
        len(rows), len(components), len(rows) * len(components)
    ))
    for row in rows:
        print("[matrix]   row {0!r} -> {1}".format(row, checkpoints[row]))
        for comp in components:
            print("[matrix]     {0}".format(comp))


def _print_summary(summary: dict, results_dir: str) -> None:
    """Print the 4×3 score grid the matrix just produced."""
    print("[matrix] ============================================================")
    print("[matrix] summary ({ran} ran, {skipped} skipped, {failed} failed) "
          "in {elapsed_s:.1f}s".format(**summary, elapsed_s=summary["elapsed_s"]))
    print("[matrix] results dir: {0}".format(results_dir))
    # Per-cell line.
    for cell in summary["cells"]:
        line = "[matrix]   {row:>5} / {component:<8} -> {status}".format(
            row=cell["row"], component=cell["component"], status=cell["status"]
        )
        if "score" in cell:
            line += "  score={0:.4f}".format(cell["score"])
        if "elapsed_s" in cell:
            line += "  ({0:.1f}s)".format(cell["elapsed_s"])
        print(line)


def _load_score(results_dir: str, component: str, model: str):
    """Read the score from a result JSON, or None on missing/malformed."""
    path = results_mod.result_path(results_dir, component, model)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("score")
    except (OSError, ValueError):
        return None


def _caffeinate_status(env: dict) -> bool:
    """Return True if the run should be wrapped in ``caffeinate -dims``.

    The MBP sleeps on idle; an unattended multi-hour matrix run is
    killed by App Nap / display sleep without ``caffeinate``. The
    matrix does not spawn caffeinate itself — it expects the user to
    wrap the invocation (``caffeinate -dims -t 14400 ./run_matrix.py``).
    The check here is informational: it tells the user the recommended
    wrap when they forget.
    """
    import shutil as _sh
    return _sh.which("caffeinate") is not None and env.get(
        "RUN_MATRIX_CAFFEINATE", "1"
    ) == "1"


def run_matrix(
    *,
    row_checkpoints: dict = DEFAULT_ROW_CHECKPOINTS,
    component_order: Iterable[str] = DEFAULT_COMPONENT_ORDER,
    results_dir: str = DEFAULT_RESULTS_DIR,
    resume: bool = False,
    env: dict | None = None,
    caffeinate: bool = True,
    serve: bool = True,
    server_command_factory: Callable | None = None,
    server_startup_timeout_s: float = DEFAULT_SERVER_STARTUP_TIMEOUT_S,
    server_python: str | None = None,
    repo_root: str | None = None,
    log_dir: str | None = None,
) -> dict:
    """Drive the row-major matrix; return a summary dict.

    The function is the matrix's *programmatic* entry point. The CLI
    (below) is a thin wrapper. Tests drive ``run_matrix`` directly;
    the MBP runs ``./run_matrix.py`` which dispatches here.

    The return value is a JSON-serializable summary suitable for an
    ops dashboard: ``{ran, skipped, failed, elapsed_s, cells}``.
    """
    rows = tuple(row_checkpoints)
    components = tuple(component_order)
    if not rows or not components:
        raise ValueError("row_checkpoints and component_order must be non-empty")
    repo_root = repo_root or os.path.dirname(os.path.abspath(__file__))
    env = dict(env) if env is not None else dict(os.environ)
    if not caffeinate:
        env["RUN_MATRIX_CAFFEINATE"] = "0"

    # Each row gets its own per-row server log; useful for post-mortem
    # if a row dies. Default: evals/matrix_logs/<row>/server.log.
    if log_dir is None:
        log_dir = os.path.join(repo_root, "evals", "matrix_logs")
    os.makedirs(results_dir, exist_ok=True)

    summary = {
        "ran": 0,
        "skipped": 0,
        "failed": 0,
        "elapsed_s": 0.0,
        "cells": [],
    }
    t0 = time.monotonic()

    for row in rows:
        checkpoint = row_checkpoints[row]
        row_log_dir = os.path.join(log_dir, row)
        os.makedirs(row_log_dir, exist_ok=True)
        server_log = open(os.path.join(row_log_dir, "server.log"), "ab")

        mgr = None
        base_url = None
        if not serve:
            # Test / dry-run path: hand adapters the sentinel URL.
            base_url = env.get(_TEST_BASE_URL_OVERRIDE_ENV,
                               "http://INVALID_NO_S")
            sys.stdout.write(
                "[matrix] row {0!r} (serve disabled; using {1})\n".format(
                    row, base_url
                )
            )
        else:
            mgr = _build_server_manager(
                checkpoint,
                repo_root=repo_root,
                server_command_factory=server_command_factory,
                server_startup_timeout_s=server_startup_timeout_s,
                server_python=server_python,
                log_file=server_log,
            )
            try:
                base_url = mgr.start()
            except (RuntimeError, TimeoutError, OSError) as exc:
                # Server failed to come up. Record per-component
                # failures for this row, then move on (a single bad
                # row must not kill the whole matrix).
                sys.stderr.write(
                    "[matrix] row {0!r} server failed: {1}\n".format(row, exc)
                )
                try:
                    mgr.stop()
                except Exception:
                    pass
                server_log.close()
                mgr = None
                for component in components:
                    summary["cells"].append({
                        "row": row, "component": component,
                        "status": "server_failed", "error": str(exc),
                    })
                    summary["failed"] += 1
                continue

        # The server is up (or the sentinel is in play). Run each
        # component. On any single-cell failure we record + continue
        # — a crash in one adapter (e.g. BFCL OOM) must not derail
        # the other two components for this row.
        for component in components:
            cell_t0 = time.monotonic()
            cell_path = results_mod.result_path(results_dir, component, row)

            if resume and is_cell_complete(cell_path, component=component):
                summary["cells"].append({
                    "row": row, "component": component, "status": "skipped_complete",
                    "score": _load_score(results_dir, component, row),
                    "path": cell_path,
                })
                summary["skipped"] += 1
                sys.stdout.write(
                    "[matrix]   {0}/{1}: skipped (complete at {2})\n".format(
                        row, component, cell_path
                    )
                )
                continue

            # A fresh run is destructive: a stale partial result is
            # useless, and the adapter will rewrite the file anyway.
            # Don't pre-delete — the adapter's write is atomic and a
            # crash mid-write leaves a partial file the next --resume
            # run will detect (is_cell_complete -> False) and
            # regenerate.

            cell_env = dict(env)
            # The fake component reads FAKE_RESULT_PATH to know where
            # to write; the real components ignore it. Setting it
            # unconditionally is harmless and lets the same env work
            # in tests.
            cell_env.setdefault(
                "FAKE_RESULT_PATH",
                os.path.abspath(cell_path),
            )

            rc = _invoke_component(
                component, checkpoint,
                model_name=row, base_url=base_url,
                resume=False,  # matrix decides per-cell; adapter never re-resumes
                env=cell_env,
                repo_root=repo_root,
            )
            cell_elapsed = time.monotonic() - cell_t0
            if rc != 0:
                summary["cells"].append({
                    "row": row, "component": component, "status": "failed",
                    "exit_code": rc, "elapsed_s": cell_elapsed,
                    "path": cell_path,
                })
                summary["failed"] += 1
                sys.stderr.write(
                    "[matrix]   {0}/{1}: FAILED (exit {2}, {3:.1f}s)\n".format(
                        row, component, rc, cell_elapsed
                    )
                )
                continue
            summary["cells"].append({
                "row": row, "component": component, "status": "ok",
                "score": _load_score(results_dir, component, row),
                "elapsed_s": cell_elapsed,
                "path": cell_path,
            })
            summary["ran"] += 1

        if mgr is not None:
            mgr.stop()
        server_log.close()

    summary["elapsed_s"] = time.monotonic() - t0
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Row-major 4×3 eval matrix orchestrator (ticket #20).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip cells whose result file is already complete; forward "
             "--resume to the per-component adapter for its own internal resume",
    )
    p.add_argument(
        "--rows",
        type=_parse_rows_arg,
        default=None,
        help="comma-separated subset of rows (default: all 4: bf16,8bit,4bit,base)",
    )
    p.add_argument(
        "--components",
        type=_parse_components_arg,
        default=None,
        help="comma-separated subset of components (default: bfcl,evalplus,hermes)",
    )
    p.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="where per-cell result JSONs are written (default: evals/results)",
    )
    p.add_argument(
        "--no-caffeinate",
        action="store_true",
        help="don't pin the MBP awake (default: caffeinate -dims wraps the run)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the matrix plan and exit (no server, no adapters)",
    )
    p.add_argument(
        "--server-startup-timeout-s",
        type=float,
        default=DEFAULT_SERVER_STARTUP_TIMEOUT_S,
        help="seconds to wait for /v1/models to answer per row (default: %(default)s)",
    )
    p.add_argument(
        "--venv-core",
        default=os.path.join(DEFAULT_SERVER_VENV, "bin", "python"),
        help="path to the .venv-core python (the mlx-lm interpreter; "
             "default: <repo>/.venv-core/bin/python)",
    )
    p.add_argument(
        "--summary-json",
        default=None,
        help="write the run summary to this path as JSON (useful for ops)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    rows = args.rows if args.rows is not None else tuple(DEFAULT_ROW_CHECKPOINTS)
    components = (
        args.components if args.components is not None else DEFAULT_COMPONENT_ORDER
    )
    checkpoints = {r: DEFAULT_ROW_CHECKPOINTS[r] for r in rows}

    if args.dry_run:
        _print_plan(rows, components, checkpoints)
        return 0

    # The MBP sleeps on idle; an unattended multi-hour matrix run is
    # killed by App Nap / display sleep without ``caffeinate``. The
    # matrix does NOT spawn caffeinate itself (the in-process python
    # would be the child — too late). Remind the user when they
    # forgot to wrap.
    if not args.no_caffeinate and _caffeinate_status(dict(os.environ)):
        # Only remind when not on a known dev machine (the build
        # host mini has no caffeinate; this no-ops there).
        if os.environ.get("RUN_MATRIX_REMIND_CAFFEINATE", "1") == "1":
            print(
                "[matrix] NOTE: wrap the run with "
                "'caffeinate -dims -t 14400 ./run_matrix.py' to keep the "
                "MBP awake (--no-caffeinate to silence this hint).",
                file=sys.stderr,
            )

    repo_root = os.path.dirname(os.path.abspath(__file__))
    summary = run_matrix(
        row_checkpoints=checkpoints,
        component_order=components,
        results_dir=os.path.abspath(args.results_dir),
        resume=bool(args.resume),
        caffeinate=not args.no_caffeinate,
        server_startup_timeout_s=args.server_startup_timeout_s,
        server_python=args.venv_core,
        repo_root=repo_root,
    )
    _print_summary(summary, args.results_dir)

    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)),
                    exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print("[matrix] summary JSON: {0}".format(args.summary_json))

    # Exit non-zero if anything failed — the CI shell sees a clean
    # exit only when every cell ran (or was skipped) successfully.
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
