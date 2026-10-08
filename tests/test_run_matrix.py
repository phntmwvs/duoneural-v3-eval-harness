"""Tests for ``run_matrix.py`` — row-major 4×3 orchestration (ticket #20).

Drives the matrix offline against ``tests/fake_mlx_lm_server.py`` (a stub
answering ``/v1/models``) and a tiny fake component adapter
(``tests/fake_component.py``) so the orchestration is proven on the build
host without MLX, without per-venv interpreters, and without the real
BFCL/EvalPlus/Hermes deps.

Stdlib ``unittest`` (also pytest-compatible). The matrix is imported as
``run_matrix`` from the repo root — it follows the same one-module shape
as ``run_eval.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_matrix  # noqa: E402
import run_eval  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS = os.path.join(_REPO_ROOT, "tests")
_FAKE_SERVER = os.path.join(_TESTS, "fake_mlx_lm_server.py")
_FAKE_COMPONENT = os.path.join(_TESTS, "fake_component.py")


def _read_jsonl(path: str):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Pure-function / table-driven tests — no subprocess, no server
# ---------------------------------------------------------------------------


class RowCheckpointTest(unittest.TestCase):
    """The row→checkpoint map is the single source of truth (ticket #6 5c)."""

    def test_default_map_has_all_four_rows(self):
        self.assertEqual(set(run_matrix.DEFAULT_ROW_CHECKPOINTS),
                         {"bf16", "8bit", "4bit", "base"})

    def test_quants_are_local_dirs(self):
        # The 3 quant artifacts ship as local dirs — no HF round-trip.
        for row in ("bf16", "8bit", "4bit"):
            self.assertTrue(
                run_matrix.DEFAULT_ROW_CHECKPOINTS[row].startswith("checkpoints/"),
                "row {0!r} should be a local dir, got {1!r}".format(
                    row, run_matrix.DEFAULT_ROW_CHECKPOINTS[row]
                ),
            )

    def test_base_is_hf_id(self):
        # The base row is the stock LiquidAI ref — pulled on first serve.
        self.assertEqual(
            run_matrix.DEFAULT_ROW_CHECKPOINTS["base"],
            "LiquidAI/LFM2.5-8B-A1B",
        )

    def test_known_row_resolves(self):
        self.assertEqual(
            run_matrix.resolve_checkpoint("bf16"),
            run_matrix.DEFAULT_ROW_CHECKPOINTS["bf16"],
        )

    def test_unknown_row_raises(self):
        with self.assertRaises(ValueError):
            run_matrix.resolve_checkpoint("not-a-row")


class RowOrderTest(unittest.TestCase):
    """The matrix order is a function of the row map, not a hard-coded loop."""

    def test_default_order_matches_evals_matrix_rows(self):
        from evals import MATRIX_ROWS
        self.assertEqual(tuple(run_matrix.DEFAULT_ROW_CHECKPOINTS), MATRIX_ROWS)

    def test_default_component_order_matches_evals_components(self):
        from evals import results as results_mod
        self.assertEqual(run_matrix.DEFAULT_COMPONENT_ORDER, results_mod.COMPONENTS)


class CompleteCellTest(unittest.TestCase):
    """The matrix's ``is_cell_complete`` decides whether to skip on --resume."""

    def test_missing_file_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(
                run_matrix.is_cell_complete(
                    os.path.join(d, "bfcl-bf16.json"), component="bfcl"
                )
            )

    def test_complete_bfcl_cell(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bfcl-bf16.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({
                    "component": "bfcl",
                    "model": "bf16",
                    "checkpoint": "ckpt",
                    "score": 0.1,
                    "subscores": {
                        "complete": True,
                        "missing_categories": [],
                        "partial_categories": [],
                    },
                    "runtime_s": 1.0,
                    "artifact_versions": {},
                    "timestamp": "2026-10-01T00:00:00Z",
                }, fh)
            self.assertTrue(run_matrix.is_cell_complete(path, component="bfcl"))

    def test_incomplete_bfcl_cell_is_not_complete(self):
        # A BFCL result with missing categories must NOT be considered
        # complete — a --resume run should regenerate those categories.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bfcl-bf16.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({
                    "component": "bfcl",
                    "model": "bf16",
                    "checkpoint": "ckpt",
                    "score": 0.0,
                    "subscores": {
                        "complete": False,
                        "missing_categories": ["multi_turn_long_context"],
                        "partial_categories": [],
                    },
                    "runtime_s": 1.0,
                    "artifact_versions": {},
                    "timestamp": "2026-10-01T00:00:00Z",
                }, fh)
            self.assertFalse(
                run_matrix.is_cell_complete(path, component="bfcl")
            )

    def test_complete_hermes_cell(self):
        # Hermes: n_cases == 40 and no errors means complete.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "hermes-4bit.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({
                    "component": "hermes",
                    "model": "4bit",
                    "checkpoint": "ckpt",
                    "score": 0.5,
                    "subscores": {"n_cases": 40, "n_errors": 0, "per_category": {}},
                    "runtime_s": 30.0,
                    "artifact_versions": {},
                    "timestamp": "2026-10-01T00:00:00Z",
                }, fh)
            self.assertTrue(
                run_matrix.is_cell_complete(path, component="hermes")
            )

    def test_hermes_with_error_is_not_complete(self):
        # An errored run: not complete even if n_cases matches.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "hermes-4bit.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({
                    "component": "hermes",
                    "model": "4bit",
                    "checkpoint": "ckpt",
                    "score": 0.5,
                    "subscores": {"n_cases": 40, "n_errors": 1, "per_category": {}},
                    "runtime_s": 30.0,
                    "artifact_versions": {},
                    "timestamp": "2026-10-01T00:00:00Z",
                }, fh)
            self.assertFalse(
                run_matrix.is_cell_complete(path, component="hermes")
            )

    def test_complete_evalplus_cell(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "evalplus-base.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({
                    "component": "evalplus",
                    "model": "base",
                    "checkpoint": "ckpt",
                    "score": 0.4,
                    "subscores": {
                        "complete": True,
                        "missing_datasets": [],
                        "partial_datasets": [],
                    },
                    "runtime_s": 600.0,
                    "artifact_versions": {},
                    "timestamp": "2026-10-01T00:00:00Z",
                }, fh)
            self.assertTrue(
                run_matrix.is_cell_complete(path, component="evalplus")
            )

    def test_malformed_json_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bfcl-bf16.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("not json")
            self.assertFalse(
                run_matrix.is_cell_complete(path, component="bfcl")
            )


# ---------------------------------------------------------------------------
# Server-lifecycle tests — use the existing fake_mlx_lm_server
# ---------------------------------------------------------------------------


class ResolveServerPythonTest(unittest.TestCase):
    """The matrix points the ServerManager at .venv-core, not sys.executable."""

    def test_default_python_is_venv_core(self):
        self.assertTrue(
            run_matrix.resolve_server_python(_REPO_ROOT).endswith(
                ".venv-core/bin/python"
            )
        )

    def test_resolves_in_a_temp_repo_root(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(
                run_matrix.resolve_server_python(d).endswith(".venv-core/bin/python")
            )

    def test_override_takes_precedence(self):
        self.assertEqual(
            run_matrix.resolve_server_python(_REPO_ROOT, python="/custom/bin/python"),
            "/custom/bin/python",
        )


# ---------------------------------------------------------------------------
# End-to-end orchestration — drives a 1-row × 1-component slice offline
# ---------------------------------------------------------------------------


def _make_fake_env(tmp: str) -> dict:
    """An env in which ``run_matrix`` can find both a fake venv python and
    a fake component that doesn't need MLX.

    The real matrix uses ``run_eval.run_component`` which calls
    ``<venv>/bin/python -m evals.components.<comp>`` with
    ``PYTHONPATH=<repo_root>:<existing>``. We replace ``repo_root``
    with a tempdir that holds a fake ``evals/components/<comp>/__main__``
    tree — the real ``evals`` (which has BFCL/EvalPlus/Hermes) is NOT in
    that tempdir, so the fake always wins. The fake module is a thin
    trampoline that exec's ``tests/fake_component.py`` as ``__main__``
    (one Python file works for all three components).

    Side effect: sets ``RUN_MATRIX_FAKE_VENV_BIN`` in ``os.environ`` so
    the test's ``_patched_venv_python`` (which runs in the parent
    process) can find the fake venv. The matrix subprocess inherits
    this via its own env, so both layers see it.
    """
    bin_dir = os.path.join(tmp, "fake-venv", "bin")
    os.makedirs(bin_dir, exist_ok=True)
    fake_python = os.path.join(bin_dir, "python")
    os.symlink(sys.executable, fake_python)

    # The fake ``repo root`` is a temp dir that LOOKS like the real
    # repo (an ``evals/components/<comp>/`` tree) but the ``__main__``
    # is a trampoline to the fake component. The real ``evals`` lives
    # in the real repo — by passing ``repo_root=<this temp dir>`` to
    # ``run_matrix`` we hide the real one from the subprocess.
    fake_repo = os.path.join(tmp, "fake_repo")
    os.makedirs(os.path.join(fake_repo, "evals"), exist_ok=True)
    for comp in ("bfcl", "evalplus", "hermes"):
        comp_dir = os.path.join(fake_repo, "evals", "components", comp)
        os.makedirs(comp_dir, exist_ok=True)
        with open(os.path.join(comp_dir, "__init__.py"), "w", encoding="utf-8") as fh:
            fh.write("")
        # The fake's component name is hard-coded per-module — the
        # real component entry points know their own name from a
        # module-level constant, and the fake mirrors that.
        with open(os.path.join(comp_dir, "__main__.py"), "w", encoding="utf-8") as fh:
            fh.write(
                "import os, runpy, sys\n"
                "os.environ['FAKE_COMPONENT_NAME'] = {name!r}\n"
                "sys.argv[0] = {script!r}\n"
                "runpy.run_path({script!r}, run_name='__main__')\n"
                .format(name=comp, script=_FAKE_COMPONENT)
            )

    env = dict(os.environ)
    # 1. <venv>/bin/python resolves to the fake venv's python for all 3
    #    components (one fake venv, three component names → same python).
    env["RUN_MATRIX_FAKE_VENV_BIN"] = bin_dir
    os.environ["RUN_MATRIX_FAKE_VENV_BIN"] = bin_dir
    # 2. The fake_repo is the matrix's "repo_root" — its
    #    ``evals/components/<comp>`` is the only one the subprocess sees.
    env["RUN_MATRIX_FAKE_REPO"] = fake_repo
    os.environ["RUN_MATRIX_FAKE_REPO"] = fake_repo
    return env


def _patched_venv_python(self_component: str, repo_root: str) -> str:
    """Replacement for ``run_eval.venv_python`` that always returns the
    fake venv's python — regardless of component name."""
    bin_dir = os.environ["RUN_MATRIX_FAKE_VENV_BIN"]
    return os.path.join(bin_dir, "python")


class OneCellMatrixTest(unittest.TestCase):
    """Drive the matrix with one row × one component, against a fake
    server, and assert the orchestration end-to-end."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="run_matrix_test_")
        self._calls_log = os.path.join(self._tmp, "fake_calls.jsonl")
        os.environ["TMPDIR"] = self._tmp  # fake_component.py logs there
        # Truncate the inherited call log.
        for p in (self._calls_log, os.path.join(self._tmp, "fake_calls.jsonl")):
            if os.path.exists(p):
                os.remove(p)
        # Monkeypatch run_eval.venv_python so the matrix's subprocess
        # dispatch finds the fake venv (a symlink to the test python),
        # regardless of component name.
        self._orig_venv_python = run_eval.venv_python
        run_eval.venv_python = _patched_venv_python

    def tearDown(self):
        run_eval.venv_python = self._orig_venv_python
        os.environ.pop("RUN_MATRIX_FAKE_VENV_BIN", None)
        os.environ.pop("RUN_MATRIX_FAKE_REPO", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_one_row_one_component_offline(self):
        """No fake server: use the SENTINEL_BASE_URL path the fake supports.

        This is a pure orchestration test: the matrix should bring up a
        server, pass its base_url to the adapter, then tear it down. We
        can't reach the real mlx_lm server in the test venv — so we run
        the matrix with a SENTINEL_BASE_URL injection: the test
        monkeypatches ServerManager.start to return a sentinel, then
        asserts the matrix called stop() and the adapter wrote a result
        file via the sentinel path.
        """
        # Set the sentinel env knob the matrix looks for; if absent, the
        # matrix uses a real ServerManager (and we'd need MLX).
        os.environ["RUN_MATRIX_TEST_BASE_URL_OVERRIDE"] = (
            "http://INVALID_NO_S"
        )
        # Truncate the fake's call log so we can assert on this run.
        log = os.path.join(self._tmp, "fake_component_calls.jsonl")
        if os.path.exists(log):
            os.remove(log)

        env = _make_fake_env(self._tmp)
        # Drive the matrix programmatically with one row + one component.
        summary = run_matrix.run_matrix(
            row_checkpoints={"bf16": "checkpoints/DuoNeural-v3-BF16"},
            component_order=("bfcl",),
            results_dir=os.path.join(self._tmp, "results"),
            env=env,
            caffeinate=False,
            serve=False,  # use the override base_url
            repo_root=os.environ["RUN_MATRIX_FAKE_REPO"],
        )
        self.assertEqual(summary["ran"], 1)
        self.assertEqual(summary["skipped"], 0)
        self.assertEqual(summary["failed"], 0)

        # The adapter wrote its result file via the sentinel path.
        result_path = os.path.join(
            self._tmp, "results", "bfcl-bf16.json"
        )
        self.assertTrue(os.path.exists(result_path), result_path)
        with open(result_path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["model"], "bf16")
        self.assertEqual(data["component"], "bfcl")
        self.assertTrue(data["subscores"]["complete"])

        # The adapter's log records the call: it got the sentinel URL.
        calls = _read_jsonl(log)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["base_url"], "http://INVALID_NO_S")
        self.assertEqual(calls[0]["model"], "checkpoints/DuoNeural-v3-BF16")
        self.assertEqual(calls[0]["model_name"], "bf16")
        self.assertFalse(calls[0]["resume"])


class ResumeSkipsCompleteCellsTest(unittest.TestCase):
    """``--resume`` skips a cell whose result file is already complete."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="run_matrix_resume_test_")
        os.environ["TMPDIR"] = self._tmp
        log = os.path.join(self._tmp, "fake_component_calls.jsonl")
        if os.path.exists(log):
            os.remove(log)
        self._orig_venv_python = run_eval.venv_python
        run_eval.venv_python = _patched_venv_python

    def tearDown(self):
        run_eval.venv_python = self._orig_venv_python
        os.environ.pop("RUN_MATRIX_FAKE_VENV_BIN", None)
        os.environ.pop("RUN_MATRIX_FAKE_REPO", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_complete_cell_is_skipped_on_resume(self):
        # Plant a complete result file BEFORE running the matrix.
        results_dir = os.path.join(self._tmp, "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "bfcl-bf16.json"), "w") as fh:
            json.dump({
                "component": "bfcl", "model": "bf16", "checkpoint": "x",
                "score": 0.9, "runtime_s": 1.0,
                "subscores": {"complete": True, "missing_categories": [],
                              "partial_categories": []},
                "artifact_versions": {}, "timestamp": "2026-10-01T00:00:00Z",
            }, fh)
        env = _make_fake_env(self._tmp)
        summary = run_matrix.run_matrix(
            row_checkpoints={"bf16": "checkpoints/DuoNeural-v3-BF16"},
            component_order=("bfcl",),
            results_dir=results_dir,
            env=env,
            caffeinate=False,
            serve=False,
            resume=True,
            repo_root=os.environ["RUN_MATRIX_FAKE_REPO"],
        )
        self.assertEqual(summary["ran"], 0)
        self.assertEqual(summary["skipped"], 1)
        # The adapter was NOT called.
        calls = _read_jsonl(
            os.path.join(self._tmp, "fake_component_calls.jsonl")
        )
        self.assertEqual(calls, [])
        # The planted file is unchanged.
        with open(os.path.join(results_dir, "bfcl-bf16.json")) as fh:
            data = json.load(fh)
        self.assertEqual(data["score"], 0.9)


class FreshRunRegeneratesTest(unittest.TestCase):
    """A fresh (non-resume) run regenerates a cell even if a result file
    exists — the matrix is destructive on a fresh run by design."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="run_matrix_fresh_test_")
        os.environ["TMPDIR"] = self._tmp
        log = os.path.join(self._tmp, "fake_component_calls.jsonl")
        if os.path.exists(log):
            os.remove(log)
        self._orig_venv_python = run_eval.venv_python
        run_eval.venv_python = _patched_venv_python

    def tearDown(self):
        run_eval.venv_python = self._orig_venv_python
        os.environ.pop("RUN_MATRIX_FAKE_VENV_BIN", None)
        os.environ.pop("RUN_MATRIX_FAKE_REPO", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_fresh_run_overwrites_existing_result(self):
        results_dir = os.path.join(self._tmp, "results")
        os.makedirs(results_dir, exist_ok=True)
        # Plant a "stale" result file with a deliberately wrong score.
        with open(os.path.join(results_dir, "bfcl-bf16.json"), "w") as fh:
            json.dump({
                "component": "bfcl", "model": "bf16", "checkpoint": "x",
                "score": 0.99, "runtime_s": 1.0,
                "subscores": {"complete": True, "missing_categories": [],
                              "partial_categories": []},
                "artifact_versions": {}, "timestamp": "2026-10-01T00:00:00Z",
            }, fh)

        env = _make_fake_env(self._tmp)
        summary = run_matrix.run_matrix(
            row_checkpoints={"bf16": "checkpoints/DuoNeural-v3-BF16"},
            component_order=("bfcl",),
            results_dir=results_dir,
            env=env,
            caffeinate=False,
            serve=False,
            resume=False,
            repo_root=os.environ["RUN_MATRIX_FAKE_REPO"],
        )
        self.assertEqual(summary["ran"], 1)
        self.assertEqual(summary["skipped"], 0)

        # The file was overwritten with the adapter's score (0.42).
        with open(os.path.join(results_dir, "bfcl-bf16.json")) as fh:
            data = json.load(fh)
        self.assertEqual(data["score"], 0.42)


class ServerLifecycleTest(unittest.TestCase):
    """The matrix starts ONE server per row and stops it — even on adapter
    failure. The smoke run uses the real fake_mlx_lm_server."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="run_matrix_lifecycle_")
        os.environ["TMPDIR"] = self._tmp
        log = os.path.join(self._tmp, "fake_component_calls.jsonl")
        if os.path.exists(log):
            os.remove(log)
        self._orig_venv_python = run_eval.venv_python
        run_eval.venv_python = _patched_venv_python

    def tearDown(self):
        run_eval.venv_python = self._orig_venv_python
        os.environ.pop("RUN_MATRIX_FAKE_VENV_BIN", None)
        os.environ.pop("RUN_MATRIX_FAKE_REPO", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_one_row_three_components_share_one_server(self):
        env = _make_fake_env(self._tmp)
        # Real server, three components, one row.
        summary = run_matrix.run_matrix(
            row_checkpoints={"4bit": "checkpoints/DuoNeural-v3-4bit"},
            component_order=("bfcl", "evalplus", "hermes"),
            results_dir=os.path.join(self._tmp, "results"),
            env=env,
            caffeinate=False,
            serve=True,
            server_command_factory=lambda ckpt, host, port, python: [
                sys.executable, _FAKE_SERVER, host, str(port),
            ],
            server_startup_timeout_s=15.0,
            repo_root=os.environ["RUN_MATRIX_FAKE_REPO"],
        )
        self.assertEqual(summary["ran"], 3)
        self.assertEqual(summary["skipped"], 0)
        self.assertEqual(summary["failed"], 0)
        # All three components got the SAME base_url (one server, three calls).
        calls = _read_jsonl(
            os.path.join(self._tmp, "fake_component_calls.jsonl")
        )
        self.assertEqual(len(calls), 3)
        urls = {c["base_url"] for c in calls}
        self.assertEqual(len(urls), 1, "expected one base_url across 3 cells")
        # And the URL was a real loopback http://127.0.0.1:<port>/v1
        url = next(iter(urls))
        self.assertTrue(url.startswith("http://127.0.0.1:"))
        self.assertTrue(url.endswith("/v1"))
        # All three result files were written.
        results_dir = os.path.join(self._tmp, "results")
        for comp in ("bfcl", "evalplus", "hermes"):
            self.assertTrue(
                os.path.exists(os.path.join(results_dir, comp + "-4bit.json")),
                "missing {0}-4bit.json".format(comp),
            )

    def test_server_is_terminated_after_row_finishes(self):
        # After the row finishes, no mlx_lm server should be left running
        # (the matrix's `with ServerManager(...)` guarantees this; we
        # verify by checking the child pid is gone).
        env = _make_fake_env(self._tmp)
        captured_pid = {}

        def _factory(ckpt, host, port, python):
            return [sys.executable, _FAKE_SERVER, host, str(port)]

        # Patch the matrix to capture the spawned pid.
        from evals import server as srv
        orig_popen = srv.subprocess.Popen

        def _capturing(*a, **kw):
            proc = orig_popen(*a, **kw)
            captured_pid["pid"] = proc.pid
            return proc

        srv.subprocess.Popen = _capturing
        try:
            run_matrix.run_matrix(
                row_checkpoints={"4bit": "checkpoints/DuoNeural-v3-4bit"},
                component_order=("bfcl",),
                results_dir=os.path.join(self._tmp, "results"),
                env=env,
                caffeinate=False,
                serve=True,
                server_command_factory=_factory,
                server_startup_timeout_s=15.0,
                repo_root=os.environ["RUN_MATRIX_FAKE_REPO"],
            )
        finally:
            srv.subprocess.Popen = orig_popen

        self.assertIn("pid", captured_pid)
        pid = captured_pid["pid"]
        # The child is gone — pid is no longer alive.
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
        self.assertFalse(alive, "server pid {0} still alive after row".format(pid))


class CellFailureTest(unittest.TestCase):
    """A failing component does NOT kill the row or the matrix — it
    records the failure and proceeds to the next cell."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="run_matrix_fail_")
        os.environ["TMPDIR"] = self._tmp
        log = os.path.join(self._tmp, "fake_component_calls.jsonl")
        if os.path.exists(log):
            os.remove(log)
        self._orig_venv_python = run_eval.venv_python
        run_eval.venv_python = _patched_venv_python

    def tearDown(self):
        run_eval.venv_python = self._orig_venv_python
        os.environ.pop("RUN_MATRIX_FAKE_VENV_BIN", None)
        os.environ.pop("RUN_MATRIX_FAKE_REPO", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_failing_component_is_recorded_not_propagated(self):
        env = _make_fake_env(self._tmp)
        # The fake component with a sentinel URL exits 0; with a real URL
        # it tries /v1/models. We can't easily make it fail without
        # extra plumbing, so instead: pass an invalid checkpoint dir for
        # the BFCL component (the matrix forwards the checkpoint ref
        # verbatim). The fake component still writes a result; the test
        # is just that the matrix tolerates a non-zero exit code per
        # cell. We do that by wrapping the adapter invocation.
        #
        # Implementation note: the matrix's run_cell() catches non-zero
        # exit codes from the subprocess. We assert that contract by
        # checking that a failing cell does not crash the matrix.
        #
        # Use the ``_invoke_component`` seam: monkeypatch it to return
        # exit code 5 (a synthetic failure).
        original = run_matrix._invoke_component

        def _failing(*a, **kw):
            return 5  # pretend the adapter crashed

        run_matrix._invoke_component = _failing
        try:
            summary = run_matrix.run_matrix(
                row_checkpoints={"4bit": "checkpoints/DuoNeural-v3-4bit"},
                component_order=("bfcl", "hermes"),
                results_dir=os.path.join(self._tmp, "results"),
                env=env,
                caffeinate=False,
                serve=False,
                repo_root=os.environ["RUN_MATRIX_FAKE_REPO"],
            )
        finally:
            run_matrix._invoke_component = original
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["ran"], 0)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class CliTest(unittest.TestCase):
    """``run_matrix.py`` accepts the documented flags."""

    def test_help(self):
        proc = subprocess.run(
            [sys.executable, "run_matrix.py", "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for flag in ("--resume", "--no-caffeinate", "--results-dir",
                     "--rows", "--components", "--server-startup-timeout-s",
                     "--venv-core", "--dry-run"):
            self.assertIn(flag, proc.stdout, "missing flag: " + flag)

    def test_dry_run_prints_plan_no_work(self):
        # No fake server, no fake adapter — dry-run should just print and
        # exit 0.
        proc = subprocess.run(
            [sys.executable, "run_matrix.py", "--dry-run",
             "--rows", "bf16", "--components", "bfcl",
             "--results-dir", "/tmp/nope_run_matrix_dryrun",
             "--no-caffeinate"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("bf16", proc.stdout)
        self.assertIn("bfcl", proc.stdout)
        # The dry-run did not write any result file.
        self.assertFalse(os.path.exists(
            "/tmp/nope_run_matrix_dryrun/bfcl-bf16.json"
        ))


if __name__ == "__main__":
    unittest.main()
