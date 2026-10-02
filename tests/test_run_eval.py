"""Tests for run_eval.py orchestration + the result schema (ticket #16).

Covers the per-venv dispatch wiring, the locked result schema, and the stub
emit path. A fake ``.venv-bfcl`` (symlinked interpreter) lets the real
subprocess dispatch run end-to-end without building three venvs. Stdlib
``unittest`` (also pytest-compatible).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import run_eval  # noqa: E402
from evals import results as results_mod  # noqa: E402


class VenvPythonTest(unittest.TestCase):
    def test_component_venv_paths(self):
        self.assertTrue(run_eval.venv_python("bfcl").endswith(".venv-bfcl/bin/python"))
        self.assertTrue(run_eval.venv_python("evalplus").endswith(".venv-evalplus/bin/python"))
        self.assertTrue(run_eval.venv_python("hermes").endswith(".venv-core/bin/python"))

    def test_every_component_has_a_venv(self):
        for component in results_mod.COMPONENTS:
            self.assertIn(component, run_eval.COMPONENT_VENV)


class ResultSchemaTest(unittest.TestCase):
    def test_new_result_has_all_locked_keys(self):
        r = results_mod.new_result(
            component="bfcl", model="bf16", checkpoint="ckpt", score=0.5, runtime_s=1.0
        )
        for key in results_mod.RESULT_KEYS:
            self.assertIn(key, r)
        self.assertEqual(results_mod.validate_result(r), [])

    def test_schema_matches_runner_docstring(self):
        # the schema locked in evals/runner.py (ticket #6, resolution 4a)
        locked = {
            "component", "model", "checkpoint", "score", "subscores",
            "runtime_s", "artifact_versions", "timestamp",
        }
        self.assertEqual(set(results_mod.RESULT_KEYS), locked)

    def test_unknown_component_rejected(self):
        with self.assertRaises(ValueError):
            results_mod.new_result(
                component="nope", model="m", checkpoint="c", score=0.0, runtime_s=0.0
            )

    def test_validate_flags_missing_key(self):
        r = results_mod.new_result(
            component="hermes", model="m", checkpoint="c", score=0.0, runtime_s=0.0
        )
        del r["score"]
        errors = results_mod.validate_result(r)
        self.assertTrue(any("score" in e for e in errors))

    def test_write_result_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            r = results_mod.new_result(
                component="evalplus", model="4bit", checkpoint="ckpt",
                score=0.42, subscores={"humaneval+": 0.4}, runtime_s=3.0,
            )
            path = results_mod.write_result(r, d)
            self.assertTrue(path.endswith("evalplus-4bit.json"))
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["score"], 0.42)
            self.assertEqual(loaded["subscores"]["humaneval+"], 0.4)
            self.assertEqual(results_mod.validate_result(loaded), [])

    def test_write_result_rejects_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                results_mod.write_result({"component": "bfcl"}, d)


class BuildCommandTest(unittest.TestCase):
    def test_real_adapter_dispatch(self):
        argv = run_eval.build_component_command("bfcl", "checkpoints/DuoNeural-v3-BF16")
        self.assertEqual(argv[1:3], ["-m", "evals.components"])
        self.assertIn("--component", argv)
        self.assertIn("bfcl", argv)
        self.assertIn("--model", argv)
        self.assertIn("checkpoints/DuoNeural-v3-BF16", argv)

    def test_resume_flag_forwarded(self):
        argv = run_eval.build_component_command("bfcl", "m", resume=True)
        self.assertIn("--resume", argv)

    def test_missing_venv_returns_error(self):
        # no .venv-bfcl in the repo on the build host -> clear error, exit 2
        code = run_eval.run_component("bfcl", "m")
        self.assertEqual(code, 2)


class StubDispatchTest(unittest.TestCase):
    """End-to-end: real subprocess into a fake venv runs the component entry."""

    def _make_fake_venv(self, root, name):
        bin_dir = os.path.join(root, name, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        os.symlink(sys.executable, os.path.join(bin_dir, "python"))

    def test_component_entry_emits_valid_result(self):
        # run `python -m evals.components` directly (no venv needed)
        proc = subprocess.run(
            [sys.executable, "-m", "evals.components", "--component", "hermes", "--model", "bf16"],
            cwd=_REPO_ROOT,
            env=dict(os.environ, PYTHONPATH=_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        path = os.path.join(_REPO_ROOT, "evals", "results", "hermes-bf16.json")
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(results_mod.validate_result(loaded), [])
        self.assertEqual(loaded["component"], "hermes")
        os.remove(path)  # keep the tree clean

    def test_full_dispatch_through_fake_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            # copy the harness into a scratch root with a fake venv
            for item in ("evals", "run_eval.py"):
                src = os.path.join(_REPO_ROOT, item)
                dst = os.path.join(tmp, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("results", "__pycache__"))
                else:
                    shutil.copy2(src, dst)
            self._make_fake_venv(tmp, ".venv-bfcl")
            code = run_eval.run_component("bfcl", "4bit", repo_root=tmp)
            self.assertEqual(code, 0)
            path = os.path.join(tmp, "evals", "results", "bfcl-4bit.json")
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(results_mod.validate_result(loaded), [])
            self.assertEqual(loaded["component"], "bfcl")


if __name__ == "__main__":
    unittest.main()
