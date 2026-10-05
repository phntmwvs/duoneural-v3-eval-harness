"""Tests for run_eval.py orchestration + the result schema (ticket #16).

Covers the per-venv dispatch wiring, the locked result schema, and the stub
emit path. A fake ``.venv-bfcl`` (symlinked interpreter) lets the real
subprocess dispatch run end-to-end without building three venvs. Stdlib
``unittest`` (also pytest-compatible).
"""

from __future__ import annotations

import json
import os
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

    def test_validate_rejects_bool_score_and_runtime(self):
        # bool is a subclass of int; a True/False score must not validate.
        for bad in ({"score": True}, {"runtime_s": False}):
            r = results_mod.new_result(
                component="bfcl", model="m", checkpoint="c", score=0.0, runtime_s=0.0
            )
            r.update(bad)
            self.assertTrue(results_mod.validate_result(r), "expected violation for {0}".format(bad))


class ResultPathTest(unittest.TestCase):
    def test_hf_id_with_slash_does_not_escape_or_crash(self):
        # regression: an HF id (LiquidAI/LFM2.5-8B-A1B) must not be treated as
        # a path separator, and must stay inside the results dir.
        with tempfile.TemporaryDirectory() as d:
            r = results_mod.new_result(
                component="bfcl",
                model="base",
                checkpoint="LiquidAI/LFM2.5-8B-A1B",
                score=0.0,
                runtime_s=0.0,
            )
            path = results_mod.write_result(r, d)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(os.path.dirname(os.path.abspath(path)), os.path.abspath(d))

    def test_model_name_with_slash_is_slugged(self):
        path = results_mod.result_path("/out", "bfcl", "LiquidAI/LFM2.5-8B-A1B")
        self.assertEqual(os.path.dirname(path), "/out")
        self.assertIn("LiquidAI__LFM2.5-8B-A1B", os.path.basename(path))

    def test_dotdot_cannot_traverse(self):
        path = results_mod.result_path("/out", "bfcl", "../evil")
        self.assertEqual(os.path.dirname(path), "/out")
        self.assertNotIn("..", os.path.basename(path))

    def test_path_stays_in_results_dir(self):
        for nasty in ("a/b", "..", "../..", "a\\b", "..\\.."):
            path = results_mod.result_path("/out", "hermes", nasty)
            self.assertEqual(os.path.dirname(path), "/out", "nasty={0!r}".format(nasty))


class BuildCommandTest(unittest.TestCase):
    def test_real_adapter_dispatch(self):
        # BFCL (#17) dispatches to its real adapter entry point.
        argv = run_eval.build_component_command(
            "bfcl", "checkpoints/DuoNeural-v3-BF16", model_name="bf16"
        )
        self.assertEqual(argv[1:3], ["-m", "evals.components.bfcl"])
        self.assertNotIn("--component", argv)  # real adapter, no selector flag
        self.assertIn("--model", argv)
        self.assertIn("checkpoints/DuoNeural-v3-BF16", argv)
        self.assertIn("--model-name", argv)
        self.assertIn("bf16", argv)

    def test_stub_components_still_use_selector(self):
        # evalplus (#18) / hermes (#19) still run the scaffold stub, which
        # takes a leading --component selector.
        for component in ("evalplus", "hermes"):
            argv = run_eval.build_component_command(component, "m", model_name="m")
            self.assertEqual(argv[1:3], ["-m", "evals.components"])
            self.assertIn("--component", argv)
            self.assertIn(component, argv)

    def test_resume_flag_forwarded(self):
        argv = run_eval.build_component_command("bfcl", "m", model_name="m", resume=True)
        self.assertIn("--resume", argv)

    def test_base_url_forwarded_when_given(self):
        argv = run_eval.build_component_command(
            "bfcl", "m", model_name="m", base_url="http://127.0.0.1:8080/v1"
        )
        self.assertIn("--base-url", argv)
        i = argv.index("--base-url")
        self.assertEqual(argv[i + 1], "http://127.0.0.1:8080/v1")

    def test_base_url_omitted_when_not_given(self):
        argv = run_eval.build_component_command("bfcl", "m", model_name="m")
        self.assertNotIn("--base-url", argv)

    def test_model_name_defaults_to_model(self):
        argv = run_eval.build_component_command("bfcl", "bf16", model_name="bf16")
        i = argv.index("--model-name")
        self.assertEqual(argv[i + 1], "bf16")

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
        # The scaffold's stub-dispatch prove-out is superseded for bfcl: #17
        # wires it to its real adapter, which imports bfcl_eval and cannot run
        # in a bare symlinked venv. What we still verify offline is that the
        # per-venv dispatch resolves the correct interpreter and entry point.
        with tempfile.TemporaryDirectory() as tmp:
            self._make_fake_venv(tmp, ".venv-bfcl")
            argv = run_eval.build_component_command(
                "bfcl", "4bit", model_name="4bit", repo_root=tmp
            )
            # Resolves the fake venv's python and the real BFCL entry point.
            self.assertTrue(argv[0].endswith(".venv-bfcl/bin/python"))
            self.assertEqual(argv[1:3], ["-m", "evals.components.bfcl"])
            self.assertIn("4bit", argv)

    def test_hf_id_checkpoint_with_distinct_row_name(self):
        # regression (PR #21 review): --model LiquidAI/LFM2.5-8B-A1B must not
        # crash dispatch; model (row name) and checkpoint (ref) stay distinct.
        # (bfcl is now a real adapter, so assert the argv rather than running
        # it in a fake venv that lacks bfcl_eval.)
        with tempfile.TemporaryDirectory() as tmp:
            self._make_fake_venv(tmp, ".venv-bfcl")
            argv = run_eval.build_component_command(
                "bfcl", "LiquidAI/LFM2.5-8B-A1B", model_name="base", repo_root=tmp
            )
            self.assertEqual(argv[1:3], ["-m", "evals.components.bfcl"])
            i = argv.index("--model")
            self.assertEqual(argv[i + 1], "LiquidAI/LFM2.5-8B-A1B")
            j = argv.index("--model-name")
            self.assertEqual(argv[j + 1], "base")

    def test_emit_stub_keeps_model_and_checkpoint_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = run_eval.run_component(
                "hermes", "LiquidAI/LFM2.5-8B-A1B",
                model_name="base", emit_stub=True, repo_root=tmp,
            )
            self.assertEqual(code, 0)
            path = os.path.join(tmp, "evals", "results", "hermes-base.json")
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["model"], "base")
            self.assertEqual(loaded["checkpoint"], "LiquidAI/LFM2.5-8B-A1B")


if __name__ == "__main__":
    unittest.main()
