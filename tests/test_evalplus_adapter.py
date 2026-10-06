"""Offline tests for the EvalPlus adapter's stdlib seams (ticket #18).

These run in the core venv (no ``evalplus``, no Docker). They exercise the
runner's argv builders, the command resolvers, the normalizer's parsing of
``*_eval_results.json`` (the real on-disk format), and the end-to-end ``run()``
pipeline with the codegen/docker subprocesses stubbed out — never a real
server or model.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from evals.components.evalplus import normalize, runner  # noqa: E402
from evals import results as results_mod  # noqa: E402


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _eval_results(dataset, n=4):
    """A small ``*_eval_results.json`` with a known pass mix.

    base_status passes on even indices (n/2 of n); plus_status passes only on
    multiples of 4 (n/4 of n), matching EvalPlus's base∩plus intersection.
    """
    prefix = "HumanEval/" if dataset == "humaneval" else "Mbpp/"
    eval_dict = {}
    for i in range(n):
        base = "pass" if i % 2 == 0 else "fail"
        plus = "pass" if i % 4 == 0 else "fail"
        eval_dict["{0}{1}".format(prefix, i)] = [{
            "task_id": "{0}{1}".format(prefix, i),
            "solution": "def f(): pass",
            "base_status": base,
            "plus_status": plus,
            "base_fail_tests": [],
            "plus_fail_tests": [],
        }]
    return {"date": "2026-01-01 00:00", "hash": "deadbeef", "eval": eval_dict}


class CodegenArgvTest(unittest.TestCase):
    def test_fresh_run_uses_noresume(self):
        argv = runner.build_codegen_argv(
            "default_model", "humaneval",
            base_url="http://127.0.0.1:9/v1", root="/tmp/r", resume=False,
        )
        self.assertEqual(argv[0], "default_model")
        self.assertEqual(argv[1], "humaneval")
        self.assertIn("--backend", argv)
        self.assertIn("openai", argv)
        i = argv.index("--base-url")
        self.assertEqual(argv[i + 1], "http://127.0.0.1:9/v1")
        self.assertIn("--greedy", argv)
        self.assertIn("--noresume", argv)
        self.assertNotIn("--resume", argv)
        j = argv.index("--root")
        self.assertEqual(argv[j + 1], "/tmp/r")

    def test_resume_run_uses_resume(self):
        argv = runner.build_codegen_argv(
            "default_model", "mbpp",
            base_url="http://127.0.0.1:9/v1", root="/tmp/r", resume=True,
        )
        self.assertIn("--resume", argv)
        self.assertNotIn("--noresume", argv)


class DockerEvalArgvTest(unittest.TestCase):
    def test_docker_run_mount_and_samples(self):
        argv = runner.build_docker_eval_argv("/tmp/root", "humaneval", "x.jsonl")
        self.assertEqual(argv[0], "run")
        self.assertIn("--rm", argv)
        i = argv.index("-v")
        self.assertEqual(argv[i + 1], "/tmp/root:/app")
        self.assertIn(runner.DOCKER_IMAGE, argv)
        j = argv.index("evalplus.evaluate")
        self.assertEqual(argv[j + 1], "humaneval")
        k = argv.index("--samples")
        self.assertEqual(argv[k + 1], "/app/humaneval/x.jsonl")


class ResolveCommandTest(unittest.TestCase):
    def test_resolve_evalplus_console_script_next_to_python(self):
        with tempfile.TemporaryDirectory() as d:
            bin_dir = os.path.join(d, "bin")
            os.makedirs(bin_dir)
            with open(os.path.join(bin_dir, "evalplus.codegen"), "w") as fh:
                fh.write("#!/bin/sh\n")
            os.chmod(os.path.join(bin_dir, "evalplus.codegen"), 0o755)
            python = os.path.join(bin_dir, "python")
            cmd = runner.resolve_evalplus_command("evalplus.codegen", python=python)
            self.assertEqual(cmd, [os.path.join(bin_dir, "evalplus.codegen")])

    def test_resolve_evalplus_falls_back_to_module(self):
        # No console script next to a bare python and not on PATH -> module form.
        with tempfile.TemporaryDirectory() as d:
            python = os.path.join(d, "python")
            cmd = runner.resolve_evalplus_command("evalplus.evaluate", python=python)
            self.assertEqual(cmd, [python, "-m", "evalplus.evaluate"])

    def test_resolve_docker_ends_in_docker(self):
        # Whether it resolves to a full path (Docker Desktop's CLI) or the
        # bare `docker` fallback, the final argv token's basename is `docker`.
        cmd = runner.resolve_docker_command()
        self.assertTrue(cmd)
        self.assertEqual(os.path.basename(cmd[-1]), "docker")


class NormalizeParseTest(unittest.TestCase):
    def test_compute_pass_at_1_matches_evalplus(self):
        # base passes 2/4 = 0.5; plus passes 1/4 = 0.25 (base∩plus).
        base, plus, n = normalize.compute_pass_at_1(_eval_results("humaneval", 4))
        self.assertEqual(n, 4)
        self.assertAlmostEqual(base, 0.5)
        self.assertAlmostEqual(plus, 0.25)

    def test_compute_pass_at_1_empty(self):
        base, plus, n = normalize.compute_pass_at_1({"eval": {}})
        self.assertEqual(n, 0)
        self.assertEqual(base, 0.0)
        self.assertEqual(plus, 0.0)

    def test_compute_pass_at_1_null_plus_is_not_pass(self):
        # plus_status null (or missing) must not count toward the + score.
        data = {"eval": {"HumanEval/0": [{
            "task_id": "HumanEval/0", "solution": "def f(): pass",
            "base_status": "pass", "plus_status": None,
        }]}}
        base, plus, n = normalize.compute_pass_at_1(data)
        self.assertAlmostEqual(base, 1.0)
        self.assertAlmostEqual(plus, 0.0)


class NormalizeFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def test_find_samples_file_skips_raw(self):
        _write_jsonl(os.path.join(self.root, "humaneval", "m_openai_temp_0.0.jsonl"),
                     [{"task_id": "HumanEval/0", "solution": "def f(): pass"}])
        _write_jsonl(os.path.join(self.root, "humaneval", "m_openai_temp_0.0.raw.jsonl"),
                     [{"task_id": "HumanEval/0", "solution": "raw"}])
        found = normalize.find_samples_file(self.root, "humaneval")
        self.assertTrue(found.endswith("m_openai_temp_0.0.jsonl"))
        self.assertNotIn(".raw.", found)

    def test_find_samples_file_missing(self):
        self.assertIsNone(normalize.find_samples_file(self.root, "mbpp"))

    def test_clear_samples_removes_both_jsonl_and_raw(self):
        _write_jsonl(os.path.join(self.root, "humaneval", "m_openai_temp_0.0.jsonl"),
                     [{"task_id": "HumanEval/0"}])
        _write_jsonl(os.path.join(self.root, "humaneval", "m_openai_temp_0.0.raw.jsonl"),
                     [{"task_id": "HumanEval/0"}])
        normalize.clear_samples(self.root, "humaneval")
        self.assertEqual(os.listdir(os.path.join(self.root, "humaneval")), [])

    def test_clear_eval_results(self):
        _write(os.path.join(self.root, "humaneval", "m_openai_temp_0.0_eval_results.json"),
               _eval_results("humaneval"))
        normalize.clear_eval_results(self.root, "humaneval")
        self.assertEqual(os.listdir(os.path.join(self.root, "humaneval")), [])

    def test_normalize_mean_of_plus(self):
        # humaneval: base .5 plus .25; mbpp: base .5 plus .25 -> score .25.
        _write(os.path.join(self.root, "humaneval", "h_openai_temp_0.0_eval_results.json"),
               _eval_results("humaneval"))
        _write(os.path.join(self.root, "mbpp", "m_openai_temp_0.0_eval_results.json"),
               _eval_results("mbpp"))
        score, per_dataset, missing, partial = normalize.normalize(self.root)
        self.assertAlmostEqual(score, 0.25)
        self.assertEqual(set(per_dataset), {"humaneval", "mbpp"})
        self.assertEqual(missing, [])
        # 4 tasks < 164/378 expected -> both flagged partial.
        self.assertEqual(partial, ["humaneval", "mbpp"])
        self.assertEqual(per_dataset["humaneval"]["hash"], "deadbeef")

    def test_normalize_none_when_nothing_scored(self):
        score, per_dataset, missing, partial = normalize.normalize(self.root)
        self.assertIsNone(score)
        self.assertEqual(per_dataset, {})
        self.assertEqual(set(missing), set(normalize.DATASETS))
        self.assertEqual(partial, [])


class RunPipelineTest(unittest.TestCase):
    """End-to-end run() with the codegen/docker subprocess seams stubbed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_root = os.path.join(self.tmp.name, "runs")
        self.results_dir = os.path.join(self.tmp.name, "results")

    def _fake_subprocess(self, calls):
        def fake(argv):
            args = list(argv)
            calls.append(args)
            if "evalplus.evaluate" in args:
                # Docker eval step: write the _eval_results.json for the dataset.
                dataset = next(t for t in args if t in ("humaneval", "mbpp"))
                host_root = args[args.index("-v") + 1].split(":")[0]
                samples = args[args.index("--samples") + 1]
                base = os.path.basename(samples)
                result_file = os.path.join(
                    host_root, dataset, base.replace(".jsonl", "_eval_results.json")
                )
                _write(result_file, _eval_results(dataset))
            else:
                # codegen step: write a sample .jsonl for the dataset.
                dataset = next(t for t in args if t in ("humaneval", "mbpp"))
                host_root = args[args.index("--root") + 1]
                sample = os.path.join(
                    host_root, dataset, "default_model_openai_temp_0.0.jsonl"
                )
                _write_jsonl(sample, [{"task_id": "t", "solution": "def f(): pass"}])
        return fake

    def _run(self, *, resume=False):
        calls = []
        import types
        from unittest import mock
        with mock.patch.object(runner, "run_subprocess",
                               side_effect=self._fake_subprocess(calls)):
            result = runner.run(
                "checkpoints/DuoNeural-v3-4bit",
                base_url="http://127.0.0.1:9/v1",
                model_name="4bit",
                resume=resume,
                run_root=self.run_root,
                results_dir=self.results_dir,
            )
        return result, calls

    def test_run_emits_normalized_result(self):
        result, calls = self._run(resume=False)

        self.assertEqual(result["component"], "evalplus")
        self.assertEqual(result["model"], "4bit")
        self.assertEqual(result["checkpoint"], "checkpoints/DuoNeural-v3-4bit")
        # score = mean(humaneval+ .25, mbpp+ .25) = .25
        self.assertAlmostEqual(result["score"], 0.25)
        subs = result["subscores"]
        self.assertAlmostEqual(subs["humaneval_plus_pass1"], 0.25)
        self.assertAlmostEqual(subs["humaneval_base_pass1"], 0.5)
        self.assertAlmostEqual(subs["mbpp_plus_pass1"], 0.25)
        self.assertEqual(subs["dataset_versions"]["humaneval"], "v0.1.10")
        self.assertEqual(subs["dataset_hashes"]["humaneval"], "deadbeef")
        self.assertEqual(subs["decoding"], {"greedy": True, "temperature": 0.0, "n_samples": 1})
        self.assertEqual(subs["missing_datasets"], [])
        self.assertEqual(subs["partial_datasets"], ["humaneval", "mbpp"])
        self.assertFalse(subs["complete"])
        # Two codegen + two docker-eval subprocess calls.
        self.assertEqual(len(calls), 4)
        codegen_calls = [c for c in calls if "evalplus.evaluate" not in c]
        self.assertEqual(len(codegen_calls), 2)
        # Fresh run: codegen used --noresume.
        self.assertTrue(all("--noresume" in c for c in codegen_calls))

        # Result file written + schema-valid.
        out = os.path.join(self.results_dir, "evalplus-4bit.json")
        self.assertTrue(os.path.exists(out))
        with open(out, encoding="utf-8") as fh:
            on_disk = json.load(fh)
        self.assertEqual(results_mod.validate_result(on_disk), [])

    def test_resume_preserves_samples_and_uses_resume_flag(self):
        # Pre-seed a sample file so resume appends rather than clearing.
        os.makedirs(os.path.join(self.run_root, "humaneval"), exist_ok=True)
        _write_jsonl(os.path.join(self.run_root, "humaneval", "default_model_openai_temp_0.0.jsonl"),
                     [{"task_id": "HumanEval/0", "solution": "def f(): pass"}])

        _result, calls = self._run(resume=True)
        codegen_calls = [c for c in calls if "evalplus.evaluate" not in c]
        self.assertTrue(all("--resume" in c and "--noresume" not in c for c in codegen_calls))

    def test_missing_sample_raises(self):
        import types
        from unittest import mock

        def fake(argv):
            # codegen writes nothing (simulates a generation failure).
            return None

        with mock.patch.object(runner, "run_subprocess", side_effect=fake):
            with self.assertRaises(RuntimeError):
                runner.run(
                    "checkpoints/DuoNeural-v3-4bit",
                    base_url="http://127.0.0.1:9/v1",
                    model_name="4bit",
                    run_root=self.run_root,
                    results_dir=self.results_dir,
                )


if __name__ == "__main__":
    unittest.main()
