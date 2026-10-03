"""Offline tests for the BFCL adapter's stdlib seams (ticket #17).

These run in the core venv (no ``bfcl_eval``). They exercise the runner's
argv builders, the ``--resume`` skip planner, the thrash-guard summarizer, and
the normalizer's parsing of BFCL's result/score file formats — all against
temporary files and a fake ``bfcl`` CLI, never a real server or model.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.components.bfcl import normalize, runner  # noqa: E402


def _write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _score_records(accuracy, correct, total):
    # Mirrors save_eval_results JSONL: header record first, then per-entry rows.
    return [{"accuracy": accuracy, "correct_count": correct, "total_count": total},
            {"id": "x", "valid": True}]


def _result_payload(turns_by_id):
    # One record per id, keyed by id. Each value maps id -> a list of step
    # counts, one per turn (mirrors BFCL: result[turn] is a list of steps).
    return {
        tid: {"id": tid,
              "result": [[{"name": "f"}] * steps for steps in steps_per_turn]}
        for tid, steps_per_turn in turns_by_id.items()
    }


class ArgvBuilderTest(unittest.TestCase):
    def test_generate_argv_core_flags(self):
        argv = runner.build_generate_argv(
            "duoneural-v3-mlx-fc", ("multi_turn_base", "multi_turn_miss_func")
        )
        self.assertEqual(argv[0], "generate")
        self.assertIn("--skip-server-setup", argv)
        self.assertIn("--include-input-log", argv)
        joined = " ".join(argv)
        self.assertIn("--test-category multi_turn_base,multi_turn_miss_func", joined)
        self.assertIn("--model duoneural-v3-mlx-fc", joined)
        # No --run-ids unless requested
        self.assertNotIn("--run-ids", argv)
        self.assertNotIn("--allow-overwrite", argv)

    def test_generate_argv_overwrite_and_result_dir(self):
        argv = runner.build_generate_argv(
            "k", ("multi_turn_base",), allow_overwrite=True,
            result_dir="/tmp/res",
        )
        self.assertIn("--allow-overwrite", argv)
        self.assertIn("/tmp/res", argv)
        # --run-ids is no longer emitted: resume uses BFCL's native skip.
        self.assertNotIn("--run-ids", argv)

    def test_generate_argv_caps_threads_for_mlx_lm(self):
        # BFCL's OSS default of 100 threads kills the single-threaded mlx_lm
        # server (observed live); the adapter defaults to serial generation.
        argv = runner.build_generate_argv("k", ("multi_turn_base",))
        i = argv.index("--num-threads")
        self.assertEqual(argv[i + 1], str(runner.DEFAULT_NUM_THREADS))
        self.assertEqual(runner.DEFAULT_NUM_THREADS, 1)
        custom = runner.build_generate_argv("k", ("multi_turn_base",),
                                            num_threads=4)
        self.assertEqual(custom[custom.index("--num-threads") + 1], "4")

    def test_evaluate_argv(self):
        # Fresh run: no --partial-eval (checker enforces completeness, M1).
        argv = runner.build_evaluate_argv(
            "k", ("multi_turn_base",), result_dir="/r", score_dir="/s"
        )
        self.assertEqual(argv[0], "evaluate")
        self.assertIn("--model", argv)
        self.assertIn("--score-dir", argv)
        self.assertIn("/s", argv)
        self.assertNotIn("--partial-eval", argv)
        # Resume run: --partial-eval passed (result files may hold id-subsets).
        argv_resume = runner.build_evaluate_argv(
            "k", ("multi_turn_base",), partial_eval=True
        )
        self.assertIn("--partial-eval", argv_resume)


class ThrashGuardTest(unittest.TestCase):
    @staticmethod
    def _rec(max_steps, turns=1):
        return {"turns": turns, "steps_per_turn": [max_steps],
                "max_steps": max_steps}

    def test_no_counts(self):
        t = runner.compute_thrash({})
        self.assertFalse(t["flag"])
        self.assertEqual(t["max_steps_observed"], 0)
        self.assertEqual(t["cap_hit_ids"], [])

    def test_below_cap_no_flag(self):
        t = runner.compute_thrash(
            {"a": self._rec(3), "b": self._rec(20)}, cap=20)
        self.assertFalse(t["flag"])
        self.assertEqual(t["max_steps_observed"], 20)
        self.assertEqual(t["cap_hit_ids"], [])

    def test_over_cap_flags(self):
        # BFCL force-quits a turn after 20 steps; 21 = 20 attempts + the
        # forced terminal message => the model never answered in that turn.
        t = runner.compute_thrash(
            {"ok": self._rec(4), "thrashy": self._rec(21, turns=3)}, cap=20)
        self.assertTrue(t["flag"])
        self.assertEqual(t["cap_hit_ids"], ["thrashy"])
        self.assertEqual(t["step_counts"]["thrashy"]["max_steps"], 21)


class ResumeSemanticsTest(unittest.TestCase):
    """--resume delegates to BFCL's native per-id skip (decision 7a).

    The adapter no longer computes an id plan: on resume it simply omits
    ``--allow-overwrite`` so BFCL reloads existing result files and generates
    only missing ids. These tests assert that contract at the argv level.
    """

    def test_resume_omits_allow_overwrite(self):
        argv = runner.build_generate_argv("k", ("multi_turn_base",),
                                          allow_overwrite=False)
        self.assertNotIn("--allow-overwrite", argv)

    def test_fresh_run_passes_allow_overwrite(self):
        argv = runner.build_generate_argv("k", ("multi_turn_base",),
                                          allow_overwrite=True)
        self.assertIn("--allow-overwrite", argv)


class NormalizeTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.result_root = os.path.join(self.tmp.name, "result")
        self.score_root = os.path.join(self.tmp.name, "score")

    def _score_file(self, model_key, category, records):
        path = os.path.join(self.score_root, model_key, "multi_turn",
                            normalize.score_filename(category))
        _write_jsonl(path, records)  # score files are JSONL on disk
        return path

    def test_read_score_accuracy(self):
        path = self._score_file("k", "multi_turn_base",
                                _score_records(0.75, 3, 4))
        self.assertEqual(normalize.read_score_accuracy(path), (0.75, 3, 4))

    def test_read_score_accuracy_missing(self):
        self.assertIsNone(normalize.read_score_accuracy(
            os.path.join(self.tmp.name, "nope.json")))

    def test_count_turns_per_id(self):
        path = os.path.join(self.result_root, "k", "multi_turn",
                            normalize.result_filename("multi_turn_base"))
        # a: 2 turns of 3 and 5 steps; b: 1 turn of 4 steps.
        _write(path, _result_payload({"a": [3, 5], "b": [4]}))
        counts = normalize.count_turns_per_id(path)
        self.assertEqual(counts["a"]["turns"], 2)
        self.assertEqual(counts["a"]["steps_per_turn"], [3, 5])
        self.assertEqual(counts["a"]["max_steps"], 5)
        self.assertEqual(counts["b"]["max_steps"], 4)

    def test_normalize_unweighted_mean_and_missing(self):
        # Two of four categories scored; one complete, one partial.
        self._score_file("k", "multi_turn_base", _score_records(0.5, 5, 10))
        self._score_file("k", "multi_turn_miss_func", _score_records(1.0, 200, 200))
        score, per_cat, missing, partial = normalize.normalize(
            self.result_root, self.score_root, "k"
        )
        self.assertAlmostEqual(score, 0.75)  # unweighted mean of 0.5 and 1.0
        self.assertEqual(set(per_cat), {"multi_turn_base", "multi_turn_miss_func"})
        self.assertEqual(
            set(missing), {"multi_turn_miss_param", "multi_turn_long_context"}
        )
        # multi_turn_base scored 10/200 -> partial; miss_func 200/200 -> not.
        self.assertEqual(partial, ["multi_turn_base"])
        self.assertEqual(per_cat["multi_turn_base"]["expected_count"], 200)

    def test_normalize_none_when_nothing_scored(self):
        score, per_cat, missing, partial = normalize.normalize(
            self.result_root, self.score_root, "k"
        )
        self.assertIsNone(score)
        self.assertEqual(per_cat, {})
        self.assertEqual(len(missing), len(normalize.MULTI_TURN_CATEGORIES))
        self.assertEqual(partial, [])


class RunPipelineTest(unittest.TestCase):
    """End-to-end run() with bfcl_cli and serving stubbed out."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_root = os.path.join(self.tmp.name, "runs")
        self.results_dir = os.path.join(self.tmp.name, "results")

    def _fake_cli_factory(self, result_root, score_root):
        calls = {"generate": [], "evaluate": []}

        def fake_cli(argv, *, project_root):
            op = argv[0]
            calls[op].append(list(argv))
            if op == "evaluate":
                # Simulate bfcl evaluate writing all 4 category score files.
                # First category is partial (5 < 200 expected) to exercise the
                # M1 completeness gate; the rest are complete.
                counts = {"multi_turn_base": 5}
                for cat in normalize.MULTI_TURN_CATEGORIES:
                    total = counts.get(cat, 200)
                    path = os.path.join(score_root, "duoneural-v3-mlx-fc",
                                        "multi_turn",
                                        normalize.score_filename(cat))
                    _write_jsonl(path, _score_records(0.8, int(0.8 * total), total))
            if op == "generate":
                # Simulate bfcl generate writing one conversation's result.
                path = os.path.join(result_root, "duoneural-v3-mlx-fc",
                                    "multi_turn",
                                    normalize.result_filename("multi_turn_base"))
                _write(path, _result_payload({"multi_turn_base_0": [3, 3, 3]}))

        return fake_cli, calls

    def _run(self, fake_cli, *, resume=False):
        import types
        from unittest import mock
        fake_register = types.ModuleType("evals.components.bfcl.register")
        fake_register.register = lambda *a, **k: "duoneural-v3-mlx-fc"
        with mock.patch.dict(sys.modules,
                             {"evals.components.bfcl.register": fake_register}), \
             mock.patch.object(runner, "run_bfcl_cli", side_effect=fake_cli):
            return runner.run(
                "checkpoints/FakeCkpt",
                base_url="http://127.0.0.1:9/v1",
                model_name="fakerow",
                resume=resume,
                run_root=self.run_root,
                results_dir=self.results_dir,
            )

    def test_run_emits_normalized_result(self):
        result_root = os.path.join(self.run_root, "result")
        score_root = os.path.join(self.run_root, "score")
        fake_cli, calls = self._fake_cli_factory(result_root, score_root)

        result = self._run(fake_cli, resume=False)

        self.assertEqual(result["component"], "bfcl")
        self.assertEqual(result["model"], "fakerow")
        self.assertEqual(result["checkpoint"], "checkpoints/FakeCkpt")
        self.assertAlmostEqual(result["score"], 0.8)
        self.assertEqual(result["subscores"]["missing_categories"], [])
        # M1 gate: multi_turn_base scored 5/200 -> partial; run not complete.
        self.assertEqual(result["subscores"]["partial_categories"],
                         ["multi_turn_base"])
        self.assertFalse(result["subscores"]["complete"])
        self.assertFalse(result["subscores"]["thrash"]["flag"])
        self.assertEqual(
            result["subscores"]["thrash"]["step_counts"]
                  ["multi_turn_base_0"]["max_steps"], 3
        )
        # Result file written + schema-valid.
        out = os.path.join(self.results_dir, "bfcl-fakerow.json")
        self.assertTrue(os.path.exists(out))
        with open(out) as fh:
            on_disk = json.load(fh)
        from evals import results as results_mod
        self.assertEqual(results_mod.validate_result(on_disk), [])
        # generate ran (fresh: --allow-overwrite), evaluate ran once.
        self.assertEqual(len(calls["generate"]), 1)
        self.assertIn("--allow-overwrite", calls["generate"][0])
        self.assertEqual(len(calls["evaluate"]), 1)
        # Fresh run: --partial-eval omitted (checker enforces completeness).
        self.assertNotIn("--partial-eval", calls["evaluate"][0])

    def test_resume_omits_allow_overwrite(self):
        result_root = os.path.join(self.run_root, "result")
        score_root = os.path.join(self.run_root, "score")
        fake_cli, calls = self._fake_cli_factory(result_root, score_root)

        self._run(fake_cli, resume=True)

        self.assertEqual(len(calls["generate"]), 1)
        self.assertNotIn("--allow-overwrite", calls["generate"][0])
        # Resume: --partial-eval passed (result files may hold id-subsets).
        self.assertIn("--partial-eval", calls["evaluate"][0])


class RunBfclCliTest(unittest.TestCase):
    """run_bfcl_cli drives the bfcl typer CLI in-process (M2).

    bfcl_eval is stubbed so no real model/server is needed; this exercises the
    typer/click invocation, the BFCL_PROJECT_ROOT env, and SystemExit handling
    that RunPipelineTest stubs away.
    """

    def _patch_bfcl(self, recorded):
        import types
        from unittest import mock

        def fake_get_command(cli):
            def cmd(args=None, standalone_mode=True):
                recorded["args"] = list(args)
                recorded["standalone_mode"] = standalone_mode
                return 0
            return cmd

        fake_typer = types.ModuleType("typer")
        fake_typer.main = types.SimpleNamespace(get_command=fake_get_command)
        fake_bfcl_main = types.ModuleType("bfcl_eval.__main__")
        fake_bfcl_main.cli = object()
        return mock.patch.dict(
            sys.modules, {"typer": fake_typer, "bfcl_eval.__main__": fake_bfcl_main}
        )

    def test_invokes_cli_and_sets_project_root(self):
        recorded = {}
        with self._patch_bfcl(recorded):
            runner.run_bfcl_cli(["generate", "--model", "k"],
                                project_root="/tmp/bfclproj")
        self.assertEqual(recorded["args"], ["generate", "--model", "k"])
        self.assertFalse(recorded["standalone_mode"])
        self.assertEqual(os.environ.get("BFCL_PROJECT_ROOT"), "/tmp/bfclproj")

    def test_systemexit_zero_swallowed_nonzero_reraises(self):
        import types
        from unittest import mock

        def make_cmd(code):
            def cmd(args=None, standalone_mode=True):
                raise SystemExit(code)
            return cmd

        for code, expect_raise in ((0, False), (2, True)):
            fake_typer = types.ModuleType("typer")
            fake_typer.main = types.SimpleNamespace(
                get_command=lambda cli, c=code: make_cmd(c))
            fake_bfcl_main = types.ModuleType("bfcl_eval.__main__")
            fake_bfcl_main.cli = object()
            with mock.patch.dict(sys.modules,
                                 {"typer": fake_typer,
                                  "bfcl_eval.__main__": fake_bfcl_main}):
                if expect_raise:
                    with self.assertRaises(SystemExit):
                        runner.run_bfcl_cli(["evaluate"], project_root="/tmp/x")
                else:
                    self.assertEqual(
                        runner.run_bfcl_cli(["evaluate"], project_root="/tmp/x"), 0)


@unittest.skipUnless(
    os.environ.get("RUN_BFCL_LIVE") == "1",
    "live .venv-bfcl smoke test — set RUN_BFCL_LIVE=1 (needs bfcl_eval installed)",
)
class LiveBfclCliSmokeTest(unittest.TestCase):
    """M2 acceptance: actually invoke run_bfcl_cli against the real bfcl_eval.

    Skipped on the build host (no bfcl_eval); run inside .venv-bfcl on the MBP:
        RUN_BFCL_LIVE=1 .venv-bfcl/bin/python -m unittest \\
            tests.test_bfcl_adapter.LiveBfclCliSmokeTest -v
    """

    def test_run_bfcl_cli_noop_command(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            # `bfcl version` is a real no-op command: proves the typer/click
            # dispatch works against the installed bfcl_eval without a model.
            rc = runner.run_bfcl_cli(["version"], project_root=d)
            self.assertEqual(rc, 0)
            self.assertEqual(os.environ.get("BFCL_PROJECT_ROOT"), d)


if __name__ == "__main__":
    unittest.main()
