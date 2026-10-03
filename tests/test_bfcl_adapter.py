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


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _score_payload(accuracy, correct, total):
    # Mirrors save_eval_results: header dict first, then per-entry rows.
    return [{"accuracy": accuracy, "correct_count": correct, "total_count": total},
            {"id": "x", "valid": True}]


def _result_payload(turns_by_id):
    # One record per id, keyed by id, with a per-turn "result" list.
    return {
        tid: {"id": tid, "result": [[{"name": "f"}] for _ in range(n)]}
        for tid, n in turns_by_id.items()
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

    def test_generate_argv_run_ids_and_overwrite(self):
        argv = runner.build_generate_argv(
            "k", ("multi_turn_base",), run_ids=True, allow_overwrite=True,
            result_dir="/tmp/res",
        )
        self.assertIn("--run-ids", argv)
        self.assertIn("--allow-overwrite", argv)
        self.assertIn("/tmp/res", argv)

    def test_evaluate_argv(self):
        argv = runner.build_evaluate_argv(
            "k", ("multi_turn_base",), result_dir="/r", score_dir="/s"
        )
        self.assertEqual(argv[0], "evaluate")
        self.assertIn("--model", argv)
        self.assertIn("--score-dir", argv)
        self.assertIn("/s", argv)


class ThrashGuardTest(unittest.TestCase):
    def test_no_counts(self):
        t = runner.compute_thrash({})
        self.assertFalse(t["flag"])
        self.assertEqual(t["max_turns_observed"], 0)
        self.assertEqual(t["cap_hit_ids"], [])

    def test_below_cap_no_flag(self):
        t = runner.compute_thrash({"a": 3, "b": 5}, cap=20)
        self.assertFalse(t["flag"])
        self.assertEqual(t["max_turns_observed"], 5)
        self.assertEqual(t["cap_hit_ids"], [])

    def test_at_cap_flags(self):
        t = runner.compute_thrash({"ok": 4, "thrashy": 20}, cap=20)
        self.assertTrue(t["flag"])
        self.assertEqual(t["cap_hit_ids"], ["thrashy"])
        self.assertEqual(t["turn_counts"]["thrashy"], 20)


class ResumePlannerTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.result_root = os.path.join(self.tmp.name, "result")

    def _result_file(self, model_key, category, payload):
        path = os.path.join(self.result_root, model_key, "multi_turn",
                            normalize.result_filename(category))
        _write(path, payload)
        return path

    def test_full_regen_when_no_result_file(self):
        plan = runner.ids_to_run(self.result_root, "k", ("multi_turn_base",))
        self.assertEqual(plan["multi_turn_base"], ["__all__"])

    def test_skip_when_complete_no_id_list(self):
        self._result_file("k", "multi_turn_base",
                          _result_payload({"multi_turn_base_0": 4}))
        plan = runner.ids_to_run(self.result_root, "k", ("multi_turn_base",))
        self.assertEqual(plan["multi_turn_base"], [])  # done -> skip

    def test_partial_with_id_list(self):
        self._result_file("k", "multi_turn_base",
                          _result_payload({"multi_turn_base_0": 4}))
        requested = {"multi_turn_base": ["multi_turn_base_0", "multi_turn_base_1"]}
        plan = runner.ids_to_run(self.result_root, "k", ("multi_turn_base",),
                                 requested_ids=requested)
        # _0 done, _1 still to run
        self.assertEqual(plan["multi_turn_base"], ["multi_turn_base_1"])

    def test_ids_file_excludes_wildcards_and_empty(self):
        plan = {
            "multi_turn_base": ["multi_turn_base_1"],
            "multi_turn_miss_func": ["__all__"],
            "multi_turn_long_context": [],
        }
        path = os.path.join(self.tmp.name, "ids.json")
        runner.write_ids_file(path, plan)
        with open(path) as fh:
            payload = json.load(fh)
        self.assertEqual(payload, {"multi_turn_base": ["multi_turn_base_1"]})


class NormalizeTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.result_root = os.path.join(self.tmp.name, "result")
        self.score_root = os.path.join(self.tmp.name, "score")

    def _score_file(self, model_key, category, payload):
        path = os.path.join(self.score_root, model_key, "multi_turn",
                            normalize.score_filename(category))
        _write(path, payload)
        return path

    def test_read_score_accuracy(self):
        path = self._score_file("k", "multi_turn_base",
                                _score_payload(0.75, 3, 4))
        self.assertEqual(normalize.read_score_accuracy(path), (0.75, 3, 4))

    def test_read_score_accuracy_missing(self):
        self.assertIsNone(normalize.read_score_accuracy(
            os.path.join(self.tmp.name, "nope.json")))

    def test_count_turns_per_id(self):
        path = os.path.join(self.result_root, "k", "multi_turn",
                            normalize.result_filename("multi_turn_base"))
        _write(path, _result_payload({"a": 2, "b": 5}))
        self.assertEqual(normalize.count_turns_per_id(path), {"a": 2, "b": 5})

    def test_completed_ids_requires_nonempty_turns(self):
        path = os.path.join(self.result_root, "k", "multi_turn",
                            normalize.result_filename("multi_turn_base"))
        _write(path, {"done": {"id": "done", "result": [[{"name": "f"}]]},
                      "empty": {"id": "empty", "result": []}})
        self.assertEqual(normalize.completed_ids(path), {"done"})

    def test_normalize_unweighted_mean_and_missing(self):
        # Two of four categories scored.
        self._score_file("k", "multi_turn_base", _score_payload(0.5, 5, 10))
        self._score_file("k", "multi_turn_miss_func", _score_payload(1.0, 8, 8))
        score, per_cat, missing = normalize.normalize(
            self.result_root, self.score_root, "k"
        )
        self.assertAlmostEqual(score, 0.75)  # unweighted mean of 0.5 and 1.0
        self.assertEqual(set(per_cat), {"multi_turn_base", "multi_turn_miss_func"})
        self.assertEqual(
            set(missing), {"multi_turn_miss_param", "multi_turn_long_context"}
        )

    def test_normalize_none_when_nothing_scored(self):
        score, per_cat, missing = normalize.normalize(
            self.result_root, self.score_root, "k"
        )
        self.assertIsNone(score)
        self.assertEqual(per_cat, {})
        self.assertEqual(len(missing), len(normalize.MULTI_TURN_CATEGORIES))


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
                for cat in normalize.MULTI_TURN_CATEGORIES:
                    path = os.path.join(score_root, "duoneural-v3-mlx-fc",
                                        "multi_turn",
                                        normalize.score_filename(cat))
                    _write(path, _score_payload(0.8, 4, 5))
            if op == "generate":
                # Simulate bfcl generate writing one conversation's result.
                path = os.path.join(result_root, "duoneural-v3-mlx-fc",
                                    "multi_turn",
                                    normalize.result_filename("multi_turn_base"))
                _write(path, _result_payload({"multi_turn_base_0": 3}))

        return fake_cli, calls

    def test_run_emits_normalized_result(self):
        import types
        from unittest import mock

        result_root = os.path.join(self.run_root, "result")
        score_root = os.path.join(self.run_root, "score")
        fake_cli, calls = self._fake_cli_factory(result_root, score_root)

        # register is imported inside run() from .register; give it a stub
        # module so run() never imports bfcl_eval.
        fake_register = types.ModuleType("evals.components.bfcl.register")
        fake_register.register = lambda *a, **k: "duoneural-v3-mlx-fc"

        with mock.patch.dict(sys.modules,
                             {"evals.components.bfcl.register": fake_register}), \
             mock.patch.object(runner, "run_bfcl_cli", side_effect=fake_cli):
            result = runner.run(
                "checkpoints/FakeCkpt",
                base_url="http://127.0.0.1:9/v1",
                model_name="fakerow",
                run_root=self.run_root,
                results_dir=self.results_dir,
            )

        self.assertEqual(result["component"], "bfcl")
        self.assertEqual(result["model"], "fakerow")
        self.assertEqual(result["checkpoint"], "checkpoints/FakeCkpt")
        self.assertAlmostEqual(result["score"], 0.8)
        self.assertEqual(result["subscores"]["missing_categories"], [])
        self.assertFalse(result["subscores"]["thrash"]["flag"])
        self.assertEqual(
            result["subscores"]["thrash"]["turn_counts"].get("multi_turn_base_0"), 3
        )
        # Result file written + schema-valid.
        out = os.path.join(self.results_dir, "bfcl-fakerow.json")
        self.assertTrue(os.path.exists(out))
        with open(out) as fh:
            on_disk = json.load(fh)
        from evals import results as results_mod
        self.assertEqual(results_mod.validate_result(on_disk), [])
        # generate ran (no resume), evaluate ran once.
        self.assertEqual(len(calls["generate"]), 1)
        self.assertEqual(len(calls["evaluate"]), 1)


if __name__ == "__main__":
    unittest.main()
