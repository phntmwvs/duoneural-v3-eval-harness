"""Unit tests for the Hermes FC adapter runner (ticket #19).

The ``post_chat`` seam is patched throughout — these tests never touch a
live ``mlx_lm server``. Covered:

* ``load_cases`` — loads the real 40-case suite in stable id order;
* ``build_payload`` — prompt/system/tools wiring, greedy decoding, the
  512-token cap, ``default_model`` key, ``tool_choice: auto`` (Q1);
* ``extract_content`` — OpenAI response shape + malformed-payload error;
* ``run_case`` — scored pass/fail, and request errors recorded (not raised);
* ``run`` — full-suite mock run: every case fed a known-good completion
  passes (the ticket's smoke test), normalized result JSON validates and
  is written; per-category subscores; request errors surface in
  ``n_errors`` and depress the score;
* resume semantics (ticket #6, decision 7a) — a resumed run skips cases
  already in ``state.json``; a fresh run clears it.

Stdlib ``unittest`` (also pytest-compatible); no third-party deps.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evals import results as results_mod  # noqa: E402
from evals.components.hermes import runner, scoring  # noqa: E402


def _chat_response(content):
    """Minimal OpenAI chat-completions response payload."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _completion_for_case(case):
    """A known-good completion for ``case``: exactly the expected calls in
    the #4 dialect (plus a ``<thought>`` when the case requires one)."""
    calls = (case.get("expected") or {}).get("calls") or []
    body = ""
    if case.get("require_thought"):
        body += "<thought>reasoning about the task</thought>"
    for c in calls:
        body += "<tool_call>\n{0}\n</tool_call>".format(json.dumps(
            {"name": c["name"], "arguments": c.get("arguments") or {}}))
    return body or "Certainly — happy to help!"


def _mock_post_chat(payloads):
    """``post_chat`` replacement returning ``_chat_response`` per call."""
    responses = [_chat_response(c) for c in payloads]
    return mock.patch.object(runner, "post_chat", side_effect=responses)


class TestLoadCases(unittest.TestCase):
    def test_loads_real_suite(self):
        cases = runner.load_cases()
        self.assertEqual(len(cases), 40)
        ids = [c["id"] for c in cases]
        self.assertEqual(ids, sorted(ids), "cases must load in stable id order")
        counts = {}
        for c in cases:
            counts[c["category"]] = counts.get(c["category"], 0) + 1
        self.assertEqual(counts, {"single": 20, "parallel": 10,
                                  "negative": 5, "system2": 5})

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                runner.load_cases(d)


class TestBuildPayload(unittest.TestCase):
    def test_prompt_and_tools(self):
        case = {"prompt": "read /etc/hosts", "tools": [{"type": "function",
                "function": {"name": "read_file"}}]}
        p = runner.build_payload(case)
        self.assertEqual(p["model"], runner.DEFAULT_MODEL_KEY)
        self.assertEqual(p["messages"], [{"role": "user", "content": "read /etc/hosts"}])
        self.assertEqual(p["tools"], case["tools"])
        self.assertEqual(p["tool_choice"], "auto")
        self.assertEqual(p["temperature"], 0.0)
        self.assertEqual(p["n"], 1)
        self.assertFalse(p["stream"])
        self.assertEqual(p["max_tokens"], 512)

    def test_system_override_included(self):
        case = {"prompt": "…", "system": "think first", "tools": []}
        p = runner.build_payload(case)
        self.assertEqual(p["messages"][0], {"role": "system", "content": "think first"})
        self.assertEqual(p["messages"][1]["role"], "user")

    def test_missing_tools_becomes_empty_list(self):
        p = runner.build_payload({"prompt": "…"})
        self.assertEqual(p["tools"], [])


class TestExtractContent(unittest.TestCase):
    def test_happy_path(self):
        self.assertEqual(runner.extract_content(_chat_response("hi")), "hi")

    def test_none_content_becomes_empty(self):
        self.assertEqual(runner.extract_content(_chat_response(None)), "")

    def test_malformed_raises(self):
        with self.assertRaises(ValueError):
            runner.extract_content({"choices": []})
        with self.assertRaises(ValueError):
            runner.extract_content({})


class TestRunCase(unittest.TestCase):
    def _case(self):
        return {
            "id": "fc-single-999", "category": "single", "prompt": "…",
            "tools": [{"type": "function", "function": {"name": "read_file",
                       "parameters": {"type": "object", "properties": {}}}}],
            "expected": {"calls": [{"name": "read_file",
                                    "arguments": {"path": "/etc/hosts"}}]},
        }

    def test_pass(self):
        with _mock_post_chat([_completion_for_case(self._case())]):
            d = runner.run_case(self._case(), base_url="http://127.0.0.1:9/v1")
        self.assertTrue(d["passed"])
        self.assertEqual(d["case_id"], "fc-single-999")
        self.assertIn("content", d)

    def test_fail(self):
        bad = _chat_response("<tool_call>\n" + json.dumps(
            {"name": "read_file", "arguments": {"path": "/wrong"}}) + "\n</tool_call>")
        with mock.patch.object(runner, "post_chat", return_value=bad):
            d = runner.run_case(self._case(), base_url="http://127.0.0.1:9/v1")
        self.assertFalse(d["passed"])

    def test_request_error_recorded_not_raised(self):
        with mock.patch.object(runner, "post_chat",
                               side_effect=TimeoutError("timed out")):
            d = runner.run_case(self._case(), base_url="http://127.0.0.1:9/v1")
        self.assertFalse(d["passed"])
        self.assertEqual(d["reason"], "request_error")
        self.assertIn("TimeoutError", d["error"])

    def test_malformed_response_recorded_not_raised(self):
        with mock.patch.object(runner, "post_chat", return_value={"nope": 1}):
            d = runner.run_case(self._case(), base_url="http://127.0.0.1:9/v1")
        self.assertFalse(d["passed"])
        self.assertEqual(d["reason"], "request_error")


class _RunTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes-run-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.run_root = os.path.join(self.tmp, "run")
        self.results_dir = os.path.join(self.tmp, "results")


class TestRunFullSuite(_RunTestBase):
    def test_smoke_all_cases_pass_on_known_good_completions(self):
        """The ticket's smoke test: feed every case a known-good
        ``<tool_call>`` mock — all 40 must pass (5 calibration singles
        included)."""
        cases = runner.load_cases()
        payloads = [_completion_for_case(c) for c in cases]
        with _mock_post_chat(payloads) as m:
            result = runner.run("checkpoints/DuoNeural-v3-4bit",
                                base_url="http://127.0.0.1:9/v1",
                                model_name="4bit",
                                run_root=self.run_root,
                                results_dir=self.results_dir)
        self.assertEqual(m.call_count, 40)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["subscores"]["n_cases"], 40)
        self.assertEqual(result["subscores"]["n_passed"], 40)
        self.assertEqual(result["subscores"]["n_errors"], 0)
        per_cat = result["subscores"]["per_category"]
        self.assertEqual(per_cat["single"], {"n": 20, "passed": 20})
        self.assertEqual(per_cat["parallel"], {"n": 10, "passed": 10})
        self.assertEqual(per_cat["negative"], {"n": 5, "passed": 5})
        self.assertEqual(per_cat["system2"], {"n": 5, "passed": 5})
        # Normalized schema validates + result JSON lands on disk.
        self.assertEqual(results_mod.validate_result(result), [])
        path = results_mod.result_path(self.results_dir, "hermes", "4bit")
        self.assertTrue(os.path.exists(path))
        with open(path) as fh:
            on_disk = json.load(fh)
        self.assertEqual(on_disk["score"], 1.0)
        # State file recorded every case for a later --resume.
        state = runner.load_state(self.run_root)
        self.assertEqual(len(state["cases"]), 40)

    def test_request_errors_depress_score(self):
        cases = runner.load_cases()
        by_prompt = {c["prompt"]: _completion_for_case(c) for c in cases}
        negative_prompts = {c["prompt"] for c in cases if c["category"] == "negative"}

        def flaky(base_url, payload, timeout_s=0):
            prompt = payload["messages"][-1]["content"]
            if prompt in negative_prompts:
                raise TimeoutError("boom")
            return _chat_response(by_prompt[prompt])

        with mock.patch.object(runner, "post_chat", side_effect=flaky):
            result = runner.run("ckpt", base_url="http://127.0.0.1:9/v1",
                                model_name="m", run_root=self.run_root,
                                results_dir=self.results_dir)
        self.assertEqual(result["subscores"]["n_errors"], 5)
        self.assertEqual(result["subscores"]["n_passed"], 35)
        self.assertAlmostEqual(result["score"], 35 / 40)
        self.assertEqual(result["subscores"]["per_category"]["negative"],
                         {"n": 5, "passed": 0})

    def test_all_fail_scores_zero_but_valid(self):
        with mock.patch.object(runner, "post_chat",
                               side_effect=TimeoutError("down")):
            result = runner.run("ckpt", base_url="http://127.0.0.1:9/v1",
                                model_name="m", run_root=self.run_root,
                                results_dir=self.results_dir)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["subscores"]["n_errors"], 40)
        self.assertEqual(results_mod.validate_result(result), [])


class TestResume(_RunTestBase):
    def test_resume_skips_completed_cases(self):
        cases = runner.load_cases()
        first = [_completion_for_case(c) for c in cases[:10]]
        with _mock_post_chat(first) as m1:
            with mock.patch.object(runner, "load_cases",
                                   return_value=cases[:10]):
                runner.run("ckpt", base_url="http://x/v1", model_name="m",
                           run_root=self.run_root, results_dir=self.results_dir)
        self.assertEqual(m1.call_count, 10)

        rest = [_completion_for_case(c) for c in cases[10:]]
        with _mock_post_chat(rest) as m2:
            result = runner.run("ckpt", base_url="http://x/v1", model_name="m",
                                resume=True, run_root=self.run_root,
                                results_dir=self.results_dir)
        self.assertEqual(m2.call_count, 30)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["subscores"]["n_cases"], 40)
        self.assertEqual(result["subscores"]["cases_ran_this_run"], 30)
        self.assertTrue(result["subscores"]["resume"])

    def test_fresh_run_clears_state(self):
        cases = runner.load_cases()
        with mock.patch.object(runner, "post_chat",
                               side_effect=TimeoutError("down")):
            runner.run("ckpt", base_url="http://x/v1", model_name="m",
                       run_root=self.run_root, results_dir=self.results_dir)
        state = runner.load_state(self.run_root)
        self.assertTrue(all(not d["passed"] for d in state["cases"].values()))

        payloads = [_completion_for_case(c) for c in cases]
        with _mock_post_chat(payloads) as m:
            result = runner.run("ckpt", base_url="http://x/v1", model_name="m",
                                run_root=self.run_root,
                                results_dir=self.results_dir)
        self.assertEqual(m.call_count, 40, "fresh run must rerun every case")
        self.assertEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
