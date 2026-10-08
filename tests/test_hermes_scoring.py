"""Unit tests for the Hermes FC one-set subset scorer (ticket #19).

Covers every clause of the scoring rule locked in #4 and extended in the
#19 contract:

* name-exact + recursive arg-equal (case-insensitive + trim on strings);
* strictness on optional args (Q7): predicted key sets must equal expected
  key sets — a predicted ``null`` the expected omits fails;
* ``command_equivalents`` (Q5/c1) widen string args only;
* parallel cases are order-insensitive unless ``depends_on_order: true``;
* ungrounded calls (name not in tools[]) and extra grounded calls are
  ignored (Q6);
* negative = no ``<tool_call>`` at all (a v3-dialect attempt also fails);
* system2 = ``<thought>`` present + stripped remainder scored;
* numeric coercion (1 == 1.0) and bool/number non-coercion;
* malformed ``<tool_call>`` JSON never matches but never crashes.

Stdlib ``unittest`` (also pytest-compatible); no third-party deps.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evals.components.hermes import scoring  # noqa: E402


def _completion(calls, *, thought=None, raw=None):
    """Build a model completion embedding #4-dialect tool calls."""
    if raw is not None:
        return raw
    body = ""
    if thought is not None:
        body += "<thought>{0}</thought>".format(thought)
    for c in calls:
        body += "<tool_call>\n{0}\n</tool_call>".format(json.dumps(c))
    return body


def _case(category, expected_calls, tools=None, **extra):
    tool_names = tools or ["read_file", "write_file", "copy_file",
                           "run_shell_command", "schedule_meeting"]
    case = {
        "id": "fc-{0}-999".format(category),
        "category": category,
        "prompt": "…",
        "tools": [{"type": "function", "function": {"name": n, "parameters": {
            "type": "object", "properties": {}}}} for n in tool_names],
        "expected": {"calls": expected_calls},
    }
    case.update(extra)
    return case


class TestParsePredictedCalls(unittest.TestCase):
    def test_single_call(self):
        content = _completion([{"name": "read_file", "arguments": {"path": "/etc/hosts"}}])
        calls = scoring.parse_predicted_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read_file")

    def test_multiple_calls(self):
        content = _completion([
            {"name": "copy_file", "arguments": {"src": "/a", "dst": "/b"}},
            {"name": "read_file", "arguments": {"path": "/b"}},
        ])
        calls = scoring.parse_predicted_calls(content)
        self.assertEqual(len(calls), 2)

    def test_malformed_json_kept_as_parse_error(self):
        content = "<tool_call>{not json</tool_call>"
        calls = scoring.parse_predicted_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["_parse_error"])

    def test_escaped_quotes_and_braces_in_strings(self):
        # Regression (caught in review): the extractor must not treat an
        # escaped quote as ending a string, nor a '}' inside a string as
        # closing the payload.
        import json as _json
        payload = _json.dumps({"name": "run_shell_command",
                               "arguments": {"command": 'echo "hi" && echo }'}})
        calls = scoring.parse_predicted_calls(
            "<tool_call>" + payload + "</tool_call>")
        self.assertEqual(calls[0]["arguments"]["command"], 'echo "hi" && echo }')

    def test_no_calls(self):
        self.assertEqual(scoring.parse_predicted_calls("just prose."), [])

    def test_whitespace_inside_tags(self):
        content = '<tool_call>  \n {"name": "read_file", "arguments": {}} \n</tool_call>'
        calls = scoring.parse_predicted_calls(content)
        self.assertEqual(calls[0]["name"], "read_file")


class TestArgEqual(unittest.TestCase):
    def test_string_case_insensitive_trim(self):
        self.assertTrue(scoring.arg_equal("/Etc/Hosts ", " /etc/hosts"))
        self.assertFalse(scoring.arg_equal("/etc/hosts", "/etc/hostname"))

    def test_command_equivalents(self):
        eqs = ["head -n 5 /var/log/syslog", "head -n5 /var/log/syslog"]
        self.assertTrue(scoring.arg_equal("head -n 5 /var/log/syslog",
                                          "HEAD -N5 /var/log/syslog",
                                          equivalents=eqs))
        self.assertFalse(scoring.arg_equal("head -n 5 /var/log/syslog",
                                           "tail -n 5 /var/log/syslog",
                                           equivalents=eqs))

    def test_equivalents_ignored_for_non_string_expected(self):
        self.assertFalse(scoring.arg_equal(30, "30", equivalents=["30"]))

    def test_numbers_coerce_int_float(self):
        self.assertTrue(scoring.arg_equal(30, 30.0))
        self.assertFalse(scoring.arg_equal(30, 31))

    def test_bool_never_equals_number(self):
        self.assertFalse(scoring.arg_equal(True, 1))
        self.assertFalse(scoring.arg_equal(1, True))
        self.assertTrue(scoring.arg_equal(True, True))

    def test_type_mismatch_fails(self):
        self.assertFalse(scoring.arg_equal("30", 30))
        self.assertFalse(scoring.arg_equal([1], 1))

    def test_list_elementwise(self):
        self.assertTrue(scoring.arg_equal(["a", "b"], [" A", "b "]))
        self.assertFalse(scoring.arg_equal(["a", "b"], ["b", "a"]))
        self.assertFalse(scoring.arg_equal(["a"], ["a", "b"]))

    def test_dict_recursive(self):
        self.assertTrue(scoring.arg_equal({"a": {"b": "C"}}, {"a": {"b": "c"}}))
        self.assertFalse(scoring.arg_equal({"a": {"b": 1}}, {"a": {"b": 2}}))

    def test_none(self):
        self.assertTrue(scoring.arg_equal(None, None))
        self.assertFalse(scoring.arg_equal(None, "x"))


class TestCallMatches(unittest.TestCase):
    def test_name_must_match_exactly(self):
        exp = {"name": "read_file", "arguments": {}}
        self.assertFalse(scoring.call_matches(exp, {"name": "Read_File", "arguments": {}}))
        self.assertTrue(scoring.call_matches(exp, {"name": "read_file", "arguments": {}}))

    def test_strictness_extra_predicted_key_fails(self):
        # Q7: predicted omits nothing but adds a key the expected omits.
        exp = {"name": "http_get", "arguments": {"url": "https://x"}}
        pred = {"name": "http_get", "arguments": {"url": "https://x", "headers": None}}
        self.assertFalse(scoring.call_matches(exp, pred))

    def test_strictness_missing_predicted_key_fails(self):
        exp = {"name": "http_get", "arguments": {"url": "https://x", "headers": None}}
        pred = {"name": "http_get", "arguments": {"url": "https://x"}}
        self.assertFalse(scoring.call_matches(exp, pred))

    def test_parse_error_never_matches(self):
        exp = {"name": "read_file", "arguments": {}}
        self.assertFalse(scoring.call_matches(exp, {"_parse_error": True, "raw": "…"}))

    def test_command_equivalents_applied_per_string_arg(self):
        exp = {"name": "run_shell_command",
               "arguments": {"command": "head -n 5 /var/log/syslog"},
               "command_equivalents": ["head -n 5 /var/log/syslog",
                                       "head -n5 /var/log/syslog"]}
        pred = {"name": "run_shell_command",
                "arguments": {"command": "head -n5 /var/log/syslog"}}
        self.assertTrue(scoring.call_matches(exp, pred))


class TestMatchSets(unittest.TestCase):
    def test_unordered_ignores_order(self):
        expected = [{"name": "copy_file", "arguments": {"src": "/a", "dst": "/b"}},
                    {"name": "read_file", "arguments": {"path": "/b"}}]
        predicted = [{"name": "read_file", "arguments": {"path": "/b"}},
                     {"name": "copy_file", "arguments": {"src": "/a", "dst": "/b"}}]
        ok, unmatched = scoring.match_calls_unordered(expected, predicted)
        self.assertTrue(ok)
        self.assertEqual(unmatched, [])

    def test_unordered_extra_predicted_ignored(self):
        expected = [{"name": "read_file", "arguments": {"path": "/a"}}]
        predicted = [{"name": "write_file", "arguments": {"path": "/z", "contents": ""}},
                     {"name": "read_file", "arguments": {"path": "/a"}}]
        ok, _ = scoring.match_calls_unordered(expected, predicted)
        self.assertTrue(ok)

    def test_unordered_one_predicted_cannot_match_twice(self):
        expected = [{"name": "read_file", "arguments": {"path": "/a"}},
                    {"name": "read_file", "arguments": {"path": "/a"}}]
        predicted = [{"name": "read_file", "arguments": {"path": "/a"}}]
        ok, unmatched = scoring.match_calls_unordered(expected, predicted)
        self.assertFalse(ok)
        self.assertEqual(len(unmatched), 1)

    def test_ordered_subsequence_passes(self):
        expected = [{"name": "copy_file", "arguments": {"src": "/a", "dst": "/b"}},
                    {"name": "read_file", "arguments": {"path": "/b"}}]
        predicted = [{"name": "copy_file", "arguments": {"src": "/a", "dst": "/b"}},
                     {"name": "write_file", "arguments": {"path": "/x", "contents": ""}},
                     {"name": "read_file", "arguments": {"path": "/b"}}]
        ok, _ = scoring.match_calls_ordered(expected, predicted)
        self.assertTrue(ok)

    def test_ordered_wrong_order_fails(self):
        expected = [{"name": "copy_file", "arguments": {"src": "/a", "dst": "/b"}},
                    {"name": "read_file", "arguments": {"path": "/b"}}]
        predicted = [{"name": "read_file", "arguments": {"path": "/b"}},
                     {"name": "copy_file", "arguments": {"src": "/a", "dst": "/b"}}]
        ok, unmatched = scoring.match_calls_ordered(expected, predicted)
        self.assertFalse(ok)
        self.assertEqual(len(unmatched), 1)


class TestScoreCase(unittest.TestCase):
    def test_single_pass(self):
        case = _case("single", [{"name": "read_file",
                                 "arguments": {"path": "/etc/hosts"}}])
        content = _completion([{"name": "read_file",
                                "arguments": {"path": " /ETC/HOSTS"}}])
        d = scoring.score_case(case, content)
        self.assertTrue(d["passed"])
        self.assertEqual(d["reason"], "match")

    def test_single_fail_wrong_args(self):
        case = _case("single", [{"name": "read_file",
                                 "arguments": {"path": "/etc/hosts"}}])
        content = _completion([{"name": "read_file",
                                "arguments": {"path": "/etc/hostname"}}])
        d = scoring.score_case(case, content)
        self.assertFalse(d["passed"])
        self.assertEqual(d["reason"], "arguments_mismatch")
        self.assertEqual(len(d["unmatched_expected"]), 1)

    def test_ungrounded_call_ignored(self):
        # Q6: a predicted call whose name is not in tools[] does not fail
        # the case; the grounded expected call still decides.
        case = _case("single", [{"name": "read_file",
                                 "arguments": {"path": "/etc/hosts"}}])
        content = _completion([
            {"name": "delete_everything", "arguments": {"path": "/"}},
            {"name": "read_file", "arguments": {"path": "/etc/hosts"}},
        ])
        d = scoring.score_case(case, content)
        self.assertTrue(d["passed"])
        self.assertEqual(d["ungrounded_names"], ["delete_everything"])

    def test_ungrounded_only_fails(self):
        case = _case("single", [{"name": "read_file",
                                 "arguments": {"path": "/etc/hosts"}}])
        content = _completion([{"name": "delete_everything", "arguments": {}}])
        d = scoring.score_case(case, content)
        self.assertFalse(d["passed"])
        self.assertEqual(d["reason"], "no_grounded_call")

    def test_negative_pass_no_call(self):
        case = _case("negative", [])
        d = scoring.score_case(case, "Hello! I'm doing well, thanks for asking.")
        self.assertTrue(d["passed"])
        self.assertEqual(d["reason"], "no_call_as_expected")

    def test_negative_fail_on_d4_call(self):
        case = _case("negative", [])
        content = _completion([{"name": "read_file", "arguments": {"path": "/etc/hosts"}}])
        d = scoring.score_case(case, content)
        self.assertFalse(d["passed"])
        self.assertEqual(d["reason"], "unexpected_call_attempt")

    def test_negative_fail_on_v3_attempt(self):
        # A v3-dialect opening tag is an *attempt* — fails a negative case
        # even though v1.0 never parses it.
        case = _case("negative", [])
        d = scoring.score_case(case, '<tool_call name="read_file" arguments=\'{}\'>')
        self.assertFalse(d["passed"])

    def test_negative_fail_on_unclosed_attempt(self):
        case = _case("negative", [])
        d = scoring.score_case(case, 'Let me check: <tool_call name="read_file"')
        self.assertFalse(d["passed"])

    def test_system2_pass_with_thought(self):
        case = _case("system2",
                     [{"name": "schedule_meeting",
                       "arguments": {"title": "meeting with Bob",
                                     "duration_minutes": 30}}],
                     tools=["schedule_meeting"],
                     require_thought=True,
                     system="think first")
        content = _completion(
            [{"name": "schedule_meeting",
              "arguments": {"title": "Meeting with Bob", "duration_minutes": 30}}],
            thought="The user wants…")
        d = scoring.score_case(case, content)
        self.assertTrue(d["passed"])
        self.assertTrue(d["thought_present"])

    def test_system2_fail_missing_thought(self):
        case = _case("system2",
                     [{"name": "schedule_meeting",
                       "arguments": {"title": "meeting with Bob",
                                     "duration_minutes": 30}}],
                     tools=["schedule_meeting"],
                     require_thought=True)
        content = _completion([{"name": "schedule_meeting",
                                "arguments": {"title": "meeting with Bob",
                                              "duration_minutes": 30}}])
        d = scoring.score_case(case, content)
        self.assertFalse(d["passed"])
        self.assertEqual(d["reason"], "missing_thought")
        self.assertFalse(d["thought_present"])

    def test_parallel_order_insensitive_default(self):
        case = _case("parallel",
                     [{"name": "copy_file", "arguments": {"src": "/a", "dst": "/b"}},
                      {"name": "read_file", "arguments": {"path": "/b"}}])
        content = _completion([
            {"name": "read_file", "arguments": {"path": "/b"}},
            {"name": "copy_file", "arguments": {"src": "/a", "dst": "/b"}},
        ])
        d = scoring.score_case(case, content)
        self.assertTrue(d["passed"])

    def test_parallel_depends_on_order(self):
        case = _case("parallel",
                     [{"name": "copy_file", "arguments": {"src": "/a", "dst": "/b"}},
                      {"name": "read_file", "arguments": {"path": "/b"}}],
                     depends_on_order=True)
        swapped = _completion([
            {"name": "read_file", "arguments": {"path": "/b"}},
            {"name": "copy_file", "arguments": {"src": "/a", "dst": "/b"}},
        ])
        self.assertFalse(scoring.score_case(case, swapped)["passed"])
        in_order = _completion([
            {"name": "copy_file", "arguments": {"src": "/a", "dst": "/b"}},
            {"name": "read_file", "arguments": {"path": "/b"}},
        ])
        self.assertTrue(scoring.score_case(case, in_order)["passed"])

    def test_malformed_call_is_a_failed_case_not_a_crash(self):
        case = _case("single", [{"name": "read_file",
                                 "arguments": {"path": "/etc/hosts"}}])
        d = scoring.score_case(case, "<tool_call>{not json</tool_call>")
        self.assertFalse(d["passed"])
        self.assertEqual(d["reason"], "no_parseable_call")
        self.assertEqual(d["parse_errors"], 1)

    def test_empty_completion(self):
        case = _case("single", [{"name": "read_file",
                                 "arguments": {"path": "/etc/hosts"}}])
        d = scoring.score_case(case, "")
        self.assertFalse(d["passed"])
        self.assertEqual(d["reason"], "no_parseable_call")


class TestThoughtExtraction(unittest.TestCase):
    def test_extract_and_strip(self):
        thought, rem = scoring.extract_thought("<thought>abc</thought>rest")
        self.assertEqual(thought, "abc")
        self.assertEqual(rem, "rest")

    def test_no_thought(self):
        thought, rem = scoring.extract_thought("plain")
        self.assertIsNone(thought)
        self.assertEqual(rem, "plain")


if __name__ == "__main__":
    unittest.main()
