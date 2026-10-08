"""Schema + reference checks for the hand-authored Hermes FC cases.

Loads every JSON file in ``evals/hermes_fc/cases/`` and asserts:

- required top-level fields are present (id, category, prompt, tools, expected);
- category is one of {single, parallel, negative, system2};
- every tool has a JSON-Schema object ``parameters`` block with ``type: object``;
- every ``expected.call.name`` is one of the case's tool names;
- every ``expected.call`` has an ``arguments`` dict;
- system2 cases have ``require_thought: true``;
- negative cases have an empty ``expected.calls`` list;
- single + parallel + system2 cases have at least one ``expected.call``;
- ids are unique across the directory;
- ids match the convention ``fc-{category}-NNN`` (zero-padded 3 digits).

Stdlib ``unittest`` (also pytest-compatible); needs no third-party deps.
Catches authoring drift: any new case file that misses a field or breaks
the id convention is caught at ``python -m unittest`` time on the mini.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES_DIR = os.path.join(_REPO_ROOT, "evals", "hermes_fc", "cases")
CATEGORIES = {"single", "parallel", "negative", "system2"}
ID_RE = re.compile(r"^fc-(single|parallel|negative|system2)-\d{3}$")


def _load_all():
    files = sorted(glob.glob(os.path.join(CASES_DIR, "*.json")))
    cases = []
    for f in files:
        with open(f) as fh:
            cases.append((os.path.basename(f), json.load(fh)))
    return cases


class TestCaseSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _load_all()
        if not cls.cases:
            raise unittest.SkipTest(f"no cases in {CASES_DIR}")

    def test_at_least_one_case(self):
        self.assertGreater(len(self.cases), 0, "no case files found")

    def test_ids_unique(self):
        ids = [c["id"] for _, c in self.cases]
        self.assertEqual(len(ids), len(set(ids)), f"duplicate ids: {ids}")

    def test_id_format(self):
        for _, c in self.cases:
            self.assertRegex(c["id"], ID_RE, f"bad id format: {c['id']!r}")

    def test_required_fields(self):
        required = {"id", "category", "prompt", "tools", "expected"}
        for fname, c in self.cases:
            missing = required - set(c.keys())
            self.assertFalse(missing, f"{fname}: missing {sorted(missing)}")

    def test_category_valid(self):
        for fname, c in self.cases:
            self.assertIn(
                c["category"], CATEGORIES,
                f"{fname}: bad category {c['category']!r}",
            )

    def test_id_matches_category(self):
        for _, c in self.cases:
            self.assertIn(c["category"], c["id"], f"id {c['id']!r} does not contain category {c['category']!r}")

    def test_tools_have_object_parameters(self):
        for fname, c in self.cases:
            self.assertIsInstance(c["tools"], list)
            self.assertGreater(len(c["tools"]), 0, f"{fname}: tools is empty")
            for i, t in enumerate(c["tools"]):
                self.assertIn("function", t, f"{fname}: tool[{i}] missing 'function'")
                f_ = t["function"]
                self.assertIn("name", f_, f"{fname}: tool[{i}] missing name")
                params = f_.get("parameters", {})
                self.assertEqual(
                    params.get("type"), "object",
                    f"{fname}: tool {f_['name']!r} parameters.type must be 'object'",
                )

    def test_expected_calls_reference_real_tools(self):
        for fname, c in self.cases:
            names = {t["function"]["name"] for t in c["tools"]}
            for i, call in enumerate(c["expected"]["calls"]):
                self.assertIn("name", call, f"{fname}: call[{i}] missing name")
                self.assertIn(call["name"], names, f"{fname}: call[{i}] name {call['name']!r} not in tools {sorted(names)}")
                self.assertIn("arguments", call, f"{fname}: call[{i}] missing arguments")

    def test_negative_has_no_expected_calls(self):
        for fname, c in self.cases:
            if c["category"] == "negative":
                self.assertEqual(
                    c["expected"]["calls"], [],
                    f"{fname}: negative category must have expected.calls=[]",
                )

    def test_non_negative_has_at_least_one_call(self):
        for fname, c in self.cases:
            if c["category"] != "negative":
                self.assertGreater(
                    len(c["expected"]["calls"]), 0,
                    f"{fname}: non-negative category must have at least one expected.call",
                )

    def test_system2_requires_thought(self):
        for fname, c in self.cases:
            if c["category"] == "system2":
                self.assertIs(
                    c.get("require_thought"), True,
                    f"{fname}: system2 must have require_thought: true",
                )

    def test_depends_on_order_only_on_parallel(self):
        # Per #4: ``depends_on_order`` is a parallel-category-only knob.
        # The ``True`` value flips the scoring rule to "calls must appear
        # in expected order"; it's meaningless on single/negative/system2
        # because those categories already have at most one call (single,
        # system2) or zero (negative). A stray ``true`` on another
        # category is almost certainly an authoring bug.
        #
        # Two assertions:
        #   (a) if the value is truthy, it must be the literal ``True``
        #       (a string like ``"yes"`` or an int like ``1`` is a wrong-
        #       typed value, not a valid flag);
        #   (b) if the value is ``True``, the case must be ``parallel``.
        for fname, c in self.cases:
            val = c.get("depends_on_order")
            if val is None or val is False:
                continue
            if val is not True:
                self.fail(
                    f"{fname}: depends_on_order must be the literal true "
                    f"or absent, got {val!r} (type {type(val).__name__})"
                )
            if c["category"] != "parallel":
                self.fail(
                    f"{fname}: depends_on_order: true is only valid on "
                    f"parallel cases (this case is {c['category']!r})"
                )

    def test_command_equivalents_only_on_string_args(self):
        # Q5/c1 extension from the review of PR #25: command_equivalents is
        # an optional list of strings per expected call, and only applies
        # to string-typed args in the call's arguments. We don't validate
        # the tool-side schema here (that's the adapter's job) but we do
        # require the value to be a list of strings if present.
        for fname, c in self.cases:
            for i, call in enumerate(c["expected"]["calls"]):
                eqs = call.get("command_equivalents")
                if eqs is None:
                    continue
                self.assertIsInstance(eqs, list, f"{fname}: call[{i}].command_equivalents must be a list")
                for j, eq in enumerate(eqs):
                    self.assertIsInstance(eq, str, f"{fname}: call[{i}].command_equivalents[{j}] must be a string")


class TestCategoryMix(unittest.TestCase):
    """The mix in #4 is 20 single / 10 parallel / 5 negative / 5 system2.
    As cases land, this test asserts the current count vs. the target.
    Target = the contracted ceiling; we only fail if we *exceed* it.
    """

    TARGETS = {"single": 20, "parallel": 10, "negative": 5, "system2": 5}

    def test_mix_within_ceiling(self):
        cases = _load_all()
        counts = {cat: 0 for cat in self.TARGETS}
        for _, c in cases:
            counts[c["category"]] = counts.get(c["category"], 0) + 1
        for cat, target in self.TARGETS.items():
            self.assertLessEqual(
                counts[cat], target,
                f"{cat}: {counts[cat]} cases exceeds contracted ceiling of {target}",
            )


if __name__ == "__main__":
    unittest.main()
