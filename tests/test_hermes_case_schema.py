"""Schema + reference checks for the hand-authored Hermes FC cases.

Loads every JSON file in ``evals/hermes_fc/cases/`` and asserts:

- required top-level fields are present (id, category, prompt, tools, expected);
- every case carries a non-empty ``_doc`` one-liner (author-facing description);
- category is one of {single, parallel, negative, system2};
- every tool has a JSON-Schema object ``parameters`` block with ``type: object``;
- every ``expected.call.name`` is one of the case's tool names;
- every ``expected.call`` has an ``arguments`` dict;
- system2 cases have a non-empty ``system`` field and ``require_thought: true``;
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
TOOLS_JSON = os.path.join(_REPO_ROOT, "evals", "hermes_fc", "tools.json")
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

    def test_all_cases_have_doc(self):
        # N5 from the review of PR #34 (recurring from PR #31): ``_doc`` is
        # an author-facing one-line description of what a case tests. It was
        # present on some cases and missing on others, so enforce it on every
        # case (as a non-empty string) to keep the suite uniform.
        for fname, c in self.cases:
            doc = c.get("_doc")
            self.assertIsInstance(
                doc, str,
                f"{fname}: missing '_doc' (or not a string) — every case "
                f"needs a one-line description of what it tests",
            )
            self.assertTrue(
                doc.strip(),
                f"{fname}: '_doc' must be a non-empty string",
            )

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

    def test_system2_has_system_field(self):
        # Per the #4 amendment (2026-10-07): system2 cases carry a
        # 'think first' system message. The v3 4-bit model does not
        # emit `` reliably under the stock template; the system
        # field is what makes the gate clear. A system2 case without
        # a system field will score 0/10 thought 100% of the time.
        for fname, c in self.cases:
            if c["category"] == "system2":
                system = c.get("system")
                self.assertIsInstance(
                    system, str,
                    f"{fname}: system2 must have a 'system' field "
                    f"(the 'think first' nudge per the #4 amendment); "
                    f"got {type(system).__name__}",
                )
                self.assertTrue(
                    system.strip(),
                    f"{fname}: system2 'system' field must be a non-empty "
                    f"string — an empty system message does not clear the "
                    f"<thought> gate",
                )

    def test_system2_has_at_most_one_tool(self):
        # The v3 4-bit model runs out of output tokens (512 cap) when
        # exposed to 4 detailed tool schemas — observed in the
        # ``<thought>``-emission probe (see evals/hermes_fc/PROBE_RESULTS.md).
        # Limiting system2 cases to 1 tool keeps the thought + call
        # inside the token budget. The probe did not test 2-tool
        # cases, so we conservatively cap at 1 until we have data.
        for fname, c in self.cases:
            if c["category"] == "system2":
                self.assertLessEqual(
                    len(c["tools"]), 1,
                    f"{fname}: system2 cases must expose at most 1 tool "
                    f"(found {len(c['tools'])})",
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


class TestToolDefinitionsSSOT(unittest.TestCase):
    """Anti-drift guard for issue #33: tools.json is the single source of
    truth for tool definitions; every case embeds a snapshot of the
    canonical definitions it uses, and any divergence fails here.

    Three assertions:

    - tools.json internal consistency: every tool named in ``domains`` has
      a ``definitions`` entry and vice versa, and each definition's
      ``name`` field matches its key;
    - every embedded tool across all cases deep-equals the canonical
      ``definitions`` entry for its name (parsed-JSON equality — a case
      that rephrases a description, drops a property, or retypes a schema
      fails);
    - every embedded tool carries the OpenAI ``{"type": "function"}``
      envelope the adapter's chat-completions payload requires.

    This runs on the mini at ``python -m unittest`` time — a case author
    who edits one embedded copy (or adds a case with a hand-typed tool)
    gets a failure naming the case, the tool, and the differing keys,
    rather than silent drift across the suite.
    """

    @classmethod
    def setUpClass(cls):
        with open(TOOLS_JSON) as fh:
            cls.tools_doc = json.load(fh)
        cls.defs = cls.tools_doc["definitions"]
        cls.cases = _load_all()
        if not cls.cases:
            raise unittest.SkipTest(f"no cases in {CASES_DIR}")

    def test_tools_json_internally_consistent(self):
        domain_names = {
            name
            for domain in self.tools_doc["domains"].values()
            for name in domain["tools"]
        }
        def_names = set(self.defs)
        self.assertEqual(
            domain_names, def_names,
            f"domains/definitions mismatch: "
            f"only in domains: {sorted(domain_names - def_names)}, "
            f"only in definitions: {sorted(def_names - domain_names)}",
        )
        for key, defn in self.defs.items():
            self.assertEqual(
                defn["name"], key,
                f"definitions[{key!r}].name is {defn['name']!r}",
            )
            self.assertEqual(
                defn.get("parameters", {}).get("type"), "object",
                f"definitions[{key!r}].parameters.type must be 'object'",
            )

    def test_embedded_tools_match_canonical_definitions(self):
        for fname, c in self.cases:
            for i, t in enumerate(c["tools"]):
                name = t["function"]["name"]
                self.assertIn(
                    name, self.defs,
                    f"{fname}: tool[{i}] {name!r} has no canonical "
                    f"definition in tools.json",
                )
                canon = self.defs[name]
                embedded = t["function"]
                if embedded != canon:
                    # Name the differing top-level keys so the failure is
                    # actionable without a manual diff.
                    diffs = [
                        k for k in set(embedded) | set(canon)
                        if embedded.get(k) != canon.get(k)
                    ]
                    self.fail(
                        f"{fname}: embedded {name!r} diverges from "
                        f"tools.json (differs in {sorted(diffs)}) — edit "
                        f"tools.json and re-snapshot, or restore the case "
                        f"copy; drift is not allowed (issue #33)"
                    )

    def test_embedded_tools_carry_function_envelope(self):
        for fname, c in self.cases:
            for i, t in enumerate(c["tools"]):
                self.assertEqual(
                    t.get("type"), "function",
                    f"{fname}: tool[{i}] must be wrapped in "
                    f"{{'type': 'function', 'function': {{...}}}}",
                )


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
