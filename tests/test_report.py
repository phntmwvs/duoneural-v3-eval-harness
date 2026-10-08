"""Tests for ``evals/report.py`` and the ``report.py`` CLI (ticket #7).

Drives the pure builders (``build_ab_table`` / ``build_crosscheck_table`` /
``load_anchors``) directly with table-driven fixtures, plus the CLI render
loop on a tmp ``--summary-json`` file. No MLX, no subprocesses — the report
is a stdlib-only pure-function module that reads the matrix's summary JSON
and the per-cell result files (for ``subscores`` only).

Stdlib ``unittest`` (also pytest-compatible).
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals import report as report_mod  # noqa: E402
from evals import results as results_mod  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ANCHORS_PATH = os.path.join(_REPO_ROOT, "evals", "report_anchors.yaml")


# ---------------------------------------------------------------------------
# Fixtures: a complete 4×3 summary, a partial one, and an all-missing one.
# Scores are chosen so the verdict bands (1.0 / 3.0 pp) are exercised.
# ---------------------------------------------------------------------------


def _make_result(component: str, model: str, *, score: float, subscores=None,
                 runtime_s: float = 1.0):
    """Build a schema-valid result dict via the canonical helper."""
    return results_mod.new_result(
        component=component,
        model=model,
        checkpoint="ckpt-{0}".format(model),
        score=score,
        subscores=subscores,
        runtime_s=runtime_s,
        artifact_versions={"component": component},
    )


def _write_result(results_dir: str, component: str, model: str, result: dict):
    path = results_mod.result_path(results_dir, component, model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def _full_summary():
    """A clean 4×3 summary where every cell has a score; BF16 is the baseline."""
    # Scores chosen so:
    #   8-bit: BFCL 0.5pp below BF16 -> verified; evalplus 1.5pp below -> approximate; hermes 3.5pp below -> disputed.
    #   4-bit: BFCL 0.0pp -> verified; evalplus 0.0pp -> verified; hermes 0.4pp -> verified.
    #   base:  BFCL 13.0pp below BF16 -> verified (delta is large but vendor is *different* model);
    #         evalplus 18.0pp below BF16 -> verified;
    #         hermes 50.0pp below BF16 -> verified (vendor says base Hermes AST is 49.7%).
    cells = []
    scores = {
        "bf16":  {"bfcl": 62.10, "evalplus": 56.10, "hermes": 75.00},
        "8bit":  {"bfcl": 61.60, "evalplus": 54.60, "hermes": 71.50},
        "4bit":  {"bfcl": 62.10, "evalplus": 56.10, "hermes": 74.60},
        "base":  {"bfcl": 49.10, "evalplus": 38.10, "hermes": 25.00},
    }
    for row, comp_scores in scores.items():
        for component, score in comp_scores.items():
            cells.append({
                "row": row, "component": component, "status": "ok",
                "score": score, "path": "evals/results/{0}-{1}.json".format(component, row),
            })
    return {"ran": 12, "skipped": 0, "failed": 0, "elapsed_s": 12.0, "cells": cells}


def _partial_summary():
    """Three cells missing (the BF16 hermes + two base cells)."""
    cells = []
    scores = {
        "bf16":  {"bfcl": 62.10, "evalplus": 56.10, "hermes": None},  # failed
        "8bit":  {"bfcl": 61.60, "evalplus": 54.60, "hermes": 71.50},
        "4bit":  {"bfcl": 62.10, "evalplus": 56.10, "hermes": 74.60},
        "base":  {"bfcl": 49.10, "evalplus": None, "hermes": 25.00},  # 2 missing
    }
    for row, comp_scores in scores.items():
        for component, score in comp_scores.items():
            cell = {"row": row, "component": component, "status": "ok",
                    "path": "evals/results/{0}-{1}.json".format(component, row)}
            if score is None:
                cell["status"] = "failed"
                cell["exit_code"] = 1
                cell["score"] = None
            else:
                cell["score"] = score
            cells.append(cell)
    return {"ran": 9, "skipped": 0, "failed": 3, "elapsed_s": 9.0, "cells": cells}


def _write_results_for(results_dir: str, summary: dict):
    """Materialize the per-cell result JSONs the summary points at."""
    for cell in summary["cells"]:
        if cell.get("status") != "ok":
            continue
        # Subscores are component-shaped; only the fields we need for the
        # breakdown footer. Keep small so the test is fast.
        if cell["component"] == "bfcl":
            sub = {"complete": True, "missing_categories": [], "partial_categories": [],
                   "per_category": {"overall": {"n": 800, "passed": int(round(cell["score"] * 8.0))}}}
        elif cell["component"] == "evalplus":
            sub = {"complete": True, "missing_datasets": [], "partial_datasets": [],
                   "humaneval_base_pct": cell["score"],
                   "mbpp_base_pct": max(0.0, cell["score"] - 5.0),
                   "humaneval_plus_pct": max(0.0, cell["score"] - 6.0),
                   "mbpp_plus_pct": max(0.0, cell["score"] - 11.0)}
        else:  # hermes
            n_pass = int(round(cell["score"] / 100.0 * 40))
            sub = {"n_cases": 40, "n_errors": 0, "n_passed": n_pass,
                   "per_category": {"single": {"n": 20, "passed": int(n_pass * 0.5)},
                                    "parallel": {"n": 10, "passed": int(n_pass * 0.25)},
                                    "system2": {"n": 5, "passed": int(n_pass * 0.125)},
                                    "negative": {"n": 5, "passed": int(n_pass * 0.125)}}}
        result = _make_result(cell["component"], cell["row"], score=cell["score"], subscores=sub)
        # Use the *result dir under test* rather than the summary's path so
        # the test reads what it wrote.
        path = results_mod.result_path(results_dir, cell["component"], cell["row"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
            fh.write("\n")
        cell["path"] = path  # overwrite so the report resolves locally


# ---------------------------------------------------------------------------
# Anchor loading
# ---------------------------------------------------------------------------


class LoadAnchorsTest(unittest.TestCase):
    """The anchors file is the only source of vendor figures."""

    def test_anchors_file_exists(self):
        # The file ships in the repo (per design choice #4).
        self.assertTrue(
            os.path.exists(_ANCHORS_PATH),
            "expected vendored anchors at {0!r}".format(_ANCHORS_PATH),
        )

    def test_loads_required_keys(self):
        anchors = report_mod.load_anchors(_ANCHORS_PATH)
        # Both model families must be present, with the locked metric keys.
        self.assertIn("duoneural_v3", anchors)
        self.assertIn("liquidai_base", anchors)
        self.assertIn("evalplus", anchors["duoneural_v3"])
        self.assertIn("hermes", anchors["duoneural_v3"])
        self.assertIn("bfcl", anchors["liquidai_base"])
        # The exact metric keys issue #7 cites.
        dvp = anchors["duoneural_v3"]["evalplus"]
        self.assertIn("humaneval_base_pct", dvp)
        self.assertIn("humaneval_plus_pct", dvp)
        self.assertIn("mbpp_base_pct", dvp)
        self.assertIn("mbpp_plus_pct", dvp)
        self.assertIn("fc_ast_pct", anchors["duoneural_v3"]["hermes"])
        self.assertIn("bfclv3_pct", anchors["liquidai_base"]["bfcl"])

    def test_anchors_have_source_url_and_retrieval_date(self):
        # The vendored file records provenance so the diff carries it
        # (design choice #4).
        with open(_ANCHORS_PATH, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("retrieved", text)
        self.assertIn("huggingface.co", text)
        self.assertIn("DuoNeural", text)
        self.assertIn("LiquidAI", text)

    def test_unknown_anchor_path_raises(self):
        with self.assertRaises((FileNotFoundError, OSError, ValueError)):
            report_mod.load_anchors("/nonexistent/anchors.yaml")


# ---------------------------------------------------------------------------
# A/B delta table (pure builder)
# ---------------------------------------------------------------------------


class ABDeltaTableTest(unittest.TestCase):
    """Per-component sections with BF16 | 8-bit Δ | 4-bit Δ | base Δ."""

    def test_full_summary_emits_three_sections_in_components_order(self):
        summary = _full_summary()
        rows = report_mod.build_ab_table(summary, baseline_row="bf16")
        # One entry per component, in evals.results.COMPONENTS order.
        self.assertEqual([r.component for r in rows], ["bfcl", "evalplus", "hermes"])

    def test_bf16_row_has_no_deltas_against_baseline_and_shows_its_own_score(self):
        rows = report_mod.build_ab_table(_full_summary(), baseline_row="bf16")
        bfcl_section = next(r for r in rows if r.component == "bfcl")
        bf16_row = next(r for r in bfcl_section.rows if r.row == "bf16")
        # BF16's own score cell is the baseline -> no Δ vs baseline column.
        self.assertEqual(bf16_row.score_pct, 62.10)
        self.assertIsNone(bf16_row.delta_vs_baseline_pp)
        # delta_vs_base_row IS computed (BF16 vs base) — that's a separate
        # comparator and is rendered only when the matrix includes both rows.
        self.assertAlmostEqual(bf16_row.delta_vs_base_row_pp, 13.0, places=4)

    def test_delta_sign_matches_quant_minus_baseline(self):
        # 8-bit BFCL = 61.60, BF16 BFCL = 62.10 -> delta = -0.5 pp.
        rows = report_mod.build_ab_table(_full_summary(), baseline_row="bf16")
        bfcl_section = next(r for r in rows if r.component == "bfcl")
        eight = next(r for r in bfcl_section.rows if r.row == "8bit")
        self.assertAlmostEqual(eight.delta_vs_baseline_pp, -0.5, places=4)

    def test_delta_against_base_row_when_base_is_present(self):
        # 8-bit BFCL = 61.60, base BFCL = 49.10 -> delta_vs_base_row = +12.5.
        rows = report_mod.build_ab_table(_full_summary(), baseline_row="bf16")
        bfcl_section = next(r for r in rows if r.component == "bfcl")
        eight = next(r for r in bfcl_section.rows if r.row == "8bit")
        self.assertAlmostEqual(eight.delta_vs_base_row_pp, 12.5, places=4)

    def test_missing_score_renders_as_none_score(self):
        rows = report_mod.build_ab_table(_partial_summary(), baseline_row="bf16")
        hermes_section = next(r for r in rows if r.component == "hermes")
        bf16_row = next(r for r in hermes_section.rows if r.row == "bf16")
        self.assertIsNone(bf16_row.score_pct)
        self.assertIsNone(bf16_row.delta_vs_baseline_pp)

    def test_unknown_baseline_raises(self):
        with self.assertRaises(ValueError):
            report_mod.build_ab_table(_full_summary(), baseline_row="not-a-row")


# ---------------------------------------------------------------------------
# Cross-check table (pure builder + verdict bands)
# ---------------------------------------------------------------------------


class CrossCheckTableTest(unittest.TestCase):
    """Local vs vendor figures; verdict from |Δ| bands (1.0 / 3.0 pp)."""

    def _builder_scores(self):
        # Local scores mirror the full summary above.
        return {
            "bfcl": {"bfclv3_pct": 62.10},
            "evalplus": {"humaneval_base_pct": 56.10,
                         "humaneval_plus_pct": 50.00,
                         "mbpp_base_pct": 60.80,
                         "mbpp_plus_pct": 49.70},
            "hermes": {"fc_ast_pct": 75.00},
        }

    def _vendor_anchors(self):
        return {
            "duoneural_v3": {
                "evalplus": {
                    "humaneval_base_pct": 56.1,
                    "humaneval_plus_pct": 50.0,
                    "mbpp_base_pct": 60.8,
                    "mbpp_plus_pct": 49.7,
                },
                "hermes": {"fc_ast_pct": 100.0},
            },
            "liquidai_base": {
                "bfcl": {"bfclv3_pct": 64.79},
            },
        }

    def test_verified_band(self):
        # |Δ| ≤ 1.0 -> verified. (evalplus humaneval_base: local 56.10 vs vendor 56.1 -> 0.0)
        rows = report_mod.build_crosscheck_table(
            self._builder_scores(), self._vendor_anchors(),
            matrix_row="bf16", vendor_model="duoneural_v3",
        )
        evals = next(s for s in rows if s.section == "EvalPlus (duoneural_v3 vendor)")
        he = next(c for c in evals.cells if c.metric == "humaneval_base_pct")
        self.assertAlmostEqual(he.local_pct, 56.10)
        self.assertAlmostEqual(he.vendor_pct, 56.1)
        self.assertAlmostEqual(he.delta_pp, 0.0)
        self.assertEqual(he.verdict, "verified")

    def test_approximate_band(self):
        # |Δ| in (1.0, 3.0] pp -> approximate. The vendor says 98.0;
        # local = 100.0 -> |Δ| = 2.0 -> approximate.
        vendor = {
            "duoneural_v3": {
                "evalplus": {
                    "humaneval_base_pct": 56.1,
                    "humaneval_plus_pct": 50.0,
                    "mbpp_base_pct": 60.8,
                    "mbpp_plus_pct": 49.7,
                },
                "hermes": {"fc_ast_pct": 98.0},
            },
            "liquidai_base": {"bfcl": {"bfclv3_pct": 64.79}},
        }
        approx_scores = {
            "bfcl": {"bfclv3_pct": 62.10},
            "evalplus": {"humaneval_base_pct": 56.10,
                         "humaneval_plus_pct": 50.00,
                         "mbpp_base_pct": 60.80,
                         "mbpp_plus_pct": 49.70},
            "hermes": {"fc_ast_pct": 100.0},  # |100 - 98| = 2.0
        }
        rows = report_mod.build_crosscheck_table(
            approx_scores, vendor, matrix_row="bf16", vendor_model="duoneural_v3",
        )
        hermes = next(s for s in rows if s.section == "Hermes FC (duoneural_v3 vendor)")
        cell = next(c for c in hermes.cells if c.metric == "fc_ast_pct")
        self.assertAlmostEqual(cell.delta_pp, 2.0, places=4)
        self.assertEqual(cell.verdict, "approximate")

    def test_disputed_band(self):
        # |Δ| > 3.0 -> disputed.
        scores = self._builder_scores()  # hermes fc_ast_pct = 75.00
        rows = report_mod.build_crosscheck_table(
            scores, self._vendor_anchors(),
            matrix_row="bf16", vendor_model="duoneural_v3",
        )
        hermes = next(s for s in rows if s.section == "Hermes FC (duoneural_v3 vendor)")
        cell = next(c for c in hermes.cells if c.metric == "fc_ast_pct")
        self.assertEqual(cell.verdict, "disputed")

    def test_base_row_uses_liquidai_anchors(self):
        # matrix_row='base' flips to the LiquidAI anchor set; local key
        # is bfclv3_pct (matches the vendor metric name).
        scores = {"bfcl": {"bfclv3_pct": 49.10},
                  "evalplus": {}, "hermes": {}}
        rows = report_mod.build_crosscheck_table(
            scores, self._vendor_anchors(),
            matrix_row="base", vendor_model="liquidai_base",
        )
        bfcl = next(s for s in rows if "BFCL" in s.section)
        cell = next(c for c in bfcl.cells if c.metric == "bfclv3_pct")
        self.assertAlmostEqual(cell.local_pct, 49.10)
        self.assertAlmostEqual(cell.vendor_pct, 64.79)
        self.assertEqual(cell.verdict, "disputed")  # |49.10 - 64.79| > 3.0

    def test_missing_metric_renders_as_none(self):
        scores = {"bfcl": {}, "evalplus": {}, "hermes": {}}
        rows = report_mod.build_crosscheck_table(
            scores, self._vendor_anchors(),
            matrix_row="bf16", vendor_model="duoneural_v3",
        )
        # Every cell has no local value -> local_pct is None, verdict = 'no_local'.
        for section in rows:
            for cell in section.cells:
                self.assertIsNone(cell.local_pct)
                self.assertEqual(cell.verdict, "no_local")

    def test_unknown_vendor_model_raises(self):
        with self.assertRaises(ValueError):
            report_mod.build_crosscheck_table(
                self._builder_scores(), self._vendor_anchors(),
                matrix_row="bf16", vendor_model="not-a-vendor",
            )


# ---------------------------------------------------------------------------
# Render — the full Markdown document (verifies column layout + footnote)
# ---------------------------------------------------------------------------


class RenderReportTest(unittest.TestCase):
    """The Markdown produced is the model-card artifact."""

    def test_full_summary_renders_all_three_sections(self):
        with tempfile.TemporaryDirectory() as results_dir:
            summary = _full_summary()
            _write_results_for(results_dir, summary)
            md = report_mod.render_report(
                summary,
                results_dir=results_dir,
                anchors_path=_ANCHORS_PATH,
                baseline_row="bf16",
            )
        # Both table titles must appear.
        self.assertIn("A/B delta vs BF16 baseline", md)
        self.assertIn("Base-vs-published cross-check", md)
        # Each component section appears in the A/B table.
        for component in ("BFCL v3 multi-turn", "EvalPlus", "Hermes FC"):
            self.assertIn(component, md)

    def test_partial_summary_renders_missing_footnote(self):
        with tempfile.TemporaryDirectory() as results_dir:
            summary = _partial_summary()
            _write_results_for(results_dir, summary)
            md = report_mod.render_report(
                summary,
                results_dir=results_dir,
                anchors_path=_ANCHORS_PATH,
                baseline_row="bf16",
            )
        self.assertIn("Missing cells", md)
        # The three failed cells (bf16/hermes, base/evalplus, base/hermes is ok actually)
        # -- check the failed ones we modeled.
        self.assertIn("bf16", md)
        self.assertIn("hermes", md)
        # The literal sentinel must appear at least once in a cell.
        self.assertIn("— (not run)", md)

    def test_delta_signs_rendered_with_signs(self):
        with tempfile.TemporaryDirectory() as results_dir:
            summary = _full_summary()
            _write_results_for(results_dir, summary)
            md = report_mod.render_report(
                summary,
                results_dir=results_dir,
                anchors_path=_ANCHORS_PATH,
                baseline_row="bf16",
            )
        # Per the new layout, each section is a Variant | Score | Δ table.
        # The BFCL section's deltas vs BF16 baseline (62.10):
        #   8-bit (61.60) -> -0.5; 4-bit (62.10) -> 0.0; base (49.10) -> -13.0.
        self.assertIn("-0.5", md)
        self.assertNotIn("+0.0", md)
        self.assertIn("0.0", md)
        self.assertIn("-13.0", md)

    def test_render_writes_to_output_arg(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = os.path.join(tmp, "results")
            os.makedirs(results_dir)
            summary = _full_summary()
            _write_results_for(results_dir, summary)
            out_path = os.path.join(tmp, "report.md")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(report_mod.render_report(
                    summary,
                    results_dir=results_dir,
                    anchors_path=_ANCHORS_PATH,
                    baseline_row="bf16",
                ))
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("A/B delta vs BF16 baseline", text)

    def test_verdict_icons_render(self):
        # `verified` = green check, `approximate` = yellow ~, `disputed` = red x.
        with tempfile.TemporaryDirectory() as results_dir:
            summary = _full_summary()
            _write_results_for(results_dir, summary)
            md = report_mod.render_report(
                summary,
                results_dir=results_dir,
                anchors_path=_ANCHORS_PATH,
                baseline_row="bf16",
            )
        # Hermes FC (75 vs 100) is disputed; EvalPlus has "approx" cells (verified).
        self.assertIn("disputed", md)
        self.assertIn("verified", md)

    def test_subscores_breakdown_under_each_section(self):
        with tempfile.TemporaryDirectory() as results_dir:
            summary = _full_summary()
            _write_results_for(results_dir, summary)
            md = report_mod.render_report(
                summary,
                results_dir=results_dir,
                anchors_path=_ANCHORS_PATH,
                baseline_row="bf16",
            )
        # The hermes section footer should mention the per-category counts.
        self.assertIn("single", md)
        self.assertIn("parallel", md)
        self.assertIn("system2", md)
        self.assertIn("negative", md)


# ---------------------------------------------------------------------------
# Sabotage run — the regression test must fail on a broken implementation.
# ---------------------------------------------------------------------------


class SabotageRunTest(unittest.TestCase):
    """If a builder regresses, the test must fail loudly."""

    def test_ab_table_sabotage_breaks_verified_band(self):
        rows = report_mod.build_ab_table(_full_summary(), baseline_row="bf16")
        bfcl = next(r for r in rows if r.component == "bfcl")
        eight = next(r for r in bfcl.rows if r.row == "8bit")
        # If the sign convention is broken (e.g. BF16 - quant instead of
        # quant - BF16), this assertion fails.
        self.assertLess(eight.delta_vs_baseline_pp, 0.0)


if __name__ == "__main__":
    unittest.main()