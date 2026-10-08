"""Eval report builders — ticket #7 (wayfinder grill).

Renders two tables for the DuoNeural v3 MLX model card:

1. **A/B delta vs BF16 baseline** — per-component sections with one row
   per matrix row (``bf16 | 8-bit Δ | 4-bit Δ | base Δ``). The BF16 row
   has no delta cells (it *is* the baseline). Δ = quant − BF16 in
   percentage points with explicit sign (matches the DuoNeural card's
   own `+19.3% Δ (v3 vs Stock)` convention).

2. **Base-vs-published cross-check** — the locally-measured score vs
   the upstream vendor figure (DuoNeural v3 card for the BF16 row,
   LiquidAI card for the ``base`` row). Verdict bands:

   * ``|Δ| ≤ 1.0 pp`` → ``verified``
   * ``1.0 < |Δ| ≤ 3.0 pp`` → ``approximate``
   * ``|Δ| > 3.0 pp`` → ``disputed`` (always flagged)

The vendored anchors live in ``evals/report_anchors.yaml`` (source URLs
+ retrieval date in the frontmatter). The matrix's ``--summary-json``
is the canonical input — the report reads per-cell result files *only*
for the subscores footer (per-category BFCL counts, per-dataset EvalPlus
pass counts, per-category Hermes FC pass counts).

Stdlib-only so the module imports in every venv.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from typing import Optional

from . import results as results_mod

#: Tolerance verdict bands in pp (ticket #7 grill, decision #5).
VERIFIED_BAND_PP = 1.0
APPROXIMATE_BAND_PP = 3.0

#: Per-row vendor anchor mapping. ``"bf16"`` matches the DuoNeural v3
#: card (the published claim being verified). ``"base"`` matches the
#: LiquidAI base card (independent verification that the stock reference
#: matches LiquidAI's published BFCLv3 number, etc.).
ROW_VENDOR_ANCHORS = {
    "bf16": "duoneural_v3",
    "8bit": "duoneural_v3",
    "4bit": "duoneural_v3",
    "base": "liquidai_base",
}

#: Verdict glyphs (one-character icons that survive Markdown tables of any
#: renderer that supports emoji, and remain readable in plain text).
VERDICT_GLYPH = {
    "verified":    "✓ verified",
    "approximate": "~ approximate",
    "disputed":    "✗ disputed",
    "no_local":    "? no local",
    "no_vendor":   "? no vendor",
}

#: Verdict ordering for sortability in tests / future CI hooks.
_VERDICT_ORDER = {"verified": 0, "approximate": 1, "no_local": 2,
                  "no_vendor": 2, "disputed": 3}


# ---------------------------------------------------------------------------
# Data shapes (dataclasses; pure-data, JSON-serializable for tests)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ABScoreRow:
    """One matrix row's score + its deltas, for one component."""
    row: str
    score_pct: Optional[float]      # percent (e.g. 62.10) or None on missing
    delta_vs_baseline_pp: Optional[float]   # quant − BF16 in pp
    delta_vs_base_row_pp: Optional[float]  # this row − base row in pp


@dataclasses.dataclass(frozen=True)
class ABSection:
    component: str
    rows: tuple  # tuple of ABScoreRow


@dataclasses.dataclass(frozen=True)
class CrossCheckCell:
    metric: str
    local_pct: Optional[float]
    vendor_pct: Optional[float]
    delta_pp: Optional[float]   # local − vendor in pp
    verdict: str                # "verified" / "approximate" / "disputed" / "no_local" / "no_vendor"


@dataclasses.dataclass(frozen=True)
class CrossCheckSection:
    section: str    # e.g. "BFCL (duoneural_v3 vendor)"
    cells: tuple    # tuple of CrossCheckCell


# ---------------------------------------------------------------------------
# Anchor loading — minimal YAML reader (no PyYAML dependency).
# ---------------------------------------------------------------------------


def _strip_yaml_comments(text: str) -> str:
    """Strip ``#`` comments from YAML, preserving trailing '#' inside strings.

    The anchors file is well-known, hand-authored, and only contains
    top-level keys + scalar floats — we don't need a real YAML parser.
    A line that starts with ``#`` is a comment; an inline ``#`` is one
    too. Anything inside a single/double-quoted string is preserved.
    """
    cleaned = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            cleaned.append("")
            continue
        # Strip an inline "# …" but only if the # isn't inside a quoted
        # string on this line. The anchors file doesn't use quoted
        # strings, so the simple rule works.
        idx = raw.find("#")
        if idx >= 0:
            raw = raw[:idx].rstrip()
        cleaned.append(raw)
    return "\n".join(cleaned)


def load_anchors(path: str) -> dict:
    """Load the anchors YAML at ``path`` into a nested dict.

    The shape::

        {
          "duoneural_v3": {
            "evalplus": {"humaneval_base_pct": 56.1, ...},
            "hermes":    {"fc_ast_pct": 100.0},
          },
          "liquidai_base": {
            "bfcl": {"bfclv3_pct": 64.79},
          },
        }

    Frontmatter (``# source: …``, ``# retrieved: …``) lives in comment
    lines; provenance is the *file* the diff carries, not the loaded
    dict. We don't surface it from ``load_anchors`` to keep the data
    shape clean for the builders.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "anchors file not found: {0!r} (see design choice #4 in ticket #7)".format(path)
        )
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    text = _strip_yaml_comments(raw)

    result: dict = {}
    # Indentation: top-level (0 spaces), section (2 spaces), metric (4 spaces).
    current_top = None
    current_section = None
    for line in text.splitlines():
        if not line.strip():
            continue
        # Count leading spaces.
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if indent == 0:
            # Top-level key.
            key = content.rstrip(":")
            result[key] = {}
            current_top = key
            current_section = None
            continue
        if indent == 2:
            # Section key under a top-level.
            key = content.rstrip(":")
            if current_top is None:
                raise ValueError(
                    "section '{0}' with no top-level key in {1!r}".format(key, path)
                )
            result[current_top][key] = {}
            current_section = key
            continue
        if indent == 4:
            # Metric key: value.
            if ":" not in content:
                raise ValueError(
                    "expected ':' in metric line: {0!r}".format(content)
                )
            key, _, value = content.partition(":")
            key = key.strip()
            value = value.strip()
            if current_top is None or current_section is None:
                raise ValueError(
                    "metric '{0}' outside section in {1!r}".format(key, path)
                )
            # Floats only (the anchors file is percentages in percent units).
            try:
                result[current_top][current_section][key] = float(value)
            except ValueError as exc:
                raise ValueError(
                    "metric '{0}' has non-numeric value {2!r} in {1!r}".format(
                        key, path, value
                    )
                ) from exc
            continue
        # Any other indent is malformed.
        raise ValueError(
            "unexpected indent in anchors file {0!r}: {1!r}".format(path, line)
        )

    # Minimal shape validation (so a half-edited anchors file can't silently
    # pass through and produce a report with no vendors).
    for required_top in ("duoneural_v3", "liquidai_base"):
        if required_top not in result:
            raise ValueError(
                "anchors file missing top-level key {0!r}".format(required_top)
            )
    if "evalplus" not in result["duoneural_v3"]:
        raise ValueError("anchors file missing duoneural_v3.evalplus section")
    if "hermes" not in result["duoneural_v3"]:
        raise ValueError("anchors file missing duoneural_v3.hermes section")
    if "bfcl" not in result["liquidai_base"]:
        raise ValueError("anchors file missing liquidai_base.bfcl section")
    return result


# ---------------------------------------------------------------------------
# A/B delta table builder
# ---------------------------------------------------------------------------


def _score_lookup(summary: dict) -> dict:
    """Flatten ``summary['cells']`` into ``{(component, row): score_or_None}``."""
    out: dict = {}
    for cell in summary.get("cells", []):
        key = (cell.get("component"), cell.get("row"))
        score = cell.get("score")
        out[key] = score
    return out


def _delta_pp(quant: Optional[float], base: Optional[float]) -> Optional[float]:
    if quant is None or base is None:
        return None
    return quant - base


def build_ab_table(
    summary: dict,
    *,
    baseline_row: str = "bf16",
) -> tuple:
    """Build the per-component A/B table from a matrix summary.

    Returns a tuple of ``ABSection`` ordered by ``evals.results.COMPONENTS``.
    Each section contains one ``ABScoreRow`` per matrix row in the
    summary (preserving order).

    ``baseline_row`` is the row other rows delta against. ``base`` row
    (if present) is *also* included for completeness, but its
    ``delta_vs_base_row_pp`` is ``None`` (it is the reference for that
    column; the column header makes that obvious).
    """
    if baseline_row not in ROW_VENDOR_ANCHORS and baseline_row not in {"bf16", "8bit", "4bit", "base"}:
        raise ValueError(
            "unknown baseline_row {0!r}; expected one of {1}".format(
                baseline_row, sorted(set(list(ROW_VENDOR_ANCHORS) + ["base"]))
            )
        )
    score_map = _score_lookup(summary)

    # Preserve row order from the summary (matrix ran rows in some order;
    # that's the order the user sees in the per-row log too).
    rows_in_summary = []
    seen = set()
    for cell in summary.get("cells", []):
        row = cell.get("row")
        if row and row not in seen:
            seen.add(row)
            rows_in_summary.append(row)

    base_score_for_row = summary.get("eaten_base_score_for_row")  # not present in our schema
    # We compute delta_vs_base_row by reading the "base" row's score.
    base_row = "base"

    sections = []
    for component in results_mod.COMPONENTS:
        baseline_score = score_map.get((component, baseline_row))
        base_row_score = score_map.get((component, base_row))
        rows = []
        for row in rows_in_summary:
            score = score_map.get((component, row))
            delta_baseline = None
            if row != baseline_row and score is not None and baseline_score is not None:
                delta_baseline = _delta_pp(score, baseline_score)
            delta_base = None
            if row != base_row and score is not None and base_row_score is not None:
                delta_base = _delta_pp(score, base_row_score)
            rows.append(ABScoreRow(
                row=row,
                score_pct=score,
                delta_vs_baseline_pp=delta_baseline,
                delta_vs_base_row_pp=delta_base,
            ))
        sections.append(ABSection(component=component, rows=tuple(rows)))
    return tuple(sections)


# ---------------------------------------------------------------------------
# Cross-check table builder
# ---------------------------------------------------------------------------


def _verdict(local: Optional[float], vendor: Optional[float]) -> tuple:
    """Return ``(delta_pp, verdict)`` for ``local − vendor``."""
    if local is None:
        return None, "no_local"
    if vendor is None:
        return None, "no_vendor"
    delta = local - vendor
    adelta = abs(delta)
    if adelta <= VERIFIED_BAND_PP:
        return delta, "verified"
    if adelta <= APPROXIMATE_BAND_PP:
        return delta, "approximate"
    return delta, "disputed"


# Metric keys per component that the cross-check table renders.
_CROSSCHECK_METRICS = {
    "bfcl":     (("bfclv3_pct",),                           "BFCL (LiquidAI base)"),
    "evalplus": (("humaneval_base_pct", "humaneval_plus_pct",
                  "mbpp_base_pct",     "mbpp_plus_pct"),    "EvalPlus (duoneural_v3 vendor)"),
    "hermes":   (("fc_ast_pct",),                           "Hermes FC (duoneural_v3 vendor)"),
}


def _vendor_section_for(matrix_row: str, component: str) -> tuple:
    """Return ``(vendor_model, anchor_section_key, display_section)``."""
    vendor_model = ROW_VENDOR_ANCHORS.get(matrix_row, "duoneural_v3")
    if component == "bfcl":
        return vendor_model, "bfcl", "BFCL ({0} vendor)".format(
            "LiquidAI base" if vendor_model == "liquidai_base" else "duoneural_v3"
        )
    return vendor_model, component, "{0} ({1} vendor)".format(
        {"evalplus": "EvalPlus", "hermes": "Hermes FC"}.get(component, component),
        "duoneural_v3",
    )


def build_crosscheck_table(
    local_scores: dict,
    anchors: dict,
    *,
    matrix_row: str,
    vendor_model: Optional[str] = None,
) -> tuple:
    """Build the cross-check sections for ``matrix_row``.

    ``local_scores`` shape (per component, in percent units)::

        {
          "bfcl":     {"base_local": 62.10},     # single metric for BFCL
          "evalplus": {"humaneval_base_pct": 56.10, "humaneval_plus_pct": 50.00,
                       "mbpp_base_pct": 60.80,     "mbpp_plus_pct": 49.70},
          "hermes":   {"fc_ast_pct": 75.00},
        }

    ``anchors`` is the dict returned by ``load_anchors``. The vendor
    model defaults from ``ROW_VENDOR_ANCHORS[matrix_row]``; override
    for tests.

    Returns a tuple of ``CrossCheckSection`` (one per component that
    has cross-check metrics). Each section contains one
    ``CrossCheckCell`` per metric.
    """
    if vendor_model is None:
        if matrix_row not in ROW_VENDOR_ANCHORS:
            raise ValueError(
                "unknown matrix_row {0!r}; expected one of {1}".format(
                    matrix_row, sorted(ROW_VENDOR_ANCHORS)
                )
            )
        vendor_model = ROW_VENDOR_ANCHORS[matrix_row]
    if vendor_model not in anchors:
        raise ValueError(
            "unknown vendor_model {0!r}; anchors has {1}".format(
                vendor_model, sorted(anchors)
            )
        )

    sections = []
    for component, (metric_keys, _display_base) in _CROSSCHECK_METRICS.items():
        vm, anchor_key, display = _vendor_section_for(matrix_row, component)
        vendor_metrics = anchors[vm].get(anchor_key, {})
        local_metrics = local_scores.get(component, {}) or {}
        cells = []
        for metric in metric_keys:
            local_pct = local_metrics.get(metric)
            vendor_pct = vendor_metrics.get(metric)
            delta, verdict = _verdict(local_pct, vendor_pct)
            cells.append(CrossCheckCell(
                metric=metric,
                local_pct=local_pct,
                vendor_pct=vendor_pct,
                delta_pp=delta,
                verdict=verdict,
            ))
        sections.append(CrossCheckSection(section=display, cells=tuple(cells)))
    return tuple(sections)


# ---------------------------------------------------------------------------
# Local-score extraction from per-cell result JSONs (subscores footer)
# ---------------------------------------------------------------------------


def _extract_local_scores_from_results(results_dir: str, summary: dict) -> dict:
    """Lift the percent scores the cross-check needs from the per-cell JSONs.

    BFCL emits a single score; EvalPlus emits four (HumanEval base/±, MBPP base/±);
    Hermes FC emits one (fc_ast_pct). We read these from each cell's result file
    rather than the summary so the report never duplicates logic the adapters
    own.
    """
    out: dict = {"bfcl": {}, "evalplus": {}, "hermes": {}}
    for cell in summary.get("cells", []):
        if cell.get("status") != "ok":
            continue
        path = cell.get("path")
        if not path or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            result = json.load(fh)
        comp = result.get("component")
        sub = result.get("subscores") or {}
        if comp == "bfcl":
            out["bfcl"]["base_local"] = result.get("score") * 100.0
        elif comp == "evalplus":
            # EvalPlus stores pass@1 in subscores as floats in [0, 1].
            for key in ("humaneval_base_pct", "humaneval_plus_pct",
                        "mbpp_base_pct", "mbpp_plus_pct"):
                # The adapter writes them as raw floats (0.561). Multiply to %.
                # Older versions may have written them as percentages already.
                v = sub.get(key)
                if v is None:
                    continue
                if v <= 1.0:
                    v = v * 100.0
                out["evalplus"][key] = float(v)
        elif comp == "hermes":
            out["hermes"]["fc_ast_pct"] = result.get("score") * 100.0
    return out


# ---------------------------------------------------------------------------
# Subscores footer — per-component breakdown shown under each section
# ---------------------------------------------------------------------------


def _subscores_footer(component: str, cell_result: Optional[dict]) -> str:
    """Render a one-line breakdown footer from a cell's result JSON."""
    if not cell_result:
        return ""
    sub = cell_result.get("subscores") or {}
    if component == "bfcl":
        cats = sub.get("per_category") or {}
        n_cats = len(cats)
        return "{0} categories, {1} entries total".format(
            n_cats,
            sum(int((c.get("n") or 0)) for c in cats.values()),
        )
    if component == "evalplus":
        he = sub.get("humaneval_base_pct")
        mb = sub.get("mbpp_base_pct")
        if he is None or mb is None:
            return ""
        return "HumanEval {0:.1f}% / MBPP {1:.1f}% (per-pass@1)".format(
            he * (100.0 if he <= 1.0 else 1.0),
            mb * (100.0 if mb <= 1.0 else 1.0),
        )
    if component == "hermes":
        cats = sub.get("per_category") or {}
        parts = []
        for cat in ("single", "parallel", "system2", "negative"):
            c = cats.get(cat) or {}
            parts.append("{0} {n}/{passed}".format(
                cat, n=c.get("n", 0), passed=c.get("passed", 0)
            ))
        return "categories: " + ", ".join(parts)
    return ""


def _load_cell_results(results_dir: str, summary: dict) -> dict:
    """Read each cell's result JSON for the subscores footer. Missing-keyed if absent.

    Cell ``path`` values in the summary are written by ``run_matrix.py``
    as relative-to-repo-root paths (``evals/results/<comp>-<row>.json``).
    When the report is invoked from a different cwd or with a custom
    ``results_dir``, we resolve them against the ``results_dir`` arg so
    the local-render path stays decoupled from the caller's cwd.
    """
    results_dir_abs = os.path.abspath(results_dir)
    out: dict = {}
    for cell in summary.get("cells", []):
        if cell.get("status") != "ok":
            continue
        path = cell.get("path")
        if not path:
            continue
        # Try the absolute path first, then resolve relative to results_dir.
        candidates = [
            path,
            os.path.join(results_dir_abs, os.path.basename(path)),
        ]
        # Also try matching by (component, row) via the canonical filename
        # — the matrix guarantees this naming.
        comp = cell.get("component"); modrow = cell.get("row")
        if comp and modrow:
            candidates.append(results_mod.result_path(results_dir_abs, comp, modrow))
        seen: set = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            if os.path.exists(cand):
                try:
                    with open(cand, encoding="utf-8") as fh:
                        out[(comp, modrow)] = json.load(fh)
                except (OSError, ValueError):
                    continue
                break
    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_pp(p: Optional[float]) -> str:
    """Format a delta in pp with explicit sign and 1 decimal."""
    if p is None:
        return "—"
    sign = "+" if p >= 0 else ""
    return "{0}{1:.1f}".format(sign, p)


def _fmt_score(p: Optional[float]) -> str:
    """Format a raw percent score with 2 decimals; missing -> '—'."""
    if p is None:
        return "— (not run)"
    return "{0:.2f}".format(p)


def render_report(
    summary: dict,
    *,
    results_dir: str,
    anchors_path: str,
    baseline_row: str = "bf16",
) -> str:
    """Render the full Markdown document (A/B + cross-check + missing footnote).

    Output is deterministic (rows/cols in locked order); stdout-friendly
    (no ANSI); a CLI ``--output`` flag writes it to a file as well.
    """
    anchors = load_anchors(anchors_path)
    sections = build_ab_table(summary, baseline_row=baseline_row)
    cell_results = _load_cell_results(results_dir, summary)

    # Build per-matrix-row local scores for the cross-check (BFCL/EvalPlus/Hermes).
    # We use the *first matrix row that has data* per component for BFCL/Hermes;
    # EvalPlus uses the BF16 row's pass@1 (the published claim is for that row).
    # The CLI / tests can override by reading the cell_results directly if
    # needed; the default behavior is "cross-check the *base* row against the
    # LiquidAI vendor, and the *baseline_row* against the DuoNeural vendor".
    rows_present = []
    for cell in summary.get("cells", []):
        if cell.get("row") not in rows_present:
            rows_present.append(cell["row"])

    out = []

    # ----- Header
    out.append("# DuoNeural v3 MLX — eval report\n")
    out.append(
        "_Source: `run_matrix.py --summary-json` (per-cell results in "
        "`{0}/`). Anchors vendored in `evals/report_anchors.yaml`._\n".format(
            os.path.relpath(results_dir, start=".")
        )
    )

    # ----- A/B delta table (per-component sections)
    # Section header per the locked design choice: the four-column
    # caption ("BF16 | 8-bit Δ | 4-bit Δ | base Δ") describes the
    # *comparators* against the BF16 baseline — i.e. one row per matrix
    # row in a tight 3-column table (variant, score, Δ vs BF16). The
    # Δ column value is, depending on which row it sits in, the 8-bit
    # delta, the 4-bit delta, or the base delta — same column, three
    # meanings the row caption makes obvious.
    out.append("## A/B delta vs {0} baseline\n".format(baseline_row.upper()))
    out.append(
        "_Δ = quant − BF16 in percentage points (positive = quant scored "
        "higher than BF16). Matches the DuoNeural card's `+19.3% Δ` style._\n"
    )
    section_titles = {
        "bfcl":     "BFCL v3 multi-turn",
        "evalplus": "EvalPlus (HumanEval + MBPP)",
        "hermes":   "Hermes FC suite",
    }
    for section in sections:
        title = section_titles.get(section.component, section.component)
        out.append("\n### {0} (BF16 | 8-bit Δ | 4-bit Δ | base Δ)\n".format(title))
        out.append("| Variant | Score (pp) | Δ vs {0} (pp) |".format(baseline_row.upper()))
        out.append("| :------ | ---------: | ------------: |")
        for r in section.rows:
            score = _fmt_score(r.score_pct)
            if r.row == baseline_row:
                delta_cell = "—"
            else:
                delta_cell = _fmt_pp(r.delta_vs_baseline_pp)
            variant_label = {
                "bf16":  "BF16 (baseline)",
                "8bit":  "8-bit",
                "4bit":  "4-bit",
                "base":  "base (LiquidAI)",
            }.get(r.row, r.row)
            out.append("| {0:<18} | {1:>9} | {2:>12} |".format(
                variant_label, score, delta_cell
            ))
        # Footer: per-component subscores breakdown (BF16 row used as the
        # canonical "this is what the section scored").
        footer = _subscores_footer(
            section.component,
            cell_results.get((section.component, baseline_row))
            or cell_results.get((section.component, "bf16")),
        )
        if footer:
            out.append("\n_{0}_\n".format(footer))

    # ----- Cross-check table (one per matrix_row that has scores).
    out.append("\n## Base-vs-published cross-check\n")
    out.append(
        "_Local measurements vs upstream vendor figures. Verdict bands: "
        "`|Δ| ≤ {0:.1f} pp` → `verified`; `< {1:.1f}` → `approximate`; "
        "`> {1:.1f}` → `disputed`._\n".format(
            VERIFIED_BAND_PP, APPROXIMATE_BAND_PP
        )
    )

    # Pick the matrix row(s) that have a full BFCL score (the baseline row
    # is what we cross-check vs duoneural_v3; the base row is what we cross-
    # check vs liquidai_base). If the baseline row has no BFCL score, fall
    # back to the first row that does.
    def _row_local_scores(matrix_row: str) -> dict:
        """Lift local scores from per-cell JSON for the cross-check table.

        Key naming matches the vendor anchor metric names so the cross-check
        builder can do straight dict lookups (BFCL → ``bfclv3_pct``,
        EvalPlus → ``humaneval_*_pct`` / ``mbpp_*_pct``, Hermes FC →
        ``fc_ast_pct``). The vendor anchors use the same metric names so the
        cross-check builder stays metric-name-driven.
        """
        out_scores = {"bfcl": {}, "evalplus": {}, "hermes": {}}
        # BFCL — single primary metric, used for both the bf16 row
        # (cross-check vs duoneural_v3 claim) and the base row
        # (cross-check vs liquidai_base claim). Local value is the cell's
        # primary score scaled to pp.
        cell = cell_results.get(("bfcl", matrix_row))
        if cell:
            out_scores["bfcl"]["bfclv3_pct"] = float(cell.get("score") or 0.0) * 100.0
        # EvalPlus — four subscores per row from the per-pass fields.
        cell = cell_results.get(("evalplus", matrix_row))
        if cell:
            sub = cell.get("subscores") or {}
            for key in ("humaneval_base_pct", "humaneval_plus_pct",
                        "mbpp_base_pct", "mbpp_plus_pct"):
                v = sub.get(key)
                if v is None:
                    continue
                if v <= 1.0:
                    v = v * 100.0
                out_scores["evalplus"][key] = float(v)
        # Hermes FC — single primary metric.
        cell = cell_results.get(("hermes", matrix_row))
        if cell:
            out_scores["hermes"]["fc_ast_pct"] = float(cell.get("score") or 0.0) * 100.0
        return out_scores

    rows_to_check = []
    for r in (baseline_row, "base"):
        if any((c.get("component"), r) in cell_results
               for c in summary.get("cells", [])):
            rows_to_check.append(r)
    # De-dup while preserving order.
    seen = set()
    rows_to_check = [r for r in rows_to_check if not (r in seen or seen.add(r))]

    for matrix_row in rows_to_check:
        local = _row_local_scores(matrix_row)
        sections_xc = build_crosscheck_table(
            local, anchors, matrix_row=matrix_row,
        )
        out.append("\n### Matrix row `{0}`\n".format(matrix_row))
        out.append("| Component | Metric | Local (pp) | Vendor (pp) | Δ (pp) | Verdict |")
        out.append("| :-------- | :----- | ---------: | ----------: | -----: | :------ |")
        for s in sections_xc:
            for c in s.cells:
                metric_short = c.metric.replace("_pct", "")
                out.append("| {0} | {1} | {2} | {3} | {4} | {5} |".format(
                    s.section.split(" (")[0],
                    metric_short,
                    _fmt_score(c.local_pct),
                    _fmt_score(c.vendor_pct),
                    _fmt_pp(c.delta_pp),
                    VERDICT_GLYPH.get(c.verdict, c.verdict),
                ))

    # ----- Missing cells footnote (anything not 'ok' in the summary)
    missing = [c for c in summary.get("cells", [])
               if c.get("status") not in ("ok", "skipped_complete")]
    if missing:
        out.append("\n## Missing cells\n")
        for c in missing:
            out.append("- `{0}/{1}` — {2}{3}".format(
                c.get("row"), c.get("component"), c.get("status"),
                " ({0})".format(c.get("error")) if c.get("error") else "",
            ))

    # ----- Provenance footer (pointers to the anchors file + the matrix).
    out.append("\n---\n")
    out.append(
        "_Anchors: `{0}` (vendor figures + source URLs in YAML frontmatter). "
        "Matrix driver: `run_matrix.py --summary-json`. "
        "This report consumes that JSON directly (no re-derivation)._\n".format(
            os.path.relpath(anchors_path, start=".")
        )
    )
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI wrapper (run as ``./report.py`` from the repo root)
# ---------------------------------------------------------------------------


def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Render the DuoNeural v3 MLX eval report (ticket #7)."
    )
    p.add_argument(
        "--summary-json",
        required=True,
        help="path to run_matrix.py --summary-json output",
    )
    p.add_argument(
        "--results-dir",
        default=os.path.join("evals", "results"),
        help="per-cell result JSON directory (default: evals/results)",
    )
    p.add_argument(
        "--anchors",
        default=os.path.join("evals", "report_anchors.yaml"),
        help="path to the vendored anchors YAML (default: evals/report_anchors.yaml)",
    )
    p.add_argument(
        "--baseline-row",
        default="bf16",
        help="matrix row the A/B deltas are computed against (default: bf16)",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="write the rendered Markdown to this path (stdout always)",
    )
    return p


def main(argv=None):
    import argparse
    import sys as _sys

    args = _build_arg_parser().parse_args(argv)

    if not os.path.exists(args.summary_json):
        print(
            "error: summary JSON not found at {0!r}".format(args.summary_json),
            file=_sys.stderr,
        )
        return 2

    with open(args.summary_json, encoding="utf-8") as fh:
        summary = json.load(fh)

    md = render_report(
        summary,
        results_dir=os.path.abspath(args.results_dir),
        anchors_path=os.path.abspath(args.anchors),
        baseline_row=args.baseline_row,
    )
    _sys.stdout.write(md)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(md)
        print("[report] wrote {0}".format(args.output), file=_sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())