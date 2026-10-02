"""Normalized result helpers (wayfinder tickets #6 / #16).

Ticket #6 (resolution 4a) locks the result schema::

    {
      "component":  "bfcl" | "evalplus" | "hermes",
      "model":      "<matrix row name, e.g. 'bf16'>",
      "checkpoint": "<HF repo id or local path>",
      "score":      <float, primary metric>,
      "subscores":  {<component-specific breakdown>},
      "runtime_s":  <float>,
      "artifact_versions": {"mlx_lm": "...", "bfcl_eval": "...", ...},
      "timestamp":  "<iso8601>"
    }

This module is the single place that knows that shape. ``new_result`` builds a
dict with every locked key present (so a component can't silently drop one),
and ``validate_result`` checks it (used by the scaffold's stub emitters and by
tests; adapters may call it before writing). Component-specific data goes in
``subscores`` only — no new top-level fields.

Stdlib-only so it imports in every component venv.
"""

from __future__ import annotations

import datetime
import json
import os

# The locked top-level keys (ticket #6, resolution 4a).
RESULT_KEYS = (
    "component",
    "model",
    "checkpoint",
    "score",
    "subscores",
    "runtime_s",
    "artifact_versions",
    "timestamp",
)

COMPONENTS = ("bfcl", "evalplus", "hermes")


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with a ``Z`` suffix."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def new_result(
    *,
    component: str,
    model: str,
    checkpoint: str,
    score: float,
    subscores=None,
    runtime_s: float,
    artifact_versions=None,
    timestamp: str | None = None,
) -> dict:
    """Build a normalized result dict with all locked keys present."""
    if component not in COMPONENTS:
        raise ValueError("unknown component {0!r} (expected one of {1})".format(component, COMPONENTS))
    return {
        "component": component,
        "model": model,
        "checkpoint": checkpoint,
        "score": float(score),
        "subscores": dict(subscores) if subscores else {},
        "runtime_s": float(runtime_s),
        "artifact_versions": dict(artifact_versions) if artifact_versions else {},
        "timestamp": timestamp if timestamp is not None else utc_now_iso(),
    }


def validate_result(result: dict) -> list[str]:
    """Return a list of schema violations (empty list = valid)."""
    errors = []
    if not isinstance(result, dict):
        return ["result is not a dict"]
    for key in RESULT_KEYS:
        if key not in result:
            errors.append("missing key: {0}".format(key))
    component = result.get("component")
    if component is not None and component not in COMPONENTS:
        errors.append("component {0!r} not in {1}".format(component, COMPONENTS))
    if "score" in result and not isinstance(result["score"], (int, float)):
        errors.append("score is not numeric: {0!r}".format(type(result["score"]).__name__))
    if "subscores" in result and not isinstance(result["subscores"], dict):
        errors.append("subscores is not a dict")
    if "artifact_versions" in result and not isinstance(result["artifact_versions"], dict):
        errors.append("artifact_versions is not a dict")
    if "runtime_s" in result and not isinstance(result["runtime_s"], (int, float)):
        errors.append("runtime_s is not numeric")
    return errors


def result_path(results_dir: str, component: str, model: str) -> str:
    """Path a component's result JSON is written to (one file per run).

    ``evals/results/`` is gitignored; files are JSON-per-run named
    ``<component>-<model>.json`` so the matrix's A/B delta can glob one row.
    """
    return os.path.join(results_dir, "{0}-{1}.json".format(component, model))


def write_result(result: dict, results_dir: str) -> str:
    """Validate + write ``result`` under ``results_dir``; return the path."""
    errors = validate_result(result)
    if errors:
        raise ValueError("invalid result: {0}".format("; ".join(errors)))
    os.makedirs(results_dir, exist_ok=True)
    path = result_path(results_dir, result["component"], result["model"])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path
