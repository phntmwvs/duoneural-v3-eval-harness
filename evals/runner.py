"""Common runner contract (wayfinder Q2 / ticket #6).

Every component adapter implements this contract: given a target checkpoint,
it is handed a live ``base_url`` by the shared ``ServerManager`` (ticket #16 —
adapters never own serve), runs the component, and returns a normalized result.
Matrix orchestration and ``--resume`` semantics are defined in ticket #6; this
is the canonical shape they conform to.

The result schema (ticket #6, resolution 4a) is defined once in
``evals/results.py`` (``RESULT_KEYS`` / ``new_result`` / ``validate_result``)::

    {
      "component": "bfcl" | "evalplus" | "hermes",
      "model": "<matrix row name, e.g. 'bf16'>",
      "checkpoint": "<HF repo id or local path>",
      "score": <float, primary metric>,
      "subscores": {<component-specific breakdown>},
      "runtime_s": <float>,
      "artifact_versions": {"mlx_lm": "...", "bfcl_eval": "...", ...},
      "timestamp": "<iso8601>"
    }
"""

from __future__ import annotations

from typing import Any, Protocol

from . import results as _results

#: Re-exported so adapters import the contract + its schema from one place.
RESULT_KEYS = _results.RESULT_KEYS
COMPONENTS = _results.COMPONENTS
new_result = _results.new_result
validate_result = _results.validate_result


class ComponentRunner(Protocol):
    """Protocol every component adapter satisfies (ticket #6).

    The adapter receives a live ``base_url`` from the shared ``ServerManager``
    (it never starts ``mlx_lm server`` itself) and returns a dict matching
    ``RESULT_KEYS`` — validate with ``validate_result`` before writing.

    Adapters are dispatched as modules (``python -m evals.components.<comp>``),
    so the component id is a module-level constant named ``COMPONENT`` (e.g.
    ``evals/components/bfcl/runner.py: COMPONENT = "bfcl"``), not a class
    attribute.
    """

    COMPONENT: str

    def run(self, checkpoint: str, *, base_url: str, resume: bool = False) -> dict[str, Any]:
        """Run the component against ``checkpoint`` served at ``base_url`` and
        return the normalized result dict described above (see
        ``evals/results.py``).

        ``resume=True`` continues an interrupted run from persisted state rather
        than restarting (required for the long BFCL multi-turn runs).
        """
        ...
