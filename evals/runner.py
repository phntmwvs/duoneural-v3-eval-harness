"""Common runner contract (wayfinder Q2 / ticket #6).

Every component adapter implements this contract: given a target checkpoint,
bring up ``mlx_lm serve`` on a free port, run the component, and return a
normalized result. Matrix orchestration and ``--resume`` semantics are defined
in ticket #6; this is the canonical shape they conform to.

The result schema (normalized JSON)::

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


class ComponentRunner(Protocol):
    """Protocol every component adapter satisfies (ticket #6)."""

    name: str

    def run(self, checkpoint: str, *, resume: bool = False) -> dict[str, Any]:
        """Run the component against ``checkpoint`` and return the normalized
        result dict described in the module docstring.

        ``resume=True`` continues an interrupted run from persisted state rather
        than restarting (required for the long BFCL multi-turn runs).
        """
        ...
