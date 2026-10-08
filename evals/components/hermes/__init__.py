"""Custom Hermes FC suite component adapter (ticket #19).

Loads the hand-authored cases (``evals/hermes_fc/cases/``, ticket #12),
drives the shared ``ServerManager``'s ``base_url`` chat+tools endpoint, and
scores with the one-set subset match (ticket #4 + the #19 contract
extensions), emitting the normalized result JSON.

* ``scoring`` — pure scoring/parse functions (no I/O).
* ``runner`` — case loading, chat+tools transport, resume state, result emit.

Runs in the shared core venv (no extra deps — SETUP.md §5).
"""
