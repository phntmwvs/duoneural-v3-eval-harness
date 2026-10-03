"""Custom Hermes FC suite component adapter (ticket #19).

Stub. Loads the hand-authored cases (``evals/hermes_fc/cases/``, ticket #12),
drives the shared ``ServerManager``'s ``base_url`` chat+tools endpoint, and
scores with the one-set subset match (ticket #4), emitting the normalized
result JSON. Runs in the shared core venv (no extra deps — SETUP.md §5).
"""
