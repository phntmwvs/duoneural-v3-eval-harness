"""BFCL v3 multi-turn component adapter (ticket #17).

Stub. The real adapter folds the #3 spike forward behind the ``ComponentRunner``
contract: take a checkpoint, receive a live ``base_url`` from the shared
``ServerManager`` (it never owns serve), run ``bfcl generate``, and emit the
normalized result JSON (``evals/results.py``) with per-turn step counts in
``subscores`` (the thrash-loop guard from #3).
"""
