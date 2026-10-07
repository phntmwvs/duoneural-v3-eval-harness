"""EvalPlus HumanEval+MBPP component adapter (ticket #18).

Runs EvalPlus 0.3.1 (HumanEval+ and MBPP+, base and ``+`` variants) at
pass@1 / temperature 0 against the shared ``ServerManager``'s ``mlx_lm
server`` (ticket #5), emitting the normalized result JSON (ticket #6).

Split (ticket #5 §3): generate on the host in ``.venv-evalplus`` via
``evalplus.codegen`` (chat-only ``openai`` backend against ``mlx_lm serve``),
then score sandboxed via the official ``ganler/evalplus`` Docker image
(``evalplus.evaluate --samples``) so untrusted generated code never executes
on the host.

Layout (mirrors the BFCL adapter, ticket #17):

- ``runner.py`` — stdlib-only orchestration: build the ``evalplus.codegen`` /
  ``docker run ... evalplus.evaluate`` argv and drive them as subprocesses
  (never imports ``evalplus``, so its pure helpers are testable anywhere).
- ``normalize.py`` — stdlib-only parsing of the ``*_eval_results.json`` files
  into the harness result schema (base/plus pass@1 recomputed from statuses).
- ``__main__.py`` — the entry point ``run_eval.py`` subprocesses into
  (``python -m evals.components.evalplus``).

The model is addressed as ``default_model`` — ``mlx_lm server``'s canonical
key for its loaded checkpoint — so a local quant path and an HF-id base row
are both served without served-id resolution (see ``runner.py``).
"""
