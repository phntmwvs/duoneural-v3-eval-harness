"""Normalize EvalPlus output files into the harness result schema.

Stdlib-only so it is importable and testable in any interpreter (the
EvalPlus adapter runs in ``.venv-evalplus``, but the matrix's A/B delta and
these tests run in the core venv).

Inputs (written by ``evalplus.codegen`` / ``evalplus.evaluate`` under the
run root — ticket #5 §4):

- Sample files: ``<root>/<dataset>/<identifier>_openai_temp_0.0.jsonl``
  (sanitized solutions; the ``.raw.jsonl`` sibling is ignored).
- Eval results: ``<root>/<dataset>/<identifier>_openai_temp_0.0_eval_results.json``
  (written by ``evalplus.evaluate --samples``), a single JSON object::

      {"date": ..., "hash": "<md5>", "eval": { "<task_id>": [
          {"task_id", "solution", "base_status", "plus_status",
           "base_fail_tests", "plus_fail_tests"} ] }}

Pass@1 is recomputed from ``eval`` exactly as EvalPlus does (it prints pass@k
to stdout but does not store it in the JSON): with ``--greedy`` (n=1) there is
one completion per task, and

- base pass@1 = mean over tasks of ``base_status == "pass"``
- plus pass@1 = mean over tasks of ``base_status == "pass" AND plus_status == "pass"``
  (EvalPlus defines the ``+`` score on the base∩plus intersection — see
  ``evalplus/evaluate.py`` ``new_correct``).

Score for the component = mean of the two ``+`` pass@1s (humaneval+ and
mbpp+), the leaderboard-style "Average" headline number (ticket #5 §4.3).
"""

from __future__ import annotations

import glob
import json
import os

#: EvalPlus dataset names this component drives (ticket #5).
DATASETS = ("humaneval", "mbpp")

#: Dataset Plus versions pinned by evalplus 0.3.1 (ticket #5 §2.1, traced to
#: ``evalplus/data/humaneval.py`` / ``evalplus/data/mbpp.py``). Recorded for
#: provenance; the authoritative per-file hash lives in each
#: ``_eval_results.json`` ``hash`` field.
DATASET_VERSIONS = {"humaneval": "v0.1.10", "mbpp": "v0.2.0"}

#: Expected task counts at pass@1 (HumanEval+ 164 + MBPP+ 378 = 542 gens/row;
#: ticket #5 §2.1). Used to flag a partial/truncated run.
EXPECTED_TASK_COUNT = {"humaneval": 164, "mbpp": 378}

#: EvalPlus status strings (``evalplus/eval/__init__.py``).
PASS = "pass"


def find_samples_file(root, dataset):
    """Return the sanitized sample JSONL for ``dataset`` under ``root``.

    ``evalplus.codegen`` writes ``<root>/<dataset>/<id>.jsonl`` (sanitized)
    and a ``.raw.jsonl`` sibling. Return the newest non-raw JSONL (the
    sanitized one) or ``None``.
    """
    pattern = os.path.join(root, dataset, "*.jsonl")
    candidates = [p for p in glob.glob(pattern) if not p.endswith(".raw.jsonl")]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def find_eval_results_file(root, dataset):
    """Return the ``*_eval_results.json`` for ``dataset`` under ``root``."""
    pattern = os.path.join(root, dataset, "*_eval_results.json")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def clear_samples(root, dataset):
    """Delete existing sample JSONL (and ``.raw.jsonl``) for ``dataset``.

    ``evalplus.codegen`` always opens its target in append mode and never
    truncates, so a fresh run must remove the prior sample file first —
    otherwise codegen appends a second copy of every task. Only the resume
    path preserves samples (codegen's own per-task skip then appends just the
    missing ids). Deletes both ``*.jsonl`` and their ``*.raw.jsonl`` siblings.
    """
    pattern = os.path.join(root, dataset, "*.jsonl")
    for path in glob.glob(pattern):
        os.remove(path)


def clear_eval_results(root, dataset):
    """Delete existing ``*_eval_results.json`` for ``dataset``.

    ``evalplus.evaluate --samples`` skips re-evaluation when a result file
    already exists (it "Load[s] from previous results"), so a resume run that
    appended new samples must clear the stale result first — otherwise the
    new samples are never scored. Evaluation is cheap (a few minutes in
    Docker); generation is the expensive part that ``--resume`` preserves.
    """
    pattern = os.path.join(root, dataset, "*_eval_results.json")
    for path in glob.glob(pattern):
        os.remove(path)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def compute_pass_at_1(eval_results):
    """Return ``(base_pass1, plus_pass1, n_tasks)`` from a loaded
    ``_eval_results.json``.

    Matches EvalPlus exactly: per task ``res`` (the list of completions for
    that task), ``base pass@1`` is ``mean(base_status == "pass" / n)`` and
    ``plus pass@1`` is ``mean((base == plus == "pass") / n)`` across tasks
    (``estimate_pass_at_k`` with k=1 reduces to the per-task pass rate, and
    with ``--greedy`` n=1 there is one completion per task).
    """
    eval_dict = eval_results.get("eval", {}) if isinstance(eval_results, dict) else {}
    base_rates = []
    plus_rates = []
    for _task_id, completions in eval_dict.items():
        if not isinstance(completions, list) or not completions:
            continue
        n = len(completions)
        base = sum(1 for c in completions
                   if isinstance(c, dict) and c.get("base_status") == PASS)
        plus = sum(1 for c in completions
                   if isinstance(c, dict) and c.get("base_status") == PASS
                   and c.get("plus_status") == PASS)
        base_rates.append(base / n)
        plus_rates.append(plus / n)
    n_tasks = len(base_rates)
    base_pass1 = sum(base_rates) / n_tasks if n_tasks else 0.0
    plus_pass1 = sum(plus_rates) / n_tasks if n_tasks else 0.0
    return base_pass1, plus_pass1, n_tasks


def read_dataset_hash(eval_results):
    """Return the dataset ``hash`` field (``None`` if absent)."""
    if not isinstance(eval_results, dict):
        return None
    return eval_results.get("hash")


def load_dataset_result(root, dataset):
    """Load + score one dataset's ``*_eval_results.json``.

    Returns ``None`` if the file is missing/unreadable, else a dict::

        {"base_pass1", "plus_pass1", "n_tasks", "expected_tasks", "hash"}
    """
    result_file = find_eval_results_file(root, dataset)
    if result_file is None:
        return None
    try:
        data = _load_json(result_file)
    except (ValueError, OSError):
        return None
    base_pass1, plus_pass1, n_tasks = compute_pass_at_1(data)
    return {
        "base_pass1": base_pass1,
        "plus_pass1": plus_pass1,
        "n_tasks": n_tasks,
        "expected_tasks": EXPECTED_TASK_COUNT.get(dataset),
        "hash": read_dataset_hash(data),
    }


def normalize(root):
    """Build ``(score, per_dataset, missing, partial)`` from a run root.

    ``score`` is the mean of the two ``+`` pass@1s (``None`` if no dataset
    scored). ``per_dataset`` maps each dataset to ``load_dataset_result``'s
    dict. ``missing`` lists datasets with no eval results. ``partial`` lists
    datasets that scored but evaluated fewer tasks than expected (a truncated
    or mid-resume run the caller must surface, not treat as complete).
    """
    per_dataset = {}
    missing = []
    partial = []
    plus_scores = []
    for dataset in DATASETS:
        res = load_dataset_result(root, dataset)
        if res is None:
            missing.append(dataset)
            continue
        per_dataset[dataset] = res
        expected = res["expected_tasks"]
        if expected is not None and res["n_tasks"] < expected:
            partial.append(dataset)
        plus_scores.append(res["plus_pass1"])
    score = (sum(plus_scores) / len(plus_scores)) if plus_scores else None
    return score, per_dataset, missing, partial
