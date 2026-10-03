"""Normalize BFCL score/result files into the harness result schema.

Stdlib-only so it is importable and testable in any interpreter (the BFCL
adapter runs in ``.venv-bfcl``, but the matrix's A/B delta and these tests run
in the core venv).

Inputs (written by ``bfcl generate`` / ``bfcl evaluate`` under
``BFCL_PROJECT_ROOT``):

- Result files: ``result/<model_key>/<subdir>/BFCL_v4_<category>_result.json``.
  Each is a JSON object keyed by test-id; each value has ``id`` and a
  per-turn ``result`` list (one inner list per turn). We count turns from it
  (the thrash-loop guard) and detect completeness (``--resume``).
- Score files: ``score/<model_key>/<subdir>/BFCL_v4_<category>_score.json``.
  Each is a JSON list whose first element is a header dict with ``accuracy``,
  ``correct_count``, ``total_count`` (see
  ``bfcl_eval.eval_checker.eval_runner_helper.save_eval_results``).

Score for the component = unweighted mean of the four multi-turn category
accuracies, matching BFCL's own ``calculate_unweighted_accuracy`` over the
multi-turn table (``data_multi_turn.csv``).
"""

from __future__ import annotations

import json
import os

#: The BFCL v3 multi-turn categories this component drives (ticket #17).
MULTI_TURN_CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)

#: Filename prefix BFCL stamps on every result/score file (``VERSION_PREFIX``).
#: Confirmed ``BFCL_v4`` against the installed ``bfcl_eval==2026.3.23``.
VERSION_PREFIX = "BFCL_v4"


def result_filename(category: str) -> str:
    """BFCL's result-file name for a category."""
    return "{0}_{1}_result.json".format(VERSION_PREFIX, category)


def score_filename(category: str) -> str:
    """BFCL's score-file name for a category."""
    return "{0}_{1}_score.json".format(VERSION_PREFIX, category)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_score_accuracy(score_file: str):
    """Return ``(accuracy, correct_count, total_count)`` from a BFCL score file.

    The score file is a list whose element 0 is the header dict inserted by
    ``save_eval_results``. Returns ``None`` if the file is missing or has no
    usable header.
    """
    if not os.path.exists(score_file):
        return None
    data = _load_json(score_file)
    if not isinstance(data, list) or not data:
        return None
    header = data[0]
    if not isinstance(header, dict) or "accuracy" not in header:
        return None
    accuracy = header.get("accuracy")
    correct = header.get("correct_count")
    total = header.get("total_count")
    if accuracy is None:
        return None
    try:
        accuracy = float(accuracy)
    except (TypeError, ValueError):
        return None
    return accuracy, correct, total


def _iter_result_entries(result_obj):
    """Yield ``(test_id, record)`` pairs from a BFCL result-file object.

    Generation writes the result file as a JSON object keyed by test-id
    (confirmed against the #3 spike's ``BFCL_v4_multi_turn_base_result.json``:
    a dict of one record keyed by id). Tolerate a top-level list too, in case a
    category writes a list of records.
    """
    if isinstance(result_obj, dict):
        # One-record file: keys are field names, not ids.
        if "id" in result_obj and "result" in result_obj:
            yield result_obj.get("id"), result_obj
            return
        for key, value in result_obj.items():
            yield key, value
    elif isinstance(result_obj, list):
        for record in result_obj:
            if isinstance(record, dict):
                yield record.get("id"), record


def count_turns_per_id(result_file: str):
    """Return ``{test_id: n_turns}`` from a BFCL multi-turn result file.

    ``n_turns`` is the length of the record's per-turn ``result`` list (one
    inner list per conversational turn). Returns an empty dict for a missing
    or unreadable file so the thrash guard degrades to "no data" rather than
    crashing the component.
    """
    if not os.path.exists(result_file):
        return {}
    try:
        obj = _load_json(result_file)
    except (ValueError, OSError):
        return {}
    counts = {}
    for test_id, record in _iter_result_entries(obj):
        if test_id is None or not isinstance(record, dict):
            continue
        turns = record.get("result")
        if isinstance(turns, list):
            counts[str(test_id)] = len(turns)
    return counts


def completed_ids(result_file: str):
    """Return the set of test-ids with a complete record in a result file.

    ``--resume`` semantics (ticket #6, decision 7a): a test-id counts as done
    when its record exists in the category's ``result/<model>/`` file and
    carries a non-empty per-turn ``result`` list. BFCL's own ``--run-ids``
    writes only the ids it (re)generated, so a record's presence means it
    completed.
    """
    if not os.path.exists(result_file):
        return set()
    try:
        obj = _load_json(result_file)
    except (ValueError, OSError):
        return set()
    done = set()
    for test_id, record in _iter_result_entries(obj):
        if test_id is None or not isinstance(record, dict):
            continue
        turns = record.get("result")
        if isinstance(turns, list) and len(turns) > 0:
            done.add(str(test_id))
    return done


def find_result_file(result_root: str, model_key: str, category: str):
    """Locate a category's result file under ``result_root/<model_key>/``.

    BFCL nests results in a category-specific subdirectory
    (``get_directory_structure_by_category``); for the multi-turn categories
    that is ``multi_turn``. Search recursively so we do not hard-code the
    subdirectory depth.
    """
    target = result_filename(category)
    base = os.path.join(result_root, model_key)
    for dirpath, _dirnames, filenames in os.walk(base):
        if target in filenames:
            return os.path.join(dirpath, target)
    return None


def find_score_file(score_root: str, model_key: str, category: str):
    """Locate a category's score file under ``score_root/<model_key>/``."""
    target = score_filename(category)
    base = os.path.join(score_root, model_key)
    for dirpath, _dirnames, filenames in os.walk(base):
        if target in filenames:
            return os.path.join(dirpath, target)
    return None


def normalize(result_root: str, score_root: str, model_key: str,
              categories=MULTI_TURN_CATEGORIES):
    """Build ``(score, per_category, missing)`` from BFCL's output tree.

    ``score`` is the unweighted mean of the available category accuracies
    (``None`` if no category scored). ``per_category`` maps each category to
    ``{"accuracy", "correct_count", "total_count"}`` when present.
    ``missing`` lists categories with no score file (or no usable header) so
    the caller can surface an incomplete run.
    """
    per_category = {}
    missing = []
    accuracies = []
    for category in categories:
        score_file = find_score_file(score_root, model_key, category)
        parsed = read_score_accuracy(score_file) if score_file else None
        if parsed is None:
            missing.append(category)
            continue
        accuracy, correct, total = parsed
        per_category[category] = {
            "accuracy": accuracy,
            "correct_count": correct,
            "total_count": total,
        }
        accuracies.append(accuracy)
    score = (sum(accuracies) / len(accuracies)) if accuracies else None
    return score, per_category, missing
