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

#: Expected number of test entries per multi-turn category, confirmed against
#: ``bfcl_eval==2026.3.23`` (``data/BFCL_v4_<category>.json`` — 200 lines each).
#: Used to flag a partial/truncated run (ticket #17 review, M1): a category
#: whose score ``total_count`` is below this did not evaluate the full set.
EXPECTED_CATEGORY_COUNT = {category: 200 for category in MULTI_TURN_CATEGORIES}


def result_filename(category: str) -> str:
    """BFCL's result-file name for a category."""
    return "{0}_{1}_result.json".format(VERSION_PREFIX, category)


def score_filename(category: str) -> str:
    """BFCL's score-file name for a category."""
    return "{0}_{1}_score.json".format(VERSION_PREFIX, category)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_json_or_jsonl(path):
    """Load a file that is either a single JSON document or JSONL.

    BFCL writes score files as JSONL: the header (accuracy) record first, then
    one per-entry record per line (``write_list_of_dicts_to_file``). Result
    files, by contrast, are a single JSON object. Handle both so the score
    parser reads the real on-disk format.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # JSONL: parse each non-blank line.
    records = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            return None
    return records if records else None


def read_score_accuracy(score_file: str):
    """Return ``(accuracy, correct_count, total_count)`` from a BFCL score file.

    The score file is JSONL whose first record is the header dict inserted by
    ``save_eval_results`` (``accuracy`` / ``correct_count`` / ``total_count``);
    later records are per-entry results. Returns ``None`` if missing or the
    header is unusable.
    """
    if not os.path.exists(score_file):
        return None
    data = _load_json_or_jsonl(score_file)
    header = None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        header = data[0]
    elif isinstance(data, dict):
        # A single-record (or single-line) score file: the record IS the header.
        header = data
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
    """Return ``{test_id: {"turns": n, "max_steps": m, "steps_per_turn": [...]}}``.

    A BFCL multi-turn record's ``result`` field is a per-turn list; each turn
    is itself a list of generation steps (the model acts, a tool responds, the
    model acts again). The thrash signal is a turn that hits BFCL's step cap
    (the model never produced a terminal answer within the turn) — so we
    record both the turn count and the per-turn step counts. Returns an empty
    dict for a missing or unreadable file so the guard degrades to "no data".
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
        if not isinstance(turns, list):
            continue
        steps = [len(t) if isinstance(t, list) else 1 for t in turns]
        counts[str(test_id)] = {
            "turns": len(turns),
            "steps_per_turn": steps,
            "max_steps": max(steps) if steps else 0,
        }
    return counts


def _find_file(root: str, model_key: str, target: str):
    """Locate ``target`` under ``root/<model_key>/`` by recursive walk.

    BFCL nests output in a category-specific subdirectory
    (``get_directory_structure_by_category``); for the multi-turn categories
    that is ``multi_turn``. Search recursively so we do not hard-code the
    subdirectory depth.
    """
    base = os.path.join(root, model_key)
    for dirpath, _dirnames, filenames in os.walk(base):
        if target in filenames:
            return os.path.join(dirpath, target)
    return None


def find_result_file(result_root: str, model_key: str, category: str):
    """Locate a category's result file under ``result_root/<model_key>/``."""
    return _find_file(result_root, model_key, result_filename(category))


def find_score_file(score_root: str, model_key: str, category: str):
    """Locate a category's score file under ``score_root/<model_key>/``."""
    return _find_file(score_root, model_key, score_filename(category))


def normalize(result_root: str, score_root: str, model_key: str,
              categories=MULTI_TURN_CATEGORIES):
    """Build ``(score, per_category, missing, partial)`` from BFCL's output.

    ``score`` is the unweighted mean of the available category accuracies
    (``None`` if no category scored). ``per_category`` maps each category to
    ``{"accuracy", "correct_count", "total_count", "expected_count"}`` when
    present. ``missing`` lists categories with no score file (or no usable
    header). ``partial`` lists categories that scored but evaluated fewer
    entries than expected (``total_count < expected_count``) — a truncated or
    mid-resume run the caller must surface, not treat as complete (M1).
    """
    per_category = {}
    missing = []
    partial = []
    accuracies = []
    for category in categories:
        score_file = find_score_file(score_root, model_key, category)
        parsed = read_score_accuracy(score_file) if score_file else None
        if parsed is None:
            missing.append(category)
            continue
        accuracy, correct, total = parsed
        expected = EXPECTED_CATEGORY_COUNT.get(category)
        per_category[category] = {
            "accuracy": accuracy,
            "correct_count": correct,
            "total_count": total,
            "expected_count": expected,
        }
        if expected is not None and isinstance(total, (int, float)) \
                and total < expected:
            partial.append(category)
        accuracies.append(accuracy)
    score = (sum(accuracies) / len(accuracies)) if accuracies else None
    return score, per_category, missing, partial
