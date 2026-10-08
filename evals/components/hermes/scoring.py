"""One-set subset scoring for the Hermes FC suite (tickets #4 / #19).

The scoring rule locked in #4 and extended in the #19 contract:

* **Predicted calls** are parsed from the model's raw completion: the #4
  dialect ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``
  (possibly several per completion). Calls whose ``name`` is not in the
  case's ``tools[]`` are **ungrounded** and ignored (Q6); extra predicted
  calls against *valid* tools are also ignored — only the expected set is
  scored (one-set subset match: every expected call must be matched).
* **Name match** is exact (case-sensitive).
* **Argument match** is a recursive equality after normalization: strings
  compare case-insensitive + trimmed; dicts compare key sets exactly
  (strictness on optional args, Q7 — a predicted key the expected call
  omits is a mismatch, including explicit ``null``); lists compare
  element-wise in order; numbers/bools compare by value (``1`` == ``1.0``,
  but bools never equal numbers).
* ``command_equivalents`` (Q5/c1): on a string-typed expected arg, the
  predicted string may instead equal any listed equivalent (after the same
  case-insensitive + trim normalization). Ignored for non-string args.
* **Parallel** cases match the expected set against the predicted list
  order-insensitively unless the case has ``depends_on_order: true``,
  in which case expected calls must match predicted calls in order.
* **Negative** cases pass iff the completion contains no ``<tool_call>``
  tag at all (any dialect — the v3-dialect opening tag also fails a
  negative case, since the model *attempted* a call).
* **System2** cases (``require_thought: true``) additionally require a
  ``<thought>...</thought>`` block; the thought is stripped and the
  remainder is scored by the rules above.

Stdlib-only, pure functions — no I/O, no server interaction.
"""

from __future__ import annotations

import json
import re

# --- dialect patterns (mirrors evals/hermes_fc/probe_thought_emission.py) ---

#: ``<thought>...</thought>`` — exact tag, no whitespace variants.
THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)

#: #4 dialect opening/closing tags. The payload between them is extracted
#: brace-balanced (``extract_d4_payloads``) because the naive
#: ``<tool_call>\s*(\{.*?\})\s*</tool_call>`` regex stops at the first ``}``
#: whenever the call has a nested ``arguments`` object.
TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>")

#: Any tool-call attempt, either dialect: the #4 opening tag, or a v3-style
#: ``<tool_call name="..." ...>`` / unclosed ``<tool_call ...`` prefix. Used
#: for the negative gate ("no <tool_call> in the response", #4) — a case is
#: failed by an *attempted* call even if it never parses.
ANY_CALL_RE = re.compile(r"<tool_call>|<tool_call\s+[^>]*?>|<tool_call\s+[^>]*$", re.DOTALL)


def extract_thought(content):
    """Return ``(thought_text, remainder)`` for the first ``<thought>`` block.

    ``thought_text`` is ``None`` when no block is present; ``remainder`` is
    the content with the first block removed (what gets scored for calls).
    """
    m = THOUGHT_RE.search(content)
    if m is None:
        return None, content
    return m.group(1), content[: m.start()] + content[m.end():]


def has_any_call_attempt(content):
    """True if the completion contains any ``<tool_call>`` attempt."""
    return ANY_CALL_RE.search(content) is not None


def _extract_balanced_object(text, start):
    """Return ``(payload, raw)`` for the JSON object starting at
    ``text[start] == '{'``.

    Walks the text tracking brace depth and string state. At each closing
    brace that returns depth to zero we try ``json.JSONDecoder().raw_decode``
    on the substring: raw_decode handles ``\\``-escaped quotes correctly
    (a naive string-skip treats ``\\"`` as ending the string and mis-counts
    braces — caught in review on ``{"cmd": "echo \\"hi\\"}"}``). If the
    balanced substring doesn't decode (the model put an unquoted ``}`` in a
    value, or truncated), we keep scanning for the next closing brace so a
    slightly-over-long tail can still salvage the payload.

    Returns ``(dict_or_None, raw_substring)``: ``payload`` is the decoded
    dict on success, ``None`` when the braces never balance or never
    decode; ``raw`` is always the substring examined (for error reports).
    """
    depth = 0
    in_string = False
    escape = False
    decoder = json.JSONDecoder()
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                raw = text[start:i + 1]
                try:
                    payload, _ = decoder.raw_decode(raw)
                except ValueError:
                    continue  # over-long candidate; keep scanning for '}'
                if isinstance(payload, dict):
                    return payload, raw
                return None, raw
    return None, text[start:]


def parse_predicted_calls(content):
    """Parse all #4-dialect ``<tool_call>`` payloads from ``content``.

    Returns a list of dicts. Unparseable or truncated payloads are kept as
    ``{"_parse_error": True, "raw": ...}`` so the scorer can report them;
    they never match an expected call. The v3 dialect is intentionally not
    parsed in v1.0 (see PROBE_RESULTS.md — proposed ticket #28).
    """
    calls = []
    for m in TOOL_CALL_OPEN_RE.finditer(content):
        # Skip whitespace between the opening tag and the payload.
        i = m.end()
        while i < len(content) and content[i] in " \t\r\n":
            i += 1
        if i >= len(content) or content[i] != "{":
            # Not a #4-dialect payload (e.g. a v3-style attribute tag) —
            # the ANY_CALL gate handles attempt detection; nothing to parse.
            continue
        payload, raw = _extract_balanced_object(content, i)
        if payload is None:
            calls.append({"_parse_error": True, "raw": raw})
        else:
            calls.append(payload)
    return calls


# --- normalization + recursive equality -------------------------------------


def _norm_str(value):
    """Case-insensitive + trim normalization for string comparison (#4)."""
    return value.strip().lower()


def _norm_number(value):
    """Normalize ints/floats so ``1`` and ``1.0`` compare equal.

    Bools are excluded (``True`` is an ``int`` subclass but must never
    equal ``1``); they compare by identity in ``arg_equal``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return float(value)
    return value


def arg_equal(expected, predicted, *, equivalents=None):
    """Recursive argument equality under the #4 normalization rules.

    ``equivalents`` is the ``command_equivalents`` list for this specific
    argument (only consulted when ``expected`` is a string).
    """
    if isinstance(expected, str):
        if not isinstance(predicted, str):
            return False
        if _norm_str(predicted) == _norm_str(expected):
            return True
        if equivalents:
            return any(_norm_str(predicted) == _norm_str(eq) for eq in equivalents)
        return False
    if isinstance(expected, bool):
        return isinstance(predicted, bool) and predicted == expected
    if isinstance(expected, (int, float)):
        if isinstance(predicted, bool) or not isinstance(predicted, (int, float)):
            return False
        return _norm_number(predicted) == _norm_number(expected)
    if isinstance(expected, list):
        if not isinstance(predicted, list) or len(predicted) != len(expected):
            return False
        return all(arg_equal(e, p) for e, p in zip(expected, predicted))
    if isinstance(expected, dict):
        if not isinstance(predicted, dict):
            return False
        # Strictness on optional args (Q7): key sets must match exactly —
        # an expected-omitted key predicted (even as null) is a mismatch.
        if set(predicted.keys()) != set(expected.keys()):
            return False
        return all(arg_equal(expected[k], predicted[k]) for k in expected)
    # None and anything else: exact equality.
    return predicted == expected


def call_matches(expected_call, predicted_call):
    """One expected call vs one predicted call (name-exact + arg-equal)."""
    if predicted_call.get("_parse_error"):
        return False
    if predicted_call.get("name") != expected_call.get("name"):
        return False
    expected_args = expected_call.get("arguments") or {}
    predicted_args = predicted_call.get("arguments")
    if not isinstance(predicted_args, dict):
        return False
    # Strictness on optional args (Q7): identical key sets.
    if set(predicted_args.keys()) != set(expected_args.keys()):
        return False
    equivalents = expected_call.get("command_equivalents") or []
    for key, exp_val in expected_args.items():
        # command_equivalents applies to any string-typed arg (Q5/c1); in
        # the v1.0 cases it is only ever the shell `command` arg, but the
        # field is named generically so we apply it per-arg.
        eqs = equivalents if isinstance(exp_val, str) else None
        if not arg_equal(exp_val, predicted_args[key], equivalents=eqs):
            return False
    return True


def match_calls_unordered(expected_calls, predicted_calls):
    """Order-insensitive one-set subset match (parallel default).

    Every expected call must consume a distinct predicted call. Extra
    predicted calls are ignored (Q6). Returns ``(matched, unmatched_idx)``
    where ``unmatched_idx`` are the indices of expected calls with no match.
    """
    remaining = list(range(len(predicted_calls)))
    unmatched = []
    for exp in expected_calls:
        hit = None
        for idx in remaining:
            if call_matches(exp, predicted_calls[idx]):
                hit = idx
                break
        if hit is None:
            unmatched.append(exp)
        else:
            remaining.remove(hit)
    return not unmatched, unmatched


def match_calls_ordered(expected_calls, predicted_calls):
    """Ordered match (``depends_on_order: true``): expected calls must
    appear in order among the predicted calls (extra predicted calls
    interleaved anywhere are ignored — the *expected* sequence must be a
    subsequence of the predicted list)."""
    pos = 0
    unmatched = []
    for exp in expected_calls:
        hit = None
        for idx in range(pos, len(predicted_calls)):
            if call_matches(exp, predicted_calls[idx]):
                hit = idx
                break
        if hit is None:
            unmatched.append(exp)
        else:
            pos = hit + 1
    return not unmatched, unmatched


# --- the case-level scorer ---------------------------------------------------


def score_case(case, content):
    """Score one case against one raw completion. Returns a detail dict::

        {
          "passed": bool,
          "reason": "<short machine-readable verdict>",
          "thought_present": bool | None,     # only meaningful on system2
          "predicted_calls": [...],           # grounded, parsed calls
          "ungrounded_names": [...],          # predicted names not in tools[]
          "parse_errors": int,
          "unmatched_expected": [...],        # expected calls with no match
        }

    Never raises on malformed model output — a broken completion is a
    failed case, not a harness error.
    """
    category = case.get("category")
    expected = (case.get("expected") or {}).get("calls") or []
    tool_names = {t.get("function", {}).get("name") for t in case.get("tools") or []}

    thought_present = None
    remainder = content or ""

    # System2 gate (#4 amendment): require <thought>, score the stripped
    # remainder.
    if case.get("require_thought"):
        thought, remainder = extract_thought(remainder)
        thought_present = thought is not None
        if not thought_present:
            return _detail(False, "missing_thought", thought_present)

    # Negative gate (#4): pass iff the model made no tool-call attempt.
    if category == "negative":
        attempted = has_any_call_attempt(remainder)
        return _detail(
            not attempted,
            "no_call_as_expected" if not attempted else "unexpected_call_attempt",
            thought_present,
        )

    raw_calls = parse_predicted_calls(remainder)
    parse_errors = sum(1 for c in raw_calls if c.get("_parse_error"))
    parsed = [c for c in raw_calls if not c.get("_parse_error")]
    grounded = [c for c in parsed if c.get("name") in tool_names]
    ungrounded_names = [c.get("name") for c in parsed if c.get("name") not in tool_names]

    if case.get("depends_on_order"):
        ok, unmatched = match_calls_ordered(expected, grounded)
    else:
        ok, unmatched = match_calls_unordered(expected, grounded)

    if ok:
        reason = "match"
    elif not grounded:
        reason = "no_parseable_call" if parse_errors or not parsed else "no_grounded_call"
    else:
        reason = "arguments_mismatch"
    return _detail(
        ok,
        reason,
        thought_present,
        predicted_calls=grounded,
        ungrounded_names=ungrounded_names,
        parse_errors=parse_errors,
        unmatched_expected=unmatched,
    )


def _detail(passed, reason, thought_present, **extra):
    d = {
        "passed": bool(passed),
        "reason": reason,
        "thought_present": thought_present,
        "predicted_calls": [],
        "ungrounded_names": [],
        "parse_errors": 0,
        "unmatched_expected": [],
    }
    d.update(extra)
    return d
