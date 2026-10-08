"""Custom Hermes FC suite runner — orchestration (ticket #19).

Loads the hand-authored cases from ``evals/hermes_fc/cases/`` (ticket #12),
runs each against the live ``mlx_lm server`` handed in by the shared
``ServerManager`` (ticket #16 — the adapter never owns serve), scores each
completion with the one-set subset match (``scoring.py``, ticket #4 plus
the #19 contract extensions), and emits the normalized result JSON
(ticket #6, resolution 4a).

Request shape (Q1 from the #19 contract): the case's ``tools[]`` are sent
in the OpenAI chat-completions ``tools`` field; the served checkpoint's own
``chat_template.jinja`` renders the ``<tools>...</tools>`` block of the
system prompt. Decoding is greedy (temperature=0, n=1, stream=False) with a
512-token completion cap — the cap the system2 probe sized so a
``<thought>`` + call fits (PROBE_RESULTS.md).

``--resume`` (ticket #6, decision 7a): per-case outcomes persist to
``evals/hermes_runs/<model>/state.json`` after every case; a resumed run
skips cases already recorded. A fresh run clears the state first.

Stdlib-only: runs in the shared core venv (``.venv-core``, SETUP.md §5).
"""

from __future__ import annotations

import glob
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

# Allow running as ``python -m evals.components.hermes`` and under tests.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evals import results as results_mod  # noqa: E402

from . import scoring  # noqa: E402

COMPONENT = "hermes"

#: ``mlx_lm server`` registers its loaded checkpoint under the literal key
#: ``default_model`` (see evals/components/evalplus/runner.py:DEFAULT_MODEL_KEY
#: for the full rationale). Sending it always selects the served checkpoint.
#: Deliberately the canonical server key, *not* BFCL's ``/v1/models`` lookup:
#: the latter exists because BFCL sends a registry key that ``mlx_lm`` would
#: 404 as an HF id (``handler.py:_server_model_id``), whereas ``default_model``
#: is mapped server-side to whatever checkpoint is loaded and also sidesteps
#: the empty-``/v1/models`` edge case when the checkpoint ref is an HF id.
DEFAULT_MODEL_KEY = "default_model"

#: Greedy decoding, one completion — reproducible pass/fail per case.
DECODING = {"temperature": 0.0, "n": 1, "stream": False}

#: Completion-token cap. The system2 probe (PROBE_RESULTS.md) sized 512 so
#: a ``<thought>`` + ``<tool_call>`` fits; the same cap is used for every
#: category so the suite is uniform.
MAX_TOKENS = 512

#: Case-directory constants.
CASES_DIR = os.path.join(_REPO_ROOT, "evals", "hermes_fc", "cases")
CATEGORIES = ("single", "parallel", "negative", "system2")

#: Per-request HTTP timeout (one greedy 512-token generation; the probe
#: measured ~1.2 s/call on the 4-bit quant, so 120 s is generous).
REQUEST_TIMEOUT_S = 120.0


def load_cases(cases_dir=None):
    """Load every case JSON in ``cases_dir`` (default: the suite dir).

    Returns the list sorted by case id — a stable, reviewable execution
    order. Raises ``FileNotFoundError`` if the directory has no cases (an
    empty suite is a setup error, not a 0-score run).
    """
    cases_dir = cases_dir or CASES_DIR
    files = sorted(glob.glob(os.path.join(cases_dir, "*.json")))
    if not files:
        raise FileNotFoundError("no Hermes FC case files in {0}".format(cases_dir))
    cases = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            cases.append(json.load(fh))
    cases.sort(key=lambda c: c.get("id", ""))
    return cases


def build_payload(case, model=DEFAULT_MODEL_KEY, *, max_tokens=MAX_TOKENS):
    """Wrap a case's prompt (+ optional system override) and tools into the
    chat-completions request body.

    The case file carries only the user-facing ``prompt`` and the tool
    schemas; ``tools`` goes in the request's ``tools`` field and the served
    checkpoint's ``chat_template.jinja`` injects the ``<tools>`` block
    (#19 contract, Q1). ``tool_choice: auto`` lets the model decline to
    call — required for the negative category to be meaningful.
    """
    messages = []
    if case.get("system"):
        messages.append({"role": "system", "content": case["system"]})
    messages.append({"role": "user", "content": case["prompt"]})
    payload = {
        "model": model,
        "messages": messages,
        "tools": case.get("tools") or [],
        "tool_choice": "auto",
        "max_tokens": max_tokens,
    }
    payload.update(DECODING)
    return payload


def post_chat(base_url, payload, timeout_s=REQUEST_TIMEOUT_S):
    """POST one chat-completions request to the served model. Stdlib only.

    The seam tests patch: the live path talks to ``mlx_lm server``; tests
    substitute a mock that returns a canned completion.

    ``base_url`` points at the server root (e.g. ``http://127.0.0.1:8080/v1``);
    any trailing slash is stripped so a ``--base-url .../v1/`` does not
    produce a ``.../v1//chat/completions`` double slash (404s live — M2).
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_content(resp):
    """Pull the assistant message content out of a chat-completions response.

    Raises ``ValueError`` on a malformed payload (missing choices/message) —
    the runner records that as a case error, not a crash.
    """
    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("malformed chat-completions response: {0}".format(exc))
    return content or ""


# --- resume state (ticket #6, decision 7a) -----------------------------------


def _state_path(run_root):
    return os.path.join(run_root, "state.json")


def load_state(run_root):
    """Read the per-model run state; ``{"cases": {id: detail}}`` or empty."""
    path = _state_path(run_root)
    if not os.path.exists(path):
        return {"cases": {}}
    with open(path, encoding="utf-8") as fh:
        state = json.load(fh)
    state.setdefault("cases", {})
    return state


def save_state(run_root, state):
    os.makedirs(run_root, exist_ok=True)
    path = _state_path(run_root)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")


def run_case(case, *, base_url, model=DEFAULT_MODEL_KEY):
    """Run one case: one chat+tools call, one scored completion.

    Returns the ``scoring.score_case`` detail dict plus ``case_id`` and
    ``category``. Transport/protocol failures are recorded as
    ``{"error": ...}`` entries (a failed case), never raised — one case's
    network blip must not kill a 40-case row.
    """
    detail = {"case_id": case.get("id"), "category": case.get("category")}
    payload = build_payload(case, model)
    try:
        resp = post_chat(base_url, payload)
        content = extract_content(resp)
    except (urllib.error.URLError, OSError, socket.timeout, TimeoutError,
            ValueError) as exc:
        detail.update({
            "passed": False,
            "reason": "request_error",
            "error": "{0}: {1}".format(type(exc).__name__, exc),
        })
        return detail
    scored = scoring.score_case(case, content)
    detail.update(scored)
    detail["content"] = content
    return detail


def run(checkpoint, *, base_url, model_name=None, resume=False,
        run_root=None, results_dir=None, cases_dir=None):
    """Run the Hermes FC component and write the normalized result JSON.

    ``checkpoint`` is the checkpoint ref (local dir or HF id); ``base_url``
    is the live ``mlx_lm server`` handed in by the shared ``ServerManager``
    (ticket #16) — per the runner contract the adapter never owns serve.
    ``model_name`` is the matrix row name for the result ``model`` field /
    filename (defaults to ``checkpoint``).

    ``--resume`` (ticket #6, decision 7a): cases already present in
    ``state.json`` are kept; only the missing ones run. A fresh run clears
    the state first.

    Returns the normalized result dict (also written to ``results_dir``).
    """
    model_name = model_name if model_name is not None else checkpoint
    run_root = run_root or os.path.join(
        _REPO_ROOT, "evals", "hermes_runs", results_mod.slug_filename(model_name)
    )
    results_dir = results_dir or os.path.join(_REPO_ROOT, "evals", "results")

    start = time.monotonic()
    cases = load_cases(cases_dir)

    if resume:
        state = load_state(run_root)
    else:
        state = {"cases": {}}
        save_state(run_root, state)
    done = state["cases"]

    n_run = 0
    for case in cases:
        cid = case.get("id")
        if cid in done:
            continue
        detail = run_case(case, base_url=base_url)
        done[cid] = detail
        save_state(run_root, state)  # persist after every case (crash-safe)
        n_run += 1

    runtime_s = time.monotonic() - start

    n_pass = sum(1 for d in done.values() if d.get("passed"))
    n_total = len(done)
    per_category = {}
    for cat in CATEGORIES:
        cat_details = [d for d in done.values() if d.get("category") == cat]
        if cat_details:
            per_category[cat] = {
                "n": len(cat_details),
                "passed": sum(1 for d in cat_details if d.get("passed")),
            }
    n_errors = sum(1 for d in done.values() if d.get("error"))
    score = (n_pass / n_total) if n_total else 0.0

    result = results_mod.new_result(
        component=COMPONENT,
        model=model_name,
        checkpoint=checkpoint,
        score=score,
        subscores={
            "n_cases": n_total,
            "n_passed": n_pass,
            "n_errors": n_errors,
            "per_category": per_category,
            "decoding": dict(DECODING, max_tokens=MAX_TOKENS),
            "scoring": "one-set subset match (name-exact + recursive arg-equal; "
                       "case-insensitive+trim strings; parallel order-insensitive "
                       "unless depends_on_order; negative=no <tool_call>; "
                       "system2=<thought> gate + stripped remainder)",
            "resume": bool(resume),
            "cases_ran_this_run": n_run,
        },
        runtime_s=runtime_s,
        artifact_versions=_artifact_versions(),
    )
    errors = results_mod.validate_result(result)
    if errors:
        raise ValueError("invalid result: {0}".format("; ".join(errors)))
    results_mod.write_result(result, results_dir)
    return result


def _artifact_versions():
    """Best-effort versions of the hermes adapter's stack (stdlib + mlx_lm)."""
    versions = {"component": COMPONENT, "python": sys.version.split()[0]}
    try:
        import mlx_lm  # noqa: PLC0415
        versions["mlx_lm"] = getattr(mlx_lm, "__version__", "unknown")
    except Exception:  # pragma: no cover - mlx_lm lives in .venv-core
        pass
    return versions
