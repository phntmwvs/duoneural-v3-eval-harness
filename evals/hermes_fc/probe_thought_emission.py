#!/usr/bin/env python3
"""Live ``<thought>``-emission probe for the Hermes FC suite (ticket #12).

Sends one system2 case (fc-system2-001) N times to an ``mlx_lm server``
instance owned by the shared ``ServerManager`` (ticket #16), and reports
how often the model emits a ``<thought>...</thought>`` block before
its tool call. This gates the System-2 category's scoring rule from
ticket #4: if the model never emits ``<thought>``, the category always
scores 0 and the suite needs a redesign before more system2 cases are
authored.

Stdlib-only (re-uses ``evals.server.ServerManager`` for the serve
lifecycle). Greedy decoding (temperature=0, n=1) — we want a
reproducible pass-rate signal, not sampling diversity.

Usage (on the M4 Pro, with the .venv-core active or .venv-core/bin/python
on PATH):

    ./.venv-core/bin/python evals/hermes_fc/probe_thought_emission.py \\
        --checkpoint checkpoints/DuoNeural-v3-4bit \\
        --case evals/hermes_fc/cases/fc-system2-001.json \\
        --n 10

    # To pin to BF16 (gold reference, more memory):
    ./.venv-core/bin/python evals/hermes_fc/probe_thought_emission.py \\
        --checkpoint checkpoints/DuoNeural-v3-BF16 \\
        --case evals/hermes_fc/cases/fc-system2-001.json \\
        --n 10

Exit code 0 iff the pass rate meets the gate (default: 10/10 — i.e.
the model emitted ``<thought>`` on every generation). The threshold is
configurable via ``--min-pass-rate`` (0.0..1.0) for non-gating runs.

Designed to be invoked from a caffeinate'd shell on the MBP so a
long generation does not collapse throughput. See SETUP.md §5.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.request

# Reuse the shared serve lifecycle (ticket #16) so the probe matches the
# production path exactly.
_REPO_ROOT = __file__.rsplit("/evals/", 1)[0]
if not _REPO_ROOT:
    sys.exit(
        "probe_thought_emission.py: could not resolve repo root from "
        f"__file__={__file__!r}. Run from the repo root or pass an absolute "
        "path to --case."
    )
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from evals.server import ServerManager  # noqa: E402

# ``mlx_lm server`` registers its loaded checkpoint under the literal key
# ``default_model`` (see evals/components/evalplus/runner.py:DEFAULT_MODEL_KEY).
# Using it sidesteps the HF-id empty-/v1/models edge case without a served-id
# lookup (we don't need one for a chat+tools POST).
DEFAULT_MODEL_KEY = "default_model"

# Match the Hermes <thought>...</thought> tag exactly. The model was trained on
# <thought> per bfcl/handler.py:_THOUGHT_RE — no whitespace variants, no other
# dialects (Qwen uses <think>, base uses different tags — neither applies here).
_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)

# Two tool-call dialects the v3 model produces. The probe reports each
# independently so the gate decision is dialect-aware.
#
# 1) #4 dialect — what the #4 design ticket specified:
#       <tool_call>{"name": "...", "arguments": {...}}</tool_call>
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
#
# 2) v3 dialect — what the v3 model emits under a 'think first' system
#    message (observed live on DuoNeural-v3-4bit, see
#    evals/hermes_fc/PROBE_RESULTS.md):
#       <tool_call name="..." arguments='{...}'>   (single-quoted args,
#                                                    often truncated, often
#                                                    multiple per generation)
#    We capture the prefix (everything up to the first '>') so the report
#    can show how many call attempts the model made and how many completed.
_TOOL_CALL_V3_RE = re.compile(r"<tool_call\s+([^>]*?)\s*/?>", re.DOTALL)
# Unclosed v3-dialect tags (the model truncates mid-args and never emits
# the closing '>'). The "attempted" count surfaces these so the operator
# can see the model was *trying* to call a tool, even though the gate
# metric correctly does not credit an incomplete call.
_TOOL_CALL_V3_UNCLOSED_RE = re.compile(r"<tool_call\s+[^>]*$", re.DOTALL)
# Match either dialect to decide "emitted any tool call" in the row's main
# metric. The two specific regexes above give the dialect breakdown.
_TOOL_CALL_ANY_RE = re.compile(
    r"<tool_call>|<tool_call\s+[^>]*?>", re.DOTALL
)

# One-line description for ``argparse`` (avoids relying on ``__doc__`` being
# non-None at the point of parser construction).
_SHORT_DOC = "live <thought>-emission probe for the Hermes FC suite (ticket #12)."


def load_case(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def post_chat(base_url: str, payload: dict, timeout_s: float = 120.0) -> dict:
    """POST one chat-completions request to the served model. Stdlib only.

    Matches the contract ``mlx_lm server`` exposes (OpenAI-compatible
    ``/v1/chat/completions`` with ``tools=`` support).
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def analyze(content: str) -> dict:
    """One row of the report: did the model think first, and did the call parse?

    Reports the #4 dialect and the v3 dialect separately, plus the union.
    For the v3 dialect we cannot use ``json.loads`` on the inner content
    (the model uses Python-repr-style single quotes and frequently
    truncates mid-call); we only check that at least one v3-dialect call
    tag was opened.
    """
    thought_match = _THOUGHT_RE.search(content)
    call4_match = _TOOL_CALL_RE.search(content)
    call4_json = None
    call4_decodes = False
    if call4_match is not None:
        try:
            call4_json = json.loads(call4_match.group(1))
            call4_decodes = True
        except (ValueError, TypeError):
            pass
    call_v3_matches = list(_TOOL_CALL_V3_RE.finditer(content))
    call_v3_unclosed = list(_TOOL_CALL_V3_UNCLOSED_RE.finditer(content))
    any_call = bool(_TOOL_CALL_ANY_RE.search(content)) or bool(call_v3_unclosed)
    return {
        "emitted_thought": thought_match is not None,
        "thought_chars": len(thought_match.group(1)) if thought_match else 0,
        "emitted_call": any_call,
        "emitted_call_d4": call4_match is not None,
        "call_decodes": call4_decodes,
        "call_name": (call4_json or {}).get("name") if isinstance(call4_json, dict) else None,
        "emitted_call_v3": len(call_v3_matches) > 0,
        "call_v3_count": len(call_v3_matches),
        "emitted_call_v3_unclosed": len(call_v3_unclosed) > 0,
        "call_v3_unclosed_count": len(call_v3_unclosed),
    }


def build_payload(case: dict, model: str) -> dict:
    """Wrap a case's prompt + tools into the chat-completions request body."""
    messages = []
    if case.get("system"):
        messages.append({"role": "system", "content": case["system"]})
    messages.append({"role": "user", "content": case["prompt"]})
    return {
        "model": model,
        "messages": messages,
        "tools": case.get("tools") or [],
        "tool_choice": "auto",
        "temperature": 0.0,
        "n": 1,
        "stream": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=_SHORT_DOC)
    p.add_argument("--checkpoint", required=True,
                   help="Path or HF id of the model to serve (e.g. checkpoints/DuoNeural-v3-4bit).")
    p.add_argument("--case", required=True, help="Path to a single system2 case JSON.")
    p.add_argument("--n", type=int, default=10, help="Number of generations (default 10).")
    p.add_argument("--port", type=int, default=0, help="Server port (default 0 = pick a free one).")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    p.add_argument("--startup-timeout-s", type=float, default=300.0)
    p.add_argument("--min-pass-rate", type=float, default=1.0,
                   help=("Pass rate to consider the gate cleared (0.0-1.0, "
                         "default 1.0 = all generations must pass). "
                         "Independent of --n."))
    p.add_argument("--log-file", default=None,
                   help="Optional path to capture server stdout/stderr.")
    args = p.parse_args()

    case = load_case(args.case)
    if case.get("category") != "system2":
        print(f"WARNING: case {case.get('id')!r} category is {case.get('category')!r}, not system2; "
              f"this probe is only meaningful for system2 cases.", file=sys.stderr)
    if not case.get("require_thought"):
        print(f"WARNING: case {case.get('id')!r} does not have require_thought=true; "
              f"<thought> emission is not a gate for this case.", file=sys.stderr)

    log_fh = open(args.log_file, "wb") if args.log_file else None
    server = ServerManager(
        args.checkpoint,
        host=args.host,
        port=args.port or None,
        startup_timeout_s=args.startup_timeout_s,
        log_file=log_fh,
    )
    try:
        base_url = server.start()
        print(f"server up at {base_url} (model_id={server.model_id})", file=sys.stderr)

        results = []
        for i in range(args.n):
            t0 = time.monotonic()
            payload = build_payload(case, DEFAULT_MODEL_KEY)
            try:
                resp = post_chat(base_url, payload)
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, socket.timeout) as exc:
                results.append({"i": i, "error": repr(exc)})
                print(f"[{i+1:>2}/{args.n}] ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
                continue
            elapsed = time.monotonic() - t0

            # OpenAI chat-completions response shape: choices[0].message.content
            # (mlx_lm server exposes this verbatim).
            try:
                content = resp["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as exc:
                results.append({"i": i, "error": f"malformed response: {exc}", "raw": resp})
                print(f"[{i+1:>2}/{args.n}] MALFORMED: {exc}", file=sys.stderr)
                continue

            row = analyze(content)
            row.update({"i": i, "elapsed_s": round(elapsed, 2), "content": content})
            results.append(row)
            # Gate condition: thought + a parseable call. We require the
            # #4 dialect because that's what the adapter (ticket #19) is
            # currently specified against; the v3 dialect is reported as
            # an extra metric but does not, by itself, satisfy the gate.
            tag = "PASS" if (row["emitted_thought"] and row["call_decodes"]) else "FAIL"
            print(f"[{i+1:>2}/{args.n}] {tag}  thought={row['emitted_thought']!s:<5} "
                  f"thought_chars={row['thought_chars']:>4}  "
                  f"call(d4)={row['emitted_call_d4']!s:<5}  call(d4_decodes)={row['call_decodes']!s:<5}  "
                  f"call(v3)={row['emitted_call_v3']!s:<5} v3_n={row['call_v3_count']}  "
                  f"call(v3_un)={row['emitted_call_v3_unclosed']!s:<5} v3_un_n={row['call_v3_unclosed_count']}  "
                  f"name={row['call_name']!r}  t={row['elapsed_s']}s", file=sys.stderr)

    finally:
        server.stop()
        if log_fh is not None:
            log_fh.close()

    # Aggregate.
    n_total = len(results)
    n_thought = sum(1 for r in results if r.get("emitted_thought"))
    n_call_d4 = sum(1 for r in results if r.get("emitted_call_d4"))
    n_call_v3 = sum(1 for r in results if r.get("emitted_call_v3"))
    n_call_v3_unclosed = sum(1 for r in results if r.get("emitted_call_v3_unclosed"))
    n_call_any = sum(1 for r in results if r.get("emitted_call"))
    n_decodes = sum(1 for r in results if r.get("call_decodes"))
    n_clean = sum(1 for r in results
                  if r.get("emitted_thought") and r.get("call_decodes"))
    n_err = sum(1 for r in results if r.get("error"))
    pass_rate = (n_clean / n_total) if n_total else 0.0

    print()
    print("=" * 72)
    print(f"PROBE REPORT — {args.checkpoint} × {args.n} generations on {args.case}")
    print("=" * 72)
    print(f"  emitted <thought>             : {n_thought}/{n_total}  ({n_thought / max(n_total, 1):.0%})")
    print(f"  emitted <tool_call> (#4)      : {n_call_d4}/{n_total}  ({n_call_d4 / max(n_total, 1):.0%})")
    print(f"  emitted <tool_call> (v3, closed)    : {n_call_v3}/{n_total}  ({n_call_v3 / max(n_total, 1):.0%})")
    print(f"  emitted <tool_call> (v3, unclosed)  : {n_call_v3_unclosed}/{n_total}  ({n_call_v3_unclosed / max(n_total, 1):.0%})")
    print(f"  emitted any call (union)      : {n_call_any}/{n_total}  ({n_call_any / max(n_total, 1):.0%})")
    print(f"  call JSON parses (#4 dialect) : {n_decodes}/{n_total}  ({n_decodes / max(n_total, 1):.0%})")
    print(f"  full pass (thought + #4 dec.) : {n_clean}/{n_total}  ({pass_rate:.0%})")
    print(f"  errors                        : {n_err}/{n_total}")
    print(f"  gate threshold                : {args.min_pass_rate:.0%}")
    print("=" * 72)
    if n_total == 0:
        print("GATE: no generations completed (all errored); see stderr.", file=sys.stderr)
        return 2
    gate_cleared = pass_rate >= args.min_pass_rate and n_err == 0
    print(f"GATE: {'CLEARED' if gate_cleared else 'NOT CLEARED'} — "
          f"{'safe to author system2-002..005' if gate_cleared else 'do not scale system2; investigate'}",
          file=sys.stderr)
    return 0 if gate_cleared else 1


if __name__ == "__main__":
    sys.exit(main())
