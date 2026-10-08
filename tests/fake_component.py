"""A fake component adapter for ``run_matrix`` tests (ticket #20).

The matrix is orchestration code — the real per-component adapters
(``evals/components/bfcl``, ``evals/components/evalplus``,
``evals/components/hermes``) need a per-venv python that this build host
does not have. Tests for the matrix stand in a tiny adapter that:

* accepts the same ``--model`` / ``--model-name`` / ``--base-url`` /
  ``--resume`` flags the real adapters take,
* when ``--base-url`` is ``http://INVALID_NO_S`` (a sentinel), writes a
  contract-valid result JSON with ``score=0.42`` and a fixed
  ``subscores.complete=True`` — proves the matrix called us and wrote a
  result, no server required.
* when ``--base-url`` is a real URL, performs a minimal reachability
  check (``GET /v1/models``) so the test can prove the matrix passed the
  *real* live URL through to the adapter, and that the readiness-poll
  contract (a live server answers) is honored end-to-end.

This is a **fake** (a test double), not a stub of the real contract. It
*records* its argv into a JSON sidecar under ``$TMPDIR/fake-component.log``
so a test can assert on what the matrix actually called.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

# Sentinel: with this --base-url the fake writes its result immediately,
# no network. Tests that don't care about the live server use this.
SENTINEL_BASE_URL = "http://INVALID_NO_S"

# A real-looking URL (loopback) but the matrix will hand whatever the
# ServerManager gave it — this is just for the path the fake takes when
# it ISN'T the sentinel.

_LOG_PATH = os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "fake_component_calls.jsonl"
)


def _log_call(record: dict) -> None:
    """Append a JSONL record of how the fake was called (test inspection)."""
    with open(_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _server_reachable(base_url: str, timeout_s: float = 2.0) -> bool:
    """Return True if ``GET {base_url}/models`` answers within ``timeout_s``."""
    try:
        with urllib.request.urlopen(base_url + "/models", timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, socket.timeout, ValueError):
        return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fake component adapter (test only)")
    p.add_argument("--model", required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--base-url", required=True)
    p.add_argument("--resume", action="store_true")
    # Optional knobs the matrix passes for ops (see the matrix's --fail-fast
    # / --startup-timeout plumbing). The fake honors them in argv only —
    # it does not actually use them, since it has no server to control.
    p.add_argument("--startup-timeout-s", type=float, default=None)
    p.add_argument("--num-threads", type=int, default=None)
    p.add_argument("--categories", default=None)
    args = p.parse_args(argv)

    model_name = args.model_name if args.model_name is not None else args.model
    now = time.time()

    reachable = None
    if args.base_url != SENTINEL_BASE_URL:
        reachable = _server_reachable(args.base_url)

    _log_call({
        "argv": sys.argv[1:],
        "base_url": args.base_url,
        "model": args.model,
        "model_name": model_name,
        "resume": bool(args.resume),
        "server_reachable": reachable,
        "ts": now,
    })

    # In the sentinel path the fake always "completes" the cell — no
    # network, no server, just a result. In the real-URL path we
    # complete only if the matrix actually handed us a live URL.
    component_name = os.environ.get("FAKE_COMPONENT_NAME", "fake")
    if args.base_url == SENTINEL_BASE_URL or reachable:
        result = {
            "component": component_name,
            "model": model_name,
            "checkpoint": args.model,
            "score": 0.42,
            "subscores": {
                "complete": True,
                "fake": True,
                "resume": bool(args.resume),
            },
            "runtime_s": 0.01,
            "artifact_versions": {"fake": "yes", "python": sys.version.split()[0]},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        out = os.environ.get("FAKE_RESULT_PATH")
        if out is None:
            print("[fake] ERROR: FAKE_RESULT_PATH not set", file=sys.stderr)
            return 3
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return 0

    # Real URL but no server — fail loudly. The matrix should not hand us
    # a non-sentinel URL it hasn't stood up.
    print(
        "[fake] ERROR: base_url={0} did not answer /v1/models".format(args.base_url),
        file=sys.stderr,
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
