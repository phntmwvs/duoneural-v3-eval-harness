"""A fake ``mlx_lm server`` for testing ``ServerManager`` without MLX.

Serves just enough of the OpenAI surface — ``GET /v1/models`` answering with a
configurable served model id — for the readiness poll. Real serving is an
MBP-only concern (the mini has no mlx-lm); this stub proves the lifecycle
(spawn / readiness / teardown / crash-cleanup) deterministically and fast.

Env knobs:
    FAKE_MODEL_ID   served id returned by /v1/models (default "fake-model")
    FAKE_BIND_DELAY seconds to sleep before binding (simulates model load)
    FAKE_DIE_AFTER  if set, exit immediately with this code instead of serving
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class _FakeServer(HTTPServer):
    model_id: str = "fake-model"

    def server_bind(self):
        # HTTPServer.server_bind calls socket.getfqdn() (reverse DNS), which can
        # block for tens of seconds in a sandboxed/loopback-only environment.
        # Bind directly; the served model id doesn't depend on the host name.
        import socket as _socket

        _socket.socket.bind(self.socket, self.server_address)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


class _Handler(BaseHTTPRequestHandler):
    server: "_FakeServer"  # type: ignore[assignment]  (http.server sets it post-init)

    def do_GET(self):  # noqa: N802 (http.server API)
        if self.path.startswith("/v1/models"):
            body = json.dumps(
                {"object": "list", "data": [{"id": self.server.model_id, "object": "model"}]}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):  # keep test output quiet  # noqa: A002
        pass


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

    die_after = os.environ.get("FAKE_DIE_AFTER")
    if die_after is not None:
        return int(die_after)

    delay = float(os.environ.get("FAKE_BIND_DELAY", "0"))
    if delay:
        time.sleep(delay)

    server = _FakeServer((host, port), _Handler)
    server.model_id = os.environ.get("FAKE_MODEL_ID", "fake-model")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
