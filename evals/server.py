"""Shared ``mlx_lm server`` lifecycle manager (wayfinder ticket #16).

One ``ServerManager`` owns one ``mlx_lm server`` process for one matrix row.
The per-component adapters never own serving: the matrix (``run_matrix.py``,
ticket #20) brings a row's server up once, hands each adapter a live
``base_url``, and tears the server down after all components have run.

Contract (per ticket #6, resolutions 1a/2a/3a):

* spawn ``mlx_lm server --model <ref> --host <h> --port <p>`` on a free port,
* readiness-poll ``GET /v1/models`` until the model answers (or timeout),
* on teardown/crash: terminate the process, escalating to kill if it does not
  exit cleanly, and always reap it.

Only the Python standard library is used — this module must import and run in
every component venv (``.venv-core`` / ``.venv-bfcl`` / ``.venv-evalplus``)
and under both the 3.12 target and the 3.9 build host.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

# Bound 127.0.0.1: all serving is loopback on the MBP (tickets #6 / SETUP.md).
DEFAULT_HOST = "127.0.0.1"
# BF16 ~17 GB weights must load + warm before /v1/models answers.
DEFAULT_STARTUP_TIMEOUT_S = 300.0
DEFAULT_READINESS_INTERVAL_S = 0.5
DEFAULT_STOP_TIMEOUT_S = 15.0


def find_free_port(host: str = DEFAULT_HOST) -> int:
    """Return an ephemeral port free at call time on ``host``.

    ``mlx_lm server`` binds its port only after the model has loaded (any
    pre-empted socket would make the server abort on start), so the port must
    be released before spawn. The readiness poll tolerates the residual race:
    a hijacker occupying the port simply never answers as our model.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def resolve_server_command(checkpoint: str, host: str, port: int,
                           python: str | None = None) -> list[str]:
    """Build the ``mlx_lm server`` argv for ``checkpoint`` on ``host``:``port``.

    ``python`` selects the interpreter/venv that provides mlx-lm; the default
    is the current interpreter (``sys.executable``). Resolution order:

    1. ``<python_dir>/mlx_lm`` if that console script exists (venv install), or
    2. an ``mlx_lm`` already on ``PATH``, or
    3. ``<python> -m mlx_lm server`` as a fallback.
    """
    if python is None:
        import sys
        python = sys.executable

    argv = ["--model", checkpoint, "--host", host, "--port", str(port)]
    script = os.path.join(os.path.dirname(python), "mlx_lm")
    if os.path.exists(script):
        return [script, "server"] + argv
    on_path = shutil.which("mlx_lm")
    if on_path:
        return [on_path, "server"] + argv
    return [python, "-m", "mlx_lm", "server"] + argv


class ServerManager:
    """Own one ``mlx_lm server`` child process from spawn to reap."""

    def __init__(
        self,
        checkpoint: str,
        *,
        host: str = DEFAULT_HOST,
        port: int | None = None,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        readiness_interval_s: float = DEFAULT_READINESS_INTERVAL_S,
        stop_timeout_s: float = DEFAULT_STOP_TIMEOUT_S,
        python: str | None = None,
        command: list[str] | None = None,
        env: dict | None = None,
        log_file=None,
    ) -> None:
        self.checkpoint = checkpoint
        self.host = host
        self.port = port if port is not None else find_free_port(host)
        self.startup_timeout_s = startup_timeout_s
        self.readiness_interval_s = readiness_interval_s
        self.stop_timeout_s = stop_timeout_s
        # Tests pass a fake ``command`` (a stub HTTP server); production leaves
        # it None and the real mlx_lm argv is resolved from ``python``.
        self.command = command if command is not None else resolve_server_command(
            checkpoint, host, self.port, python
        )
        self.env = env  # None -> inherit os.environ at spawn
        self._log_file = log_file
        self.process: subprocess.Popen | None = None
        self.model_id: str | None = None

    @property
    def base_url(self) -> str:
        """OpenAI base URL handed to adapters (``REMOTE_OPENAI_BASE_URL``)."""
        return "http://{0}:{1}/v1".format(self.host, self.port)

    def start(self) -> str:
        """Spawn the server and block until it answers ``/v1/models``.

        Returns ``base_url``. Raises ``TimeoutError`` if the model is not ready
        within ``startup_timeout_s`` (reaping the child and surfacing any early
        exit code / stderr tail), or ``RuntimeError`` if it failed to spawn.
        """
        if self.process is not None:
            raise RuntimeError("ServerManager.start() called twice")
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=self._log_file if self._log_file is not None else subprocess.DEVNULL,
                stderr=subprocess.STDOUT if self._log_file is not None else subprocess.DEVNULL,
                start_new_session=True,  # own pgid: a kill can't take down the harness
                env=self.env,
            )
        except OSError as exc:
            raise RuntimeError(
                "failed to spawn mlx_lm server: {0} (argv: {1})".format(exc, self.command)
            ) from exc
        try:
            self.model_id = self._wait_ready()
        except BaseException:
            self.stop()
            raise
        return self.base_url

    def _wait_ready(self) -> str:
        """Poll ``/v1/models`` until the served model id answers."""
        url = self.base_url + "/models"
        deadline = time.monotonic() + self.startup_timeout_s
        while True:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    "mlx_lm server exited during startup (code {0}) for {1}".format(
                        self.process.returncode, self.checkpoint
                    )
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                data = payload.get("data") or []
                if data and data[0].get("id"):
                    return str(data[0]["id"])
            except (urllib.error.URLError, ValueError, KeyError, IndexError, AttributeError, OSError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "mlx_lm server not ready within {0:.0f}s at {1} (checkpoint: {2})".format(
                        self.startup_timeout_s, url, self.checkpoint
                    )
                )
            time.sleep(self.readiness_interval_s)

    def stop(self) -> None:
        """Terminate the server, escalating to kill, and always reap it."""
        proc, self.process = self.process, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=self.stop_timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=self.stop_timeout_s)
                except subprocess.TimeoutExpired:
                    pass
        else:
            proc.wait()  # already dead: reap the zombie

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def __enter__(self) -> "ServerManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
