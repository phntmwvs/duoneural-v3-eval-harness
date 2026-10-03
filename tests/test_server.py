"""Tests for the shared ServerManager (ticket #16).

Runs against ``tests/fake_mlx_lm_server.py`` (a stub answering
``GET /v1/models``) so the lifecycle is proven without MLX — real serving is
MBP-only. Stdlib ``unittest`` (also pytest-compatible); needs no third-party
deps so it runs in any venv and under 3.9.
"""

from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals import server as srv  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_FAKE = os.path.join(_HERE, "fake_mlx_lm_server.py")


def make_manager(startup_timeout_s=15.0, readiness_interval_s=0.1, env_extra=None, **kw):
    """A ServerManager pointed at the fake server (free port, fake command+env)."""
    port = srv.find_free_port()
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    command = [sys.executable, _FAKE, srv.DEFAULT_HOST, str(port)]
    return srv.ServerManager(
        "ckpt",
        port=port,
        command=command,
        env=env,
        startup_timeout_s=startup_timeout_s,
        readiness_interval_s=readiness_interval_s,
        **kw,
    )


class FindFreePortTest(unittest.TestCase):
    def test_returns_bindable_port(self):
        import socket

        port = srv.find_free_port()
        self.assertIsInstance(port, int)
        self.assertTrue(1024 <= port <= 65535)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((srv.DEFAULT_HOST, port))  # free right after the call


class ResolveCommandTest(unittest.TestCase):
    def test_module_fallback_when_no_script(self):
        cmd = srv.resolve_server_command("ckpt", "127.0.0.1", 9000, python=sys.executable)
        for token in ("--model", "ckpt", "--host", "127.0.0.1", "--port", "9000", "server"):
            self.assertIn(token, cmd)
        # no venv mlx_lm next to the test interpreter and none on PATH -> -m form
        self.assertEqual(cmd[:3], [sys.executable, "-m", "mlx_lm"])


class LifecycleTest(unittest.TestCase):
    def test_start_ready_stop(self):
        mgr = make_manager()
        base_url = mgr.start()
        proc = mgr.process
        self.assertIsNotNone(proc)
        try:
            self.assertEqual(base_url, mgr.base_url)
            self.assertTrue(base_url.startswith("http://127.0.0.1:"))
            self.assertTrue(base_url.endswith("/v1"))
            self.assertTrue(mgr.is_running())
            self.assertEqual(mgr.model_id, "fake-model")
        finally:
            mgr.stop()
        self.assertFalse(mgr.is_running())
        assert proc is not None
        self.assertIsNotNone(proc.poll())  # reaped, not a zombie

    def test_context_manager(self):
        with make_manager() as mgr:
            self.assertTrue(mgr.is_running())
            pid = mgr.process.pid
        self.assertFalse(mgr.is_running())
        with self.assertRaises(OSError):  # child pid is gone
            os.kill(pid, 0)

    def test_crash_before_ready_raises_and_cleans(self):
        mgr = make_manager(env_extra={"FAKE_DIE_AFTER": "3"})
        with self.assertRaises(RuntimeError):
            mgr.start()
        self.assertFalse(mgr.is_running())

    def test_timeout_raises_and_cleans(self):
        port = srv.find_free_port()
        mgr = srv.ServerManager(
            "ckpt",
            port=port,
            command=[sys.executable, "-c", "import time; time.sleep(5)"],  # never serves
            startup_timeout_s=1.0,
            readiness_interval_s=0.1,
        )
        with self.assertRaises(TimeoutError):
            mgr.start()
        self.assertFalse(mgr.is_running())

    def test_stop_idempotent(self):
        mgr = make_manager()
        mgr.start()
        mgr.stop()
        mgr.stop()  # second call must not raise
        self.assertFalse(mgr.is_running())

    def test_bind_delay_within_timeout(self):
        mgr = make_manager(startup_timeout_s=20.0, env_extra={"FAKE_BIND_DELAY": "1.0"})
        t0 = time.monotonic()
        mgr.start()
        try:
            self.assertGreaterEqual(time.monotonic() - t0, 1.0)
            self.assertTrue(mgr.is_running())
        finally:
            mgr.stop()

    def test_custom_model_id_propagates(self):
        mgr = make_manager(env_extra={"FAKE_MODEL_ID": "DuoNeural-v3-4bit"})
        mgr.start()
        try:
            self.assertEqual(mgr.model_id, "DuoNeural-v3-4bit")
        finally:
            mgr.stop()


if __name__ == "__main__":
    unittest.main()
