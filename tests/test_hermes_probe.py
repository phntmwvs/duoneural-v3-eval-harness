"""Tests for the ``<thought>``-emission probe (ticket #12).

Pure-function tests only — we exercise ``analyze()`` and ``build_payload()``
without ever starting a real ``mlx_lm server`` (real serving is MBP-only).
The end-to-end probe is a one-shot CLI run on the M4 Pro, not a unit test.

Stdlib ``unittest`` (also pytest-compatible); needs no third-party deps
so it runs in any venv and under 3.9.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct import of the probe module (it's a CLI script but exposes pure
# helpers). We don't trigger its argparse/main by importing it — it has
# no top-level side effects.
import importlib.util
_PROBE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, "evals", "hermes_fc", "probe_thought_emission.py",
)
_spec = importlib.util.spec_from_file_location("hermes_probe", _PROBE)
assert _spec is not None, f"failed to load probe spec from {_PROBE}"
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)  # type: ignore[union-attr]  # spec is non-None after the assert above


class TestAnalyze(unittest.TestCase):
    """Cover the four pass/fail branches the probe will report."""

    def test_pass(self):
        content = (
            "<thought>14:00 UTC -> NY is 4 hours behind, so 10:00 ET.</thought>"
            '<tool_call>{"name": "schedule_meeting", "arguments": '
            '{"title": "meeting with Bob", "attendees": ["bob@example.com"], '
            '"date": "2026-11-04", "time": "10:00", '
            '"timezone": "America/New_York", "duration_minutes": 30}}'
            "</tool_call>"
        )
        r = probe.analyze(content)
        self.assertTrue(r["emitted_thought"])
        self.assertEqual(r["thought_chars"], len("14:00 UTC -> NY is 4 hours behind, so 10:00 ET."))
        self.assertTrue(r["emitted_call"])
        self.assertTrue(r["call_decodes"])
        self.assertEqual(r["call_name"], "schedule_meeting")

    def test_no_thought(self):
        content = (
            '<tool_call>{"name": "schedule_meeting", "arguments": {}}'
            "</tool_call>"
        )
        r = probe.analyze(content)
        self.assertFalse(r["emitted_thought"])
        self.assertEqual(r["thought_chars"], 0)
        self.assertTrue(r["emitted_call"])
        self.assertTrue(r["call_decodes"])

    def test_malformed_call(self):
        content = "<thought>thinking</thought><tool_call>{not json}</tool_call>"
        r = probe.analyze(content)
        self.assertTrue(r["emitted_thought"])
        self.assertTrue(r["emitted_call"])
        self.assertFalse(r["call_decodes"])
        self.assertIsNone(r["call_name"])

    def test_no_call(self):
        content = "<thought>I'll just answer in prose.</thought>"
        r = probe.analyze(content)
        self.assertTrue(r["emitted_thought"])
        self.assertFalse(r["emitted_call"])
        self.assertFalse(r["call_decodes"])
        self.assertIsNone(r["call_name"])

    def test_empty_content(self):
        r = probe.analyze("")
        for v in r.values():
            self.assertIn(v, (False, 0, None))

    def test_thought_with_newlines(self):
        # <thought> blocks span newlines (DOTALL); make sure we count them
        # in the length and don't truncate on the first \n.
        content = "<thought>line one\nline two\nline three</thought>"
        r = probe.analyze(content)
        self.assertTrue(r["emitted_thought"])
        self.assertEqual(r["thought_chars"], len("line one\nline two\nline three"))


class TestBuildPayload(unittest.TestCase):
    """The request body sent to ``mlx_lm server`` must conform to the
    OpenAI chat-completions contract: ``model``, ``messages``, ``tools``,
    ``temperature``, ``n``, ``stream``."""

    def test_minimal_case(self):
        case = {
            "id": "fc-x", "category": "single",
            "prompt": "Read /etc/hostname.",
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        }
        p = probe.build_payload(case, "default_model")
        self.assertEqual(p["model"], "default_model")
        self.assertEqual(p["messages"], [{"role": "user", "content": "Read /etc/hostname."}])
        self.assertEqual(p["tools"], case["tools"])
        self.assertEqual(p["temperature"], 0.0)
        self.assertEqual(p["n"], 1)
        self.assertFalse(p["stream"])

    def test_system_override(self):
        case = {
            "id": "fc-x", "category": "system2",
            "prompt": "Think first.",
            "system": "Use <thought> before calling any tool.",
            "tools": [],
        }
        p = probe.build_payload(case, "default_model")
        self.assertEqual(p["messages"], [
            {"role": "system", "content": "Use <thought> before calling any tool."},
            {"role": "user", "content": "Think first."},
        ])

    def test_no_system_field(self):
        case = {"id": "fc-x", "category": "single", "prompt": "Hi.", "tools": []}
        p = probe.build_payload(case, "default_model")
        # ``system`` absent (or None) should produce a user-only message list.
        self.assertEqual(p["messages"], [{"role": "user", "content": "Hi."}])

    def test_missing_tools(self):
        case = {"id": "fc-x", "category": "single", "prompt": "Hi."}
        p = probe.build_payload(case, "default_model")
        self.assertEqual(p["tools"], [])


if __name__ == "__main__":
    unittest.main()
