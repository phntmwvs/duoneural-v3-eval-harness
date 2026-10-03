"""BFCL handler for DuoNeural v3 MLX (LFM2.5-8B-A1B Hermes) — ticket #17.

Folded forward from the #3 spike (``spike/bfcl-osshandler``, PR #14), which
validated this seam live: a thin ``QwenFCHandler`` subclass serves DuoNeural v3
over ``mlx_lm server`` and completes a BFCL multi-turn conversation end-to-end
(prompt → ``<tool_call>`` decode → tool-result re-render). Carries both #3
hardening notes:

1. **Served-id resolution from ``/v1/models``** — the registry key
   (``duoneural-v3-mlx-fc``) is not a loadable HF id; sending it as the
   completion's ``model`` makes mlx_lm try to pull it and 404. We read the
   served id once from ``GET {base_url}/models`` and send that.
2. **``<thought>`` (not ``<think>``) reasoning** — DuoNeural v3 was trained on
   Hermes ``<thought>``; the inherited Qwen template re-renders stored
   reasoning as ``<think>``, a distribution mismatch on later turns. We strip
   ``<thought>`` before ``<tool_call>`` decode and render reasoning as
   ``<thought>``.

Importable only inside ``.venv-bfcl`` (imports ``bfcl_eval``).
"""

from __future__ import annotations

import re
from typing import Any

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override

# DuoNeural v3 emits reasoning in <thought>...</thought> (not <think>).
_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL)


def _strip_thought(text: str) -> str:
    """Remove <thought>...</thought> blocks so <tool_call> JSON is isolated."""
    return _THOUGHT_RE.sub("", text)


class DuoNeuralV3FCHandler(QwenFCHandler):
    """QwenFC-style handler with the DuoNeural v3 (<thought>) decode path."""

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(
            model_name, temperature, registry_name, is_fc_model, dtype=dtype, **kwargs
        )

    # NOTE: OSSHandler marks spin_up_local_server @final, and its metaclass
    # (EnforceOverrides) forbids subclassing it at all — so we cannot correct the
    # `model` field there. Instead we fix it in _query_prompting (not final),
    # lazily, the first time a query is sent (server is guaranteed up by then).
    _resolved_server_model = None

    def _server_model_id(self) -> str:
        """The model id the remote server actually has loaded (#3 hardening 1).

        Root cause of the spike's live 404: ``OSSHandler._query_prompting`` sends
        ``client.completions.create(model=self.model_path_or_id)``. For a remote
        endpoint ``model_path_or_id`` falls back to the registry key
        (``duoneural-v3-mlx-fc``), which mlx_lm treats as an HF id to pull — and
        it 404s. Read the served id from ``GET {base_url}/models`` and send
        that. Retry briefly: a freshly-spawned server can drop the first
        connection while it warms up (observed live — a one-shot lookup that
        failed here sank the run with the 404 above).
        """
        if self._resolved_server_model is None:
            import time

            import requests
            last_err = None
            for _attempt in range(10):
                try:
                    resp = requests.get("{0}/models".format(self.base_url), timeout=10)
                    resp.raise_for_status()  # fail fast on non-200 (M3)
                    self._resolved_server_model = resp.json()["data"][0]["id"]
                    break
                except Exception as e:  # connection reset / not-ready / parse
                    last_err = e
                    time.sleep(1.0)
            if self._resolved_server_model is None:
                raise RuntimeError(
                    "[bfcl] could not resolve the served model id from "
                    "{0}/models after retries ({1}); refusing "
                    "to send the registry key, which mlx_lm would 404 as an HF id.".format(
                        self.base_url, last_err
                    )
                )
        return self._resolved_server_model

    @override
    def _query_prompting(self, inference_data: dict):
        # Name the served model in the completion, then run the stock path.
        self.model_path_or_id = self._server_model_id()
        return super()._query_prompting(inference_data)

    def _tool_calls_or_raise(self, result):
        """Strip <thought> and extract tool calls, raising on a non-list.

        Shared by decode_ast / decode_execute: same strip → extract → guard
        (the guard idiom matches the upstream QwenFCHandler).
        """
        result = _strip_thought(result)
        tool_calls = self._extract_tool_calls(result)
        if not isinstance(tool_calls, list) or \
                any(not isinstance(item, dict) for item in tool_calls):
            raise ValueError(
                "Model did not return a list of function calls: {0}".format(result)
            )
        return tool_calls

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        tool_calls = self._tool_calls_or_raise(result)
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        # _tool_calls_or_raise guarantees every element is a dict, so the
        # literal-string branch the upstream QwenFCHandler carries is dead here
        # (M4) — build the execute form straight from the dicts.
        tool_calls = self._tool_calls_or_raise(result)
        decoded_result = [{item["name"]: item["arguments"]} for item in tool_calls]
        return convert_to_function_call(decoded_result)

    @override
    def _format_prompt(self, messages, function):
        """Qwen FC template, but reasoning renders as <thought> (#3 hardening 2).

        The inherited QwenFCHandler._format_prompt re-renders the assistant's
        stored ``reasoning_content`` inside ``<think>...</think>``. DuoNeural v3
        was trained on Hermes ``<thought>``; feeding it ``<think>`` on later
        turns is a distribution mismatch.

        The swap is scoped to the reasoning wrapper, not a blunt global
        replace (M5): upstream's own template emits the wrapper only at a
        message boundary as ``<think>\n<reasoning>\n</think>\n\n`` (qwen_fc.py
        line 95), so ``<think>\n`` / ``</think>\n`` are unambiguous — a literal
        ``<think>`` inside user/model content does not match and is left alone.
        """
        rendered = super()._format_prompt(messages, function)
        return (rendered
                .replace("<think>\n", "<thought>\n")
                .replace("</think>\n", "</thought>\n"))

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text
        extracted_tool_calls = self._extract_tool_calls(model_response)

        thought_content = ""
        cleaned_response = model_response
        if "</thought>" in model_response:
            parts = model_response.split("</thought>")
            thought_content = parts[0].rstrip("\n").split("<thought>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        if len(extracted_tool_calls) > 0:
            chat_history_message = {
                "role": "assistant",
                "content": "",
                "tool_calls": extracted_tool_calls,
            }
        else:
            chat_history_message = {
                "role": "assistant",
                "content": cleaned_response,
            }

        chat_history_message["reasoning_content"] = thought_content

        return {
            "model_responses": cleaned_response,
            "reasoning_content": thought_content,
            "model_responses_message_for_chat_history": chat_history_message,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }
