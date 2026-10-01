"""BFCL OSSHandler subclass for DuoNeural v3 MLX (LFM2.5-8B-A1B Hermes).

PROTOTYPE (wayfinder ticket #3) — throwaway. This is the seam the spike tests:
can a thin OSSHandler subclass serve DuoNeural v3 over ``mlx_lm serve`` and
complete one BFCL multi-turn conversation end-to-end
(prompt → ``<tool_call>`` decode → tool-result re-render)?

Modeled on ``bfcl_eval.model_handler.local_inference.qwen_fc.QwenFCHandler``:
same ``<|im_start|>``/``<|im_end|>`` role tokens, same ``<tool_call>`` /
``<tool_response>`` XML. The one structural delta vs QwenFC is the model's
``<thought>`` reasoning, which is stripped before ``<tool_call>`` decode
(QwenFC strips ``<think>``; DuoNeural v3 emits ``<thought>`` — see the
conversion record in duoneural-v3-mlx/docs/conversion).

The completions-API path and remote-endpoint plumbing are inherited unchanged
from ``OSSHandler`` (``client.completions.create`` against
``REMOTE_OPENAI_BASE_URL``; tokenizer from ``REMOTE_OPENAI_TOKENIZER_PATH``).
"""

from __future__ import annotations

import json
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
        """The model id the remote server actually has loaded.

        Root cause of the live 404 (ticket #3): OSSHandler._query_prompting sends
        ``client.completions.create(model=self.model_path_or_id)``. For a remote
        endpoint ``model_path_or_id`` falls back to ``model_name_huggingface``
        (the registry key, ``duoneural-v3-mlx-fc``). mlx_lm's server treats any
        ``model`` value that is NOT the served id as a model to load — and the
        registry key isn't a local path, so it tries to pull it from HF and the
        completion 404s with "Repository Not Found .../models/duoneural-v3-mlx-fc".

        Fix: read the served id from ``GET {base_url}/models`` once and send that.
        """
        if self._resolved_server_model is None:
            try:
                import requests

                resp = requests.get(f"{self.base_url}/models", timeout=10)
                self._resolved_server_model = resp.json()["data"][0]["id"]
                print(f"[spike] served model id from /v1/models: "
                      f"{self._resolved_server_model!r}")
            except Exception as e:  # pragma: no cover - spike diagnostics
                print(f"[spike] WARNING: /v1/models lookup failed ({e}); "
                      f"falling back to model_path_or_id={self.model_path_or_id!r}")
                self._resolved_server_model = self.model_path_or_id
        return self._resolved_server_model

    @override
    def _query_prompting(self, inference_data: dict):
        # Ensure the completion names the served model, then run the stock path.
        self.model_path_or_id = self._server_model_id()
        return super()._query_prompting(inference_data)

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        result = _strip_thought(result)
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        result = _strip_thought(result)
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

    @override
    def _format_prompt(self, messages, function):
        """Qwen FC template, but reasoning renders as <thought> (not <think>).

        The inherited QwenFCHandler._format_prompt re-renders the assistant's
        stored ``reasoning_content`` inside ``<think>...</think>``. DuoNeural v3
        was trained on Hermes ``<thought>``; feeding it ``<think>`` on later
        turns is a distribution mismatch. Qwen's rendering is otherwise
        identical (same <|im_start|>/<|im_end|> roles, same <tool_call> /
        <tool_response> XML), so we reuse it and swap the reasoning tag.
        """
        rendered = super()._format_prompt(messages, function)
        return rendered.replace("<think>", "<thought>").replace("</think>", "</thought>")

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text
        extracted_tool_calls = self._extract_tool_calls(model_response)

        thought_content = ""
        cleaned_response = model_response
        if "</thought>" in model_response:
            parts = model_response.split("</thought>")
            thought_content = (
                parts[0].rstrip("\n").split("<thought>")[-1].lstrip("\n")
            )
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
