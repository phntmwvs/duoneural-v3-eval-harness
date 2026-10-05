"""Register DuoNeural v3 MLX in BFCL's model maps — no fork edit (tickets #3/#17).

Importing :func:`register` inserts a ``ModelConfig`` into
``MODEL_CONFIG_MAPPING`` and appends the key to ``SUPPORTED_MODELS``, so
``bfcl generate --model duoneural-v3-mlx-fc`` resolves against the live
``mlx_lm server`` pointed to by ``REMOTE_OPENAI_BASE_URL``.

Runtime monkey-patch only — BFCL's source tree is never edited. The model key
uses the ``-FC`` suffix (function-calling mode, ``is_fc_model=True``) matching
BFCL's convention for FC variants. Importable only inside ``.venv-bfcl``
(imports ``bfcl_eval``).
"""

from __future__ import annotations

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.constants.supported_models import SUPPORTED_MODELS

from .handler import DuoNeuralV3FCHandler

MODEL_KEY = "duoneural-v3-mlx-fc"


def register(model_key: str = MODEL_KEY) -> str:
    """Idempotently register the DuoNeural v3 FC config; return the model key."""
    if model_key not in MODEL_CONFIG_MAPPING:
        MODEL_CONFIG_MAPPING[model_key] = ModelConfig(
            model_name=model_key,
            display_name="DuoNeural v3 MLX (FC)",
            url="https://huggingface.co/DuoNeural/LFM2.5-8B-A1B-Hermes-Agentic-Coder-Abliterated-v3",
            org="DuoNeural",
            license="lfm-open-license-v1.0",
            model_handler=DuoNeuralV3FCHandler,
            input_price=None,
            output_price=None,
            is_fc_model=True,
            underscore_to_dot=False,
        )
    if model_key not in SUPPORTED_MODELS:
        SUPPORTED_MODELS.append(model_key)
    return model_key
