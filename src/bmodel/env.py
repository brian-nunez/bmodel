from .base import ModelConfig
from .registry import (
    MODEL_GEMMA4_URL,
    MODEL_GEMMA4_API_KEY,
    MODEL_GEMMA4_MODEL_ID,
    MODEL_TRANSLATEGEMMA_URL,
    MODEL_TRANSLATEGEMMA_API_KEY,
    MODEL_TRANSLATEGEMMA_MODEL_ID,
    MODEL_EMBEDDINGGEMMA_URL,
    MODEL_EMBEDDINGGEMMA_API_KEY,
    MODEL_EMBEDDINGGEMMA_MODEL_ID,
)

GEMMA4_CONFIG = ModelConfig(
    provider="llama.cpp",
    base_url=MODEL_GEMMA4_URL or "",
    api_key=MODEL_GEMMA4_API_KEY,
    model=MODEL_GEMMA4_MODEL_ID,
    supports_vision=True,
    supports_audio=True,
)

TRANSLATEGEMMA_CONFIG = ModelConfig(
    provider="llama.cpp",
    base_url=MODEL_TRANSLATEGEMMA_URL or "",
    api_key=MODEL_TRANSLATEGEMMA_API_KEY,
    model=MODEL_TRANSLATEGEMMA_MODEL_ID,
)

EMBEDDINGGEMMA_CONFIG = ModelConfig(
    provider="llama.cpp",
    base_url=MODEL_EMBEDDINGGEMMA_URL or "",
    api_key=MODEL_EMBEDDINGGEMMA_API_KEY,
    model=MODEL_EMBEDDINGGEMMA_MODEL_ID,
)
