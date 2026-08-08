from dataclasses import dataclass
from typing import Literal

ChatModelCapability = Literal[
    "chat",
    "vision",
    "reasoning",
    "translation",
    "code",
]

ModelProvider = Literal[
    "llama.cpp",
    "openai",
    "openrouter",
]

SimilarityMetric = Literal[
    "cosine",
    "dot",
    "euclidean",
]


@dataclass(frozen=True)
class ModelConfig:
    provider: ModelProvider
    base_url: str
    api_key: str
    model: str

    temperature: float = 0.0
    timeout: int = 300
    max_tokens: int | None = None
    streaming: bool = True
    supports_vision: bool = False
    supports_audio: bool = False
    stop: list[str] | None = None


ModelsAvailable = dict[ChatModelCapability, ModelConfig]
