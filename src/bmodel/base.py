from dataclasses import dataclass
from typing import Literal

ChatModelCapability = Literal[
    "chat",
    "vision",
    "reasoning",
    "translation",
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

CheckpointerBackend = Literal[
    "memory",
    "sqlite",
    "postgres",
]


@dataclass(frozen=True)
class CheckpointerConfig:
    backend: CheckpointerBackend = "memory"
    sqlite_path: str | None = None
    postgres_url: str | None = None
    async_mode: bool = True


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


ModelsAvailable = dict[ChatModelCapability, ModelConfig]
