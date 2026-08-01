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


ModelsAvailable = dict[ChatModelCapability, ModelConfig]
