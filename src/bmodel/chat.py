from langchain.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .base import ChatModelCapability, ModelConfig
from .config import get_model_config


def build_chat_model(config: ModelConfig) -> BaseChatModel:
    def api_key_func() -> str:
        return config.api_key

    match config.provider:
        case "llama.cpp" | "openai" | "openrouter":
            return ChatOpenAI(
                base_url=config.base_url,
                model=config.model,
                temperature=config.temperature,
                api_key=api_key_func,
                timeout=config.timeout,
                streaming=config.streaming,
                max_completion_tokens=config.max_tokens,
                stream_usage=True,
            )
        case _:
            raise ValueError(
                f"Unsupported model provider: `{config.provider}`",
            )


def init_model(
    *,
    capability: ChatModelCapability = "chat",
    model: ModelConfig | None = None,
) -> BaseChatModel:
    config = get_model_config(
        capability,
        model=model,
    )

    return build_chat_model(config)


def init_chat_model(
    *,
    model: ModelConfig | None = None,
) -> BaseChatModel:
    return init_model(
        capability="chat",
        model=model,
    )


def init_vision_model(
    *,
    model: ModelConfig | None = None,
) -> BaseChatModel:
    return init_model(
        capability="vision",
        model=model,
    )


def init_reasoning_model(
    *,
    model: ModelConfig | None = None,
) -> BaseChatModel:
    return init_model(
        capability="reasoning",
        model=model,
    )


def init_translation_model(
    *,
    model: ModelConfig | None = None,
) -> BaseChatModel:
    return init_model(
        capability="translation",
        model=model,
    )
