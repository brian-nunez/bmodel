from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from .base import ModelConfig
from .config import get_embedding_model_config


def init_embedding_model(
    *,
    model: ModelConfig | None = None,
) -> Embeddings:
    config = get_embedding_model_config(model=model)

    def api_key_func() -> str:
        return config.api_key

    match config.provider:
        case "llama.cpp" | "openai" | "openrouter":
            return OpenAIEmbeddings(
                base_url=config.base_url,
                model=config.model,
                api_key=api_key_func,
                timeout=config.timeout,
            )
        case _:
            raise ValueError(
                f"Unsupported model provider: `{config.provider}`",
            )
