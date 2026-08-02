import math

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from .base import ModelConfig, SimilarityMetric
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


def similarity(
    a: list[float],
    b: list[float],
    *,
    metric: SimilarityMetric = "cosine",
) -> float:
    match metric:
        case "cosine":
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = math.sqrt(sum(x * x for x in a))
            magnitude_b = math.sqrt(sum(y * y for y in b))
            return dot_product / (magnitude_a * magnitude_b)
        case "dot":
            return sum(x * y for x, y in zip(a, b))
        case "euclidean":
            distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
            return 1 / (1 + distance)
        case _:
            raise ValueError(f"Unsupported similarity metric: `{metric}`")
