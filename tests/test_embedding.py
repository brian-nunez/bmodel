from typing import cast, get_args
from unittest.mock import patch

import pytest

from bmodel.base import ModelConfig, ModelProvider, SimilarityMetric
from bmodel.embedding import init_embedding_model, similarity


def _make_config(provider: ModelProvider) -> ModelConfig:
    return ModelConfig(
        provider=provider,
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model="testing",
    )


def test_init_embedding_model_uses_default():
    assert init_embedding_model() is not None


@pytest.mark.parametrize("provider", get_args(ModelProvider))
def test_init_embedding_model_accepts_known_parameters(provider):
    model_config = _make_config(provider)

    assert init_embedding_model(model=model_config) is not None


def test_init_embedding_model_rejects_unknown_parameters():
    model_config = ModelConfig(
        provider=cast(ModelProvider, "unknown-provider"),
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model="testing",
    )

    with pytest.raises(ValueError, match="Unsupported model provider"):
        init_embedding_model(model=model_config)


def test_init_embedding_model_api_key_is_lazily_resolved():
    model_config = ModelConfig(
        provider="openai",
        base_url="http://localhost:8080/v1",
        api_key="secret-key",
        model="testing",
    )

    with patch("bmodel.embedding.OpenAIEmbeddings") as mock_openai_embeddings:
        init_embedding_model(model=model_config)

    _, kwargs = mock_openai_embeddings.call_args
    assert kwargs["api_key"]() == "secret-key"


def test_similarity_cosine_identical_vectors():
    assert similarity([1.0, 0.0], [1.0, 0.0], metric="cosine") == pytest.approx(1.0)


def test_similarity_cosine_orthogonal_vectors():
    assert similarity([1.0, 0.0], [0.0, 1.0], metric="cosine") == pytest.approx(0.0)


def test_similarity_defaults_to_cosine():
    assert similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_similarity_dot_product():
    assert similarity([2.0, 3.0], [4.0, 5.0], metric="dot") == pytest.approx(23.0)


def test_similarity_euclidean_identical_vectors():
    assert similarity([1.0, 0.0], [1.0, 0.0], metric="euclidean") == pytest.approx(1.0)


def test_similarity_euclidean_known_distance():
    # distance = sqrt((3-0)**2 + (4-0)**2) = 5, so similarity = 1 / (1 + 5)
    assert similarity([0.0, 0.0], [3.0, 4.0], metric="euclidean") == pytest.approx(1 / 6)


def test_similarity_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unsupported similarity metric"):
        similarity([1.0, 0.0], [0.0, 1.0], metric=cast(SimilarityMetric, "unknown"))


@pytest.mark.parametrize("metric", get_args(SimilarityMetric))
def test_similarity_accepts_all_known_metrics(metric):
    assert similarity([1.0, 0.0], [0.0, 1.0], metric=metric) is not None
