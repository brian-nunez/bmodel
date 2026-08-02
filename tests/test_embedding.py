from typing import cast, get_args
from unittest.mock import patch

import pytest

from bmodel.base import ModelConfig, ModelProvider
from bmodel.embedding import init_embedding_model


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
