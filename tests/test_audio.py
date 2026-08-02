from typing import cast, get_args
from unittest.mock import patch

import pytest

from bmodel.audio import init_audio_model
from bmodel.base import ModelConfig, ModelProvider
from bmodel.config import configure_audio_model


def _make_config(provider: ModelProvider) -> ModelConfig:
    return ModelConfig(
        provider=provider,
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model="testing",
        supports_audio=True,
    )


def test_init_audio_model_uses_default():
    assert init_audio_model() is not None


@pytest.mark.parametrize("provider", get_args(ModelProvider))
def test_init_audio_model_accepts_known_parameters(provider):
    model_config = _make_config(provider)

    assert init_audio_model(model=model_config) is not None


def test_init_audio_model_rejects_unknown_parameters():
    model_config = ModelConfig(
        provider=cast(ModelProvider, "unknown-provider"),
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model="testing",
        supports_audio=True,
    )

    with pytest.raises(ValueError, match="Unsupported model provider"):
        init_audio_model(model=model_config)


def test_init_audio_model_rejects_model_without_audio_support():
    model_config = ModelConfig(
        provider="openai",
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model="text-only-model",
    )

    with pytest.raises(ValueError, match="does not declare audio support"):
        init_audio_model(model=model_config)


def test_init_audio_model_respects_configure_audio_model():
    override = _make_config("openai")
    configure_audio_model(override)

    with patch("bmodel.audio.build_chat_model", return_value=object()) as mock_build:
        init_audio_model()

    mock_build.assert_called_once_with(override)
