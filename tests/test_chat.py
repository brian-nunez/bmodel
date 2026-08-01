from unittest.mock import patch
from typing import get_args, cast
from langchain_core.messages import AIMessage
import pytest

from bmodel.base import ModelProvider, ModelConfig, ChatModelCapability
from bmodel.chat import (
    init_model,
    init_chat_model,
    init_vision_model,
    init_reasoning_model,
    init_translation_model,
)


@pytest.mark.parametrize("capability", get_args(ChatModelCapability))
def test_init_model_accepts_known_capbability_parameters(capability):
    assert init_model(capability=capability) is not None


def test_init_model_rejects_known_capbability_parameters():
    with pytest.raises(
        ValueError,
        match="Unsupported capability `unknown-capability",
    ):
        init_model(capability=cast(ChatModelCapability, "unknown-capability"))


@pytest.mark.parametrize("provider", get_args(ModelProvider))
def test_init_model_accepts_known_parameters(provider):
    model_config = ModelConfig(
        provider=provider,
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model="testing",
    )

    assert init_model(model=model_config) is not None


def test_init_model_rejects_unknown_parameters():
    model_config = ModelConfig(
        provider=cast(ModelProvider, "uknown-provider"),
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model="testing",
    )

    with pytest.raises(ValueError, match="Unsupported model provide"):
        init_model(model=model_config)


def test_init_chat_model():
    assert init_chat_model() is not None


def test_init_vision_model():
    assert init_vision_model() is not None


def test_init_reasoning_model():
    assert init_reasoning_model() is not None


def test_init_translation_model():
    assert init_translation_model() is not None


def test_init_model_api_key_is_lazily_resolved():
    model_config = ModelConfig(
        provider="openai",
        base_url="http://localhost:8080/v1",
        api_key="secret-key",
        model="testing",
    )

    with patch("bmodel.chat.ChatOpenAI") as mock_chat_openai:
        init_model(model=model_config)

    _, kwargs = mock_chat_openai.call_args
    assert kwargs["api_key"]() == "secret-key"


def test_init_chat_model_invoke():
    with patch(
        "langchain_openai.ChatOpenAI.invoke",
        return_value=AIMessage(content="Hello there!"),
    ) as mock_invoke:
        model = init_chat_model()
        result = model.invoke("Hello")

    mock_invoke.assert_called_once_with("Hello")
    assert result.content == "Hello there!"
