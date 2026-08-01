from typing import cast

import pytest

import bmodel.config as config_module
from bmodel.base import ChatModelCapability, ModelConfig
from bmodel.config import DEFAULT_MODELS, configure, get_model_config, reset_defaults


def _make_config(model: str) -> ModelConfig:
    return ModelConfig(
        provider="openai",
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model=model,
    )


def test_get_model_config_returns_default_for_each_capability():
    for capability, expected in DEFAULT_MODELS.items():
        assert get_model_config(capability) == expected


def test_get_model_config_returns_explicit_model():
    override = _make_config("explicit-model")

    assert get_model_config("chat", model=override) is override


def test_get_model_config_rejects_unknown_capability():
    with pytest.raises(ValueError, match="Unsupported capability"):
        get_model_config(cast(ChatModelCapability, "unknown-capability"))


def test_get_model_config_raises_when_model_missing(monkeypatch):
    monkeypatch.delitem(config_module._live_models, "chat")

    with pytest.raises(ValueError, match="Model not available"):
        get_model_config("chat")


def test_configure_overrides_single_capability():
    override = _make_config("chat-override")

    configure(chat_model=override)

    assert get_model_config("chat") == override
    assert get_model_config("vision") == DEFAULT_MODELS["vision"]


def test_configure_overrides_all_capabilities():
    chat = _make_config("chat-model")
    vision = _make_config("vision-model")
    reasoning = _make_config("reasoning-model")
    translation = _make_config("translation-model")

    configure(
        chat_model=chat,
        vision_model=vision,
        reasoning_model=reasoning,
        translation_model=translation,
    )

    assert get_model_config("chat") == chat
    assert get_model_config("vision") == vision
    assert get_model_config("reasoning") == reasoning
    assert get_model_config("translation") == translation


def test_configure_with_no_arguments_leaves_defaults_unchanged():
    configure()

    for capability, expected in DEFAULT_MODELS.items():
        assert get_model_config(capability) == expected


def test_reset_defaults_clears_overrides():
    override = _make_config("chat-override")
    configure(chat_model=override)
    assert get_model_config("chat") == override

    result = reset_defaults()

    assert get_model_config("chat") == DEFAULT_MODELS["chat"]
    assert result == DEFAULT_MODELS
