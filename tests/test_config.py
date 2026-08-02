from typing import cast

import pytest

import bmodel.config as config_module
from bmodel.base import ChatModelCapability, CheckpointerConfig, ModelConfig
from bmodel.config import (
    DEFAULT_AUDIO_MODEL,
    DEFAULT_CHECKPOINTER_CONFIG,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODELS,
    DEFAULT_VIDEO_MODEL,
    configure,
    configure_audio_model,
    configure_checkpointer,
    configure_embedding_model,
    configure_video_model,
    get_audio_model_config,
    get_checkpointer_config,
    get_embedding_model_config,
    get_model_config,
    get_video_model_config,
    reset_defaults,
)


def _make_config(model: str) -> ModelConfig:
    return ModelConfig(
        provider="openai",
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model=model,
        supports_vision=True,
        supports_audio=True,
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


def test_get_embedding_model_config_returns_default():
    assert get_embedding_model_config() == DEFAULT_EMBEDDING_MODEL


def test_get_embedding_model_config_returns_explicit_model():
    override = _make_config("embedding-override")

    assert get_embedding_model_config(model=override) is override


def test_configure_embedding_model_overrides_default():
    override = _make_config("embedding-override")

    configure_embedding_model(override)

    assert get_embedding_model_config() == override


def test_reset_defaults_clears_embedding_override():
    override = _make_config("embedding-override")
    configure_embedding_model(override)
    assert get_embedding_model_config() == override

    reset_defaults()

    assert get_embedding_model_config() == DEFAULT_EMBEDDING_MODEL


def test_get_video_model_config_returns_default():
    assert get_video_model_config() == DEFAULT_VIDEO_MODEL


def test_get_video_model_config_returns_explicit_model():
    override = _make_config("video-override")

    assert get_video_model_config(model=override) is override


def test_configure_video_model_overrides_default():
    override = _make_config("video-override")

    configure_video_model(override)

    assert get_video_model_config() == override


def test_configure_video_model_independent_of_vision_and_embedding():
    video_override = _make_config("video-override")
    vision_override = _make_config("vision-override")
    embedding_override = _make_config("embedding-override")

    configure_video_model(video_override)
    configure(vision_model=vision_override)
    configure_embedding_model(embedding_override)

    assert get_video_model_config() == video_override
    assert get_model_config("vision") == vision_override
    assert get_embedding_model_config() == embedding_override


def test_reset_defaults_clears_video_override():
    override = _make_config("video-override")
    configure_video_model(override)
    assert get_video_model_config() == override

    reset_defaults()

    assert get_video_model_config() == DEFAULT_VIDEO_MODEL


def test_get_audio_model_config_returns_default():
    assert get_audio_model_config() == DEFAULT_AUDIO_MODEL


def test_get_audio_model_config_returns_explicit_model():
    override = _make_config("audio-override")

    assert get_audio_model_config(model=override) is override


def test_configure_audio_model_overrides_default():
    override = _make_config("audio-override")

    configure_audio_model(override)

    assert get_audio_model_config() == override


def test_configure_audio_model_independent_of_vision_and_video():
    audio_override = _make_config("audio-override")
    vision_override = _make_config("vision-override")
    video_override = _make_config("video-override")

    configure_audio_model(audio_override)
    configure(vision_model=vision_override)
    configure_video_model(video_override)

    assert get_audio_model_config() == audio_override
    assert get_model_config("vision") == vision_override
    assert get_video_model_config() == video_override


def test_reset_defaults_clears_audio_override():
    override = _make_config("audio-override")
    configure_audio_model(override)
    assert get_audio_model_config() == override

    reset_defaults()

    assert get_audio_model_config() == DEFAULT_AUDIO_MODEL


def test_get_checkpointer_config_returns_default():
    assert get_checkpointer_config() == DEFAULT_CHECKPOINTER_CONFIG


def test_get_checkpointer_config_returns_explicit_config():
    override = CheckpointerConfig(backend="sqlite", sqlite_path="/tmp/checkpoints.db")

    assert get_checkpointer_config(config=override) is override


def test_configure_checkpointer_overrides_default():
    override = CheckpointerConfig(backend="postgres", postgres_url="postgresql://x")

    configure_checkpointer(override)

    assert get_checkpointer_config() == override


def test_reset_defaults_clears_checkpointer_override():
    override = CheckpointerConfig(backend="sqlite", sqlite_path="/tmp/checkpoints.db")
    configure_checkpointer(override)
    assert get_checkpointer_config() == override

    reset_defaults()

    assert get_checkpointer_config() == DEFAULT_CHECKPOINTER_CONFIG


def _make_unsupported_config(model: str) -> ModelConfig:
    return ModelConfig(
        provider="openai",
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model=model,
    )


def test_get_model_config_rejects_vision_model_without_vision_support():
    text_only = _make_unsupported_config("text-only-model")

    with pytest.raises(ValueError, match="does not declare vision support"):
        get_model_config("vision", model=text_only)


def test_get_model_config_does_not_require_vision_support_for_other_capabilities():
    text_only = _make_unsupported_config("text-only-model")

    assert get_model_config("chat", model=text_only) is text_only
    assert get_model_config("reasoning", model=text_only) is text_only
    assert get_model_config("translation", model=text_only) is text_only


def test_configure_vision_default_rejects_model_without_vision_support():
    text_only = _make_unsupported_config("text-only-model")
    configure(vision_model=text_only)

    with pytest.raises(ValueError, match="does not declare vision support"):
        get_model_config("vision")


def test_get_video_model_config_rejects_model_without_vision_support():
    text_only = _make_unsupported_config("text-only-model")

    with pytest.raises(ValueError, match="does not declare vision support"):
        get_video_model_config(model=text_only)


def test_configure_video_model_rejects_model_without_vision_support():
    text_only = _make_unsupported_config("text-only-model")
    configure_video_model(text_only)

    with pytest.raises(ValueError, match="does not declare vision support"):
        get_video_model_config()


def test_get_audio_model_config_rejects_model_without_audio_support():
    text_only = _make_unsupported_config("text-only-model")

    with pytest.raises(ValueError, match="does not declare audio support"):
        get_audio_model_config(model=text_only)


def test_configure_audio_model_rejects_model_without_audio_support():
    text_only = _make_unsupported_config("text-only-model")
    configure_audio_model(text_only)

    with pytest.raises(ValueError, match="does not declare audio support"):
        get_audio_model_config()
