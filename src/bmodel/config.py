from typing import get_args

from .base import ModelConfig, ChatModelCapability, ModelsAvailable
from .env import GEMMA4_CONFIG, EMBEDDINGGEMMA_CONFIG


DEFAULT_MODELS: ModelsAvailable = {
    "chat": GEMMA4_CONFIG,
    "vision": GEMMA4_CONFIG,
    "reasoning": GEMMA4_CONFIG,
    "translation": GEMMA4_CONFIG,
}

DEFAULT_EMBEDDING_MODEL: ModelConfig = EMBEDDINGGEMMA_CONFIG
DEFAULT_VIDEO_MODEL: ModelConfig = GEMMA4_CONFIG
DEFAULT_AUDIO_MODEL: ModelConfig = GEMMA4_CONFIG

_live_models: ModelsAvailable = dict(DEFAULT_MODELS)
_live_embedding_model: ModelConfig = DEFAULT_EMBEDDING_MODEL
_live_video_model: ModelConfig = DEFAULT_VIDEO_MODEL
_live_audio_model: ModelConfig = DEFAULT_AUDIO_MODEL


def reset_defaults() -> ModelsAvailable:
    global _live_models
    global _live_embedding_model
    global _live_video_model
    global _live_audio_model

    _live_models = dict(DEFAULT_MODELS)
    _live_embedding_model = DEFAULT_EMBEDDING_MODEL
    _live_video_model = DEFAULT_VIDEO_MODEL
    _live_audio_model = DEFAULT_AUDIO_MODEL

    return dict(_live_models)


def configure(
    *,
    chat_model: ModelConfig | None = None,
    vision_model: ModelConfig | None = None,
    reasoning_model: ModelConfig | None = None,
    translation_model: ModelConfig | None = None,
):
    global _live_models

    updated_models: ModelsAvailable = {}

    if chat_model is not None:
        updated_models["chat"] = chat_model

    if vision_model is not None:
        updated_models["vision"] = vision_model

    if reasoning_model is not None:
        updated_models["reasoning"] = reasoning_model

    if translation_model is not None:
        updated_models["translation"] = translation_model

    _live_models = {
        **_live_models,
        **updated_models,
    }

    return dict(_live_models)


def configure_embedding_model(model: ModelConfig) -> ModelConfig:
    global _live_embedding_model

    _live_embedding_model = model

    return _live_embedding_model


def get_embedding_model_config(
    *,
    model: ModelConfig | None = None,
) -> ModelConfig:
    if model is not None:
        return model

    return _live_embedding_model


def configure_video_model(model: ModelConfig) -> ModelConfig:
    global _live_video_model

    _live_video_model = model

    return _live_video_model


def get_video_model_config(
    *,
    model: ModelConfig | None = None,
) -> ModelConfig:
    resolved = model if model is not None else _live_video_model

    if not resolved.supports_vision:
        raise ValueError(
            f"Configured video model `{resolved.model}` does not declare "
            "vision support (`ModelConfig.supports_vision=False`). Set "
            "`supports_vision=True` on the ModelConfig if this model "
            "actually accepts image input."
        )

    return resolved


def configure_audio_model(model: ModelConfig) -> ModelConfig:
    global _live_audio_model

    _live_audio_model = model

    return _live_audio_model


def get_audio_model_config(
    *,
    model: ModelConfig | None = None,
) -> ModelConfig:
    resolved = model if model is not None else _live_audio_model

    if not resolved.supports_audio:
        raise ValueError(
            f"Configured audio model `{resolved.model}` does not declare "
            "audio support (`ModelConfig.supports_audio=False`). Set "
            "`supports_audio=True` on the ModelConfig if this model "
            "actually accepts audio input."
        )

    return resolved


def get_model_config(
    capability: ChatModelCapability,
    *,
    model: ModelConfig | None = None,
) -> ModelConfig:
    if model is not None:
        resolved = model
    else:
        if capability not in get_args(ChatModelCapability):
            raise ValueError(f"Unsupported capability `{capability}`")

        resolved = _live_models.get(capability)

        if resolved is None:
            raise ValueError(f"Model not available for capbability `{capability}`")

    if capability == "vision" and not resolved.supports_vision:
        raise ValueError(
            f"Configured vision model `{resolved.model}` does not declare "
            "vision support (`ModelConfig.supports_vision=False`). Set "
            "`supports_vision=True` on the ModelConfig if this model "
            "actually accepts image input."
        )

    return resolved
