from typing import get_args

from .base import ModelConfig, ChatModelCapability, ModelsAvailable
from .env import GEMMA4_CONFIG


DEFAULT_MODELS: ModelsAvailable = {
    "chat": GEMMA4_CONFIG,
    "vision": GEMMA4_CONFIG,
    "reasoning": GEMMA4_CONFIG,
    "translation": GEMMA4_CONFIG,
}

_live_models: ModelsAvailable = dict(DEFAULT_MODELS)


def reset_defaults() -> ModelsAvailable:
    global _live_models

    _live_models = dict(DEFAULT_MODELS)

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


def get_model_config(
    capability: ChatModelCapability,
    *,
    model: ModelConfig | None = None,
) -> ModelConfig:
    if model is not None:
        return model

    if capability not in get_args(ChatModelCapability):
        raise ValueError(f"Unsupported capability `{capability}`")

    model = _live_models.get(capability)

    if model is None:
        raise ValueError(f"Model not available for capbability `{capability}`")

    return model
