from langchain.chat_models import BaseChatModel

from .base import ModelConfig
from .chat import build_chat_model
from .config import get_audio_model_config


def init_audio_model(
    *,
    model: ModelConfig | None = None,
) -> BaseChatModel:
    return build_chat_model(get_audio_model_config(model=model))
