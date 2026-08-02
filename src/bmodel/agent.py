from typing import Any

from langchain.agents import create_agent
from langgraph.checkpoint.base import BaseCheckpointSaver

from .base import CheckpointerConfig, ModelConfig
from .chat import build_chat_model
from .checkpoint import init_checkpointer
from .config import get_model_config


async def init_agent(
    *,
    tools: list[Any],
    model: ModelConfig | None = None,
    checkpointer: BaseCheckpointSaver | CheckpointerConfig | None = None,
    **kwargs: Any,
):
    chat_model = build_chat_model(get_model_config("chat", model=model))

    if isinstance(checkpointer, BaseCheckpointSaver):
        resolved_checkpointer = checkpointer
    elif isinstance(checkpointer, CheckpointerConfig):
        resolved_checkpointer = await init_checkpointer(config=checkpointer)
    else:
        resolved_checkpointer = await init_checkpointer()

    return create_agent(
        model=chat_model,
        tools=tools,
        checkpointer=resolved_checkpointer,
        **kwargs,
    )
