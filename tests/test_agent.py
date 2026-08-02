from unittest.mock import AsyncMock, MagicMock, patch

from bmodel.agent import init_agent
from bmodel.base import CheckpointerConfig, ModelConfig


def _make_model_config(model: str) -> ModelConfig:
    return ModelConfig(
        provider="openai",
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model=model,
    )


async def test_init_agent_builds_checkpointer_from_default_when_none_given():
    mock_chat_model = object()
    mock_checkpointer = object()
    mock_agent = object()

    with (
        patch("bmodel.agent.build_chat_model", return_value=mock_chat_model) as mock_build,
        patch(
            "bmodel.agent.init_checkpointer",
            new=AsyncMock(return_value=mock_checkpointer),
        ) as mock_init_checkpointer,
        patch("bmodel.agent.create_agent", return_value=mock_agent) as mock_create_agent,
    ):
        result = await init_agent(tools=[])

    mock_build.assert_called_once()
    mock_init_checkpointer.assert_awaited_once_with()
    mock_create_agent.assert_called_once_with(
        model=mock_chat_model,
        tools=[],
        checkpointer=mock_checkpointer,
    )
    assert result is mock_agent


async def test_init_agent_builds_checkpointer_from_explicit_config():
    mock_checkpointer = object()
    config = CheckpointerConfig(backend="sqlite", sqlite_path="/tmp/checkpoints.db")

    with (
        patch("bmodel.agent.build_chat_model", return_value=object()),
        patch(
            "bmodel.agent.init_checkpointer",
            new=AsyncMock(return_value=mock_checkpointer),
        ) as mock_init_checkpointer,
        patch("bmodel.agent.create_agent", return_value=object()) as mock_create_agent,
    ):
        await init_agent(tools=[], checkpointer=config)

    mock_init_checkpointer.assert_awaited_once_with(config=config)
    _, kwargs = mock_create_agent.call_args
    assert kwargs["checkpointer"] is mock_checkpointer


async def test_init_agent_uses_raw_checkpointer_instance_directly():
    from langgraph.checkpoint.base import BaseCheckpointSaver

    raw_checkpointer = MagicMock(spec=BaseCheckpointSaver)

    with (
        patch("bmodel.agent.build_chat_model", return_value=object()),
        patch("bmodel.agent.init_checkpointer", new=AsyncMock()) as mock_init_checkpointer,
        patch("bmodel.agent.create_agent", return_value=object()) as mock_create_agent,
    ):
        await init_agent(tools=[], checkpointer=raw_checkpointer)

    mock_init_checkpointer.assert_not_awaited()
    _, kwargs = mock_create_agent.call_args
    assert kwargs["checkpointer"] is raw_checkpointer


async def test_init_agent_resolves_model_via_get_model_config():
    override = _make_model_config("chat-override")

    with (
        patch("bmodel.agent.build_chat_model", return_value=object()) as mock_build,
        patch("bmodel.agent.init_checkpointer", new=AsyncMock()),
        patch("bmodel.agent.create_agent", return_value=object()),
    ):
        await init_agent(tools=[], model=override)

    mock_build.assert_called_once_with(override)


async def test_init_agent_passes_tools_and_extra_kwargs_through():
    tool = object()

    with (
        patch("bmodel.agent.build_chat_model", return_value=object()),
        patch("bmodel.agent.init_checkpointer", new=AsyncMock()),
        patch("bmodel.agent.create_agent", return_value=object()) as mock_create_agent,
    ):
        await init_agent(tools=[tool], system_prompt="be helpful")

    _, kwargs = mock_create_agent.call_args
    assert kwargs["tools"] == [tool]
    assert kwargs["system_prompt"] == "be helpful"
