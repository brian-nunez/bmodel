from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bmodel.base import CheckpointerBackend, CheckpointerConfig
from bmodel.checkpoint import init_checkpointer
from bmodel.config import configure_checkpointer


async def test_init_checkpointer_defaults_to_memory():
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = await init_checkpointer()

    assert isinstance(checkpointer, InMemorySaver)


async def test_init_checkpointer_explicit_memory_config():
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = await init_checkpointer(config=CheckpointerConfig(backend="memory"))

    assert isinstance(checkpointer, InMemorySaver)


async def test_init_checkpointer_respects_configure_checkpointer():
    from langgraph.checkpoint.memory import InMemorySaver

    configure_checkpointer(CheckpointerConfig(backend="memory"))

    checkpointer = await init_checkpointer()

    assert isinstance(checkpointer, InMemorySaver)


async def test_init_checkpointer_sqlite_async(tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = tmp_path / "checkpoints.db"

    checkpointer = await init_checkpointer(
        config=CheckpointerConfig(
            backend="sqlite",
            sqlite_path=str(db_path),
            async_mode=True,
        ),
    )

    assert isinstance(checkpointer, AsyncSqliteSaver)
    assert db_path.exists()

    await checkpointer.conn.close()


async def test_init_checkpointer_sqlite_sync(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = tmp_path / "checkpoints.db"

    checkpointer = await init_checkpointer(
        config=CheckpointerConfig(
            backend="sqlite",
            sqlite_path=str(db_path),
            async_mode=False,
        ),
    )

    assert isinstance(checkpointer, SqliteSaver)
    assert db_path.exists()

    checkpointer.conn.close()


async def test_init_checkpointer_sqlite_uses_default_path_when_unset(monkeypatch, tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    monkeypatch.setattr("bmodel.checkpoint.Path.home", lambda: tmp_path)

    checkpointer = await init_checkpointer(
        config=CheckpointerConfig(backend="sqlite", async_mode=False),
    )

    expected_path = tmp_path / ".local" / "share" / "bmodel" / "checkpoints.db"
    assert expected_path.exists()
    assert isinstance(checkpointer, SqliteSaver)
    checkpointer.conn.close()


async def test_init_checkpointer_postgres_requires_url():
    with pytest.raises(ValueError, match="requires `postgres_url`"):
        await init_checkpointer(config=CheckpointerConfig(backend="postgres"))


async def test_init_checkpointer_postgres_async():
    mock_conn = object()
    mock_saver = MagicMock()
    mock_saver.setup = AsyncMock()

    with (
        patch(
            "psycopg.AsyncConnection.connect",
            new=AsyncMock(return_value=mock_conn),
        ) as mock_connect,
        patch(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
            return_value=mock_saver,
        ) as mock_saver_cls,
    ):
        checkpointer = await init_checkpointer(
            config=CheckpointerConfig(
                backend="postgres",
                postgres_url="postgresql://user:pass@localhost/db",
                async_mode=True,
            ),
        )

    mock_connect.assert_called_once()
    mock_saver_cls.assert_called_once_with(mock_conn)
    mock_saver.setup.assert_awaited_once()
    assert checkpointer is mock_saver


async def test_init_checkpointer_postgres_sync():
    mock_conn = object()
    mock_saver = MagicMock()

    with (
        patch(
            "psycopg.Connection.connect",
            return_value=mock_conn,
        ) as mock_connect,
        patch(
            "langgraph.checkpoint.postgres.PostgresSaver",
            return_value=mock_saver,
        ) as mock_saver_cls,
    ):
        checkpointer = await init_checkpointer(
            config=CheckpointerConfig(
                backend="postgres",
                postgres_url="postgresql://user:pass@localhost/db",
                async_mode=False,
            ),
        )

    mock_connect.assert_called_once()
    mock_saver_cls.assert_called_once_with(mock_conn)
    mock_saver.setup.assert_called_once()
    assert checkpointer is mock_saver


async def test_init_checkpointer_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported checkpointer backend"):
        await init_checkpointer(
            config=CheckpointerConfig(
                backend=cast(CheckpointerBackend, "unknown-backend"),
            ),
        )
