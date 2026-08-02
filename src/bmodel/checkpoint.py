import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from .base import CheckpointerConfig
from .config import get_checkpointer_config


def _default_sqlite_path() -> str:
    path = Path.home() / ".local" / "share" / "bmodel" / "checkpoints.db"
    path.parent.mkdir(parents=True, exist_ok=True)

    return str(path)


async def _init_sqlite_checkpointer(config: CheckpointerConfig) -> BaseCheckpointSaver:
    sqlite_path = config.sqlite_path or _default_sqlite_path()

    if config.async_mode:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        conn = await aiosqlite.connect(sqlite_path)
        saver = AsyncSqliteSaver(conn)
        await saver.setup()

        return saver

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    sync_saver = SqliteSaver(conn)
    sync_saver.setup()

    return sync_saver


async def _init_postgres_checkpointer(config: CheckpointerConfig) -> BaseCheckpointSaver:
    if not config.postgres_url:
        raise ValueError(
            'CheckpointerConfig(backend="postgres") requires `postgres_url` to be set.'
        )

    if config.async_mode:
        from psycopg import AsyncConnection
        from psycopg.rows import DictRow, dict_row
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async_conn = await AsyncConnection[DictRow].connect(
            config.postgres_url,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
        async_saver = AsyncPostgresSaver(async_conn)
        await async_saver.setup()

        return async_saver

    from psycopg import Connection
    from psycopg.rows import DictRow, dict_row
    from langgraph.checkpoint.postgres import PostgresSaver

    conn = Connection[DictRow].connect(
        config.postgres_url,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
    sync_saver = PostgresSaver(conn)
    sync_saver.setup()

    return sync_saver


async def init_checkpointer(
    *,
    config: CheckpointerConfig | None = None,
) -> BaseCheckpointSaver:
    resolved = get_checkpointer_config(config=config)

    match resolved.backend:
        case "memory":
            return InMemorySaver()
        case "sqlite":
            return await _init_sqlite_checkpointer(resolved)
        case "postgres":
            return await _init_postgres_checkpointer(resolved)
        case _:
            raise ValueError(f"Unsupported checkpointer backend: `{resolved.backend}`")
