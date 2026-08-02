import asyncio
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from .base import CheckpointerConfig
from .config import get_checkpointer_config

_background_loop: asyncio.AbstractEventLoop | None = None
_background_loop_lock = threading.Lock()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    """A single event loop, running forever in its own daemon thread, that
    outlives any individual `asyncio.run()` call the caller makes. Async
    checkpointers (`AsyncSqliteSaver`, `AsyncPostgresSaver`) bind to whatever
    loop is running when they're constructed and use that same loop for every
    later call, including their synchronous `.get_tuple()`/`.put()` bridge
    methods — so they're built here instead of on the caller's own transient
    loop, which would already be closed by the time a later `.invoke()` call
    tries to use it.
    """
    global _background_loop

    with _background_loop_lock:
        if _background_loop is not None and _background_loop.is_running():
            return _background_loop

        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        threading.Thread(target=_run, daemon=True, name="bmodel-checkpointer-loop").start()

        _background_loop = loop

        return loop


async def _run_on_background_loop(coro: Coroutine[Any, Any, Any]) -> Any:
    loop = _get_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)

    return await asyncio.wrap_future(future)


def _default_sqlite_path() -> str:
    path = Path.home() / ".local" / "share" / "bmodel" / "checkpoints.db"
    path.parent.mkdir(parents=True, exist_ok=True)

    return str(path)


async def _init_sqlite_checkpointer(config: CheckpointerConfig) -> BaseCheckpointSaver:
    sqlite_path = config.sqlite_path or _default_sqlite_path()

    async def _build() -> BaseCheckpointSaver:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        conn = await aiosqlite.connect(sqlite_path)
        saver = AsyncSqliteSaver(conn)
        await saver.setup()

        return saver

    return await _run_on_background_loop(_build())


async def _init_postgres_checkpointer(config: CheckpointerConfig) -> BaseCheckpointSaver:
    if not config.postgres_url:
        raise ValueError(
            'CheckpointerConfig(backend="postgres") requires `postgres_url` to be set.'
        )

    async def _build() -> BaseCheckpointSaver:
        from psycopg import AsyncConnection
        from psycopg.rows import DictRow, dict_row
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        assert config.postgres_url is not None

        conn = await AsyncConnection[DictRow].connect(
            config.postgres_url,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
        saver = AsyncPostgresSaver(conn)
        await saver.setup()

        return saver

    return await _run_on_background_loop(_build())


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
