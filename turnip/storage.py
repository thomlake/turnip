from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

try:
    import asyncpg
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    asyncpg = None  # type: ignore


@dataclass
class LLMResult:
    prompt: str
    response: str


class ResultStore:
    """Asynchronous result store using PostgreSQL."""

    def __init__(self, dsn: str) -> None:
        if asyncpg is None:
            raise RuntimeError("asyncpg is required for ResultStore")
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(dsn=self._dsn)
            await self._init_table()

    async def _init_table(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_results (
                    prompt TEXT PRIMARY KEY,
                    response TEXT
                );
                """
            )

    async def fetch(self, prompt: str) -> Optional[LLMResult]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT prompt, response FROM llm_results WHERE prompt=$1",
                prompt,
            )
            if row:
                return LLMResult(prompt=row["prompt"], response=row["response"])
            return None

    async def insert(self, result: LLMResult) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO llm_results(prompt, response) VALUES ($1, $2)"
                " ON CONFLICT (prompt) DO NOTHING",
                result.prompt,
                result.response,
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
