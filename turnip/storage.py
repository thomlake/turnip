from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

try:
    import asyncpg
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    asyncpg = None  # type: ignore


@dataclass
class LLMResponse:
    """Response returned from an LLM provider."""

    messages: List[Dict[str, str]]
    content: str
    parameters: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class LLMResult:
    experiment: str
    run: str
    instance: str
    response: LLMResponse


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
                    experiment TEXT,
                    run TEXT,
                    instance TEXT,
                    response JSONB,
                    PRIMARY KEY (experiment, run, instance)
                );
                """
            )

    async def fetch(
        self, experiment: str, run: str, instance: str
    ) -> Optional[LLMResult]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT response FROM llm_results
                WHERE experiment=$1 AND run=$2 AND instance=$3
                """,
                experiment,
                run,
                instance,
            )
            if row:
                data = row["response"]
                resp = LLMResponse(
                    messages=data["messages"],
                    content=data["content"],
                    parameters=data.get("parameters"),
                    tool_calls=data.get("tool_calls"),
                )
                return LLMResult(
                    experiment=experiment,
                    run=run,
                    instance=instance,
                    response=resp,
                )
            return None

    async def insert(self, result: LLMResult) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO llm_results(experiment, run, instance, response)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (experiment, run, instance) DO NOTHING
                """,
                result.experiment,
                result.run,
                result.instance,
                json.dumps(asdict(result.response)),
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
