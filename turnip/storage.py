from __future__ import annotations

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
    """Record of a single LLM call within an experiment."""

    project: str
    experiment: str
    run: str
    instance: str
    turn: int
    cache_key: str
    response: LLMResponse


class CacheStore:
    """Asynchronous cache for provider responses."""

    async def fetch(self, cache_key: str) -> Optional[LLMResponse]:
        """Return cached response for the given key if present."""
        raise NotImplementedError


class MemoryCacheStore(CacheStore):
    """In-memory cache used for testing and simple runs."""

    def __init__(self) -> None:
        self._data: Dict[str, LLMResponse] = {}

    async def fetch(self, cache_key: str) -> Optional[LLMResponse]:
        return self._data.get(cache_key)

    async def insert(self, cache_key: str, response: LLMResponse) -> None:
        if cache_key not in self._data:
            self._data[cache_key] = response


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
                    project TEXT,
                    experiment TEXT,
                    run TEXT,
                    instance TEXT,
                    turn INTEGER,
                    cache_key TEXT,
                    response JSONB,
                    PRIMARY KEY (project, experiment, run, instance, turn)
                );
                """
            )

    async def fetch(
        self,
        project: str,
        experiment: str,
        run: str,
        instance: str,
        turn: int,
    ) -> Optional[LLMResult]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT response, cache_key FROM llm_results
                WHERE project=$1 AND experiment=$2 AND run=$3
                  AND instance=$4 AND turn=$5
                """,
                project,
                experiment,
                run,
                instance,
                turn,
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
                    project=project,
                    experiment=experiment,
                    run=run,
                    instance=instance,
                    turn=turn,
                    cache_key=row["cache_key"],
                    response=resp,
                )
            return None

    async def insert(self, result: LLMResult) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO llm_results(
                    project,
                    experiment,
                    run,
                    instance,
                    turn,
                    cache_key,
                    response
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (project, experiment, run, instance, turn) DO NOTHING
                """,
                result.project,
                result.experiment,
                result.run,
                result.instance,
                result.turn,
                result.cache_key,
                json.dumps(asdict(result.response)),
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


class MemoryResultStore:
    """In-memory implementation of ``ResultStore`` for tests."""

    def __init__(self) -> None:
        self._data: Dict[tuple[str, str, str, str, int], LLMResult] = {}

    async def fetch(
        self,
        project: str,
        experiment: str,
        run: str,
        instance: str,
        turn: int,
    ) -> Optional[LLMResult]:
        key = (project, experiment, run, instance, turn)
        return self._data.get(key)

    async def insert(self, result: LLMResult) -> None:
        key = (
            result.project,
            result.experiment,
            result.run,
            result.instance,
            result.turn,
        )
        self._data.setdefault(key, result)

    async def close(self) -> None:  # pragma: no cover - for API compat
        self._data.clear()
