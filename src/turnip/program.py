from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

from turnip.errors import FailedMapItem, MapExecutionError
from turnip.scope import RunContext, reset_run_context, set_run_context
from turnip.storage import ExperimentStorage

T = TypeVar("T")


def _default_db_path(experiment: str) -> Path:
    return Path(".turnip") / "experiments" / f"{experiment}.sqlite3"


class Program:
    def __init__(self, experiment: str, config: dict[str, Any], *, db_path: str | Path | None = None) -> None:
        self.experiment = experiment
        self.config = config
        self.db_path = Path(db_path) if db_path is not None else _default_db_path(experiment)

    async def map(
        self,
        fn: Callable[[dict[str, Any]], Awaitable[T]],
        items: list[dict[str, Any]],
        *,
        repeat: int = 1,
        stage: str | None = None,
        key: str | Callable[[dict[str, Any]], str],
        max_concurrency: int = 32,
    ) -> list[T]:
        if repeat <= 0:
            raise ValueError("repeat must be > 0")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")

        stage_name = stage or fn.__name__
        storage = ExperimentStorage(self.db_path)
        await storage.connect()
        await storage.upsert_experiment(self.experiment, json.dumps(self.config))

        jobs: list[tuple[int, dict[str, Any], str, int]] = []
        for i, item in enumerate(items):
            item_key = self._extract_key(item, key)
            for trial in range(repeat):
                jobs.append((i, item, item_key, trial))

        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_job(index: int, item: dict[str, Any], item_key: str, trial: int) -> tuple[int, str, int, T]:
            context = RunContext(
                experiment=self.experiment,
                stage=stage_name,
                key=item_key,
                trial=trial,
                storage=storage,
            )
            token = set_run_context(context)
            context.stack.append(f"{fn.__name__}[0]")
            try:
                async with semaphore:
                    value = await fn(item)
                return index, item_key, trial, value
            finally:
                context.stack.pop()
                reset_run_context(token)

        tasks = [asyncio.create_task(run_job(i, item, item_key, trial)) for i, item, item_key, trial in jobs]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        failures: list[FailedMapItem] = []
        ordered_results: list[T | None] = [None] * len(jobs)

        for job_i, result in enumerate(task_results):
            item_index, _, item_key, trial = jobs[job_i]
            if isinstance(result, BaseException):
                failures.append(FailedMapItem(index=item_index, key=item_key, trial=trial, error=result))
            else:
                ordered_results[job_i] = result[3]

        await storage.close()

        if failures:
            raise MapExecutionError(stage_name, failures)

        return [r for r in ordered_results if r is not None]

    @staticmethod
    def _extract_key(item: dict[str, Any], key: str | Callable[[dict[str, Any]], str]) -> str:
        if isinstance(key, str):
            if key not in item:
                raise KeyError(f"item is missing key field '{key}'")
            value = item[key]
        else:
            value = key(item)

        if value is None:
            raise ValueError("derived key cannot be None")

        return str(value)
