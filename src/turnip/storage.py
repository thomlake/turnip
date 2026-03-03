import asyncio
import datetime as dt
import json
import traceback as tb
from typing import Any, Awaitable, Callable, Literal

import aiosqlite

from .types import ExperimentId, JsonDict, Key, Namespace


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalize_key(k: Any) -> Key:
    """Normalize user keys to a stable string key."""
    if isinstance(k, str):
        return k
    if isinstance(k, (int, float, bool)):
        return str(k)
    # For tuples/lists/dicts and anything else JSON-able
    return json.dumps(k, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _split_step_id(step_id: str) -> tuple[str, str, str]:
    """
    Parse "{namespace}/{program}/{step}" into parts.
    If namespace is empty, step_id is "{program}/{step}".
    """
    parts = step_id.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid step_id {step_id!r}; expected 'program/step' or 'ns/program/step'.")
    if len(parts) == 2:
        return "", parts[0], parts[1]
    # namespace may itself contain slashes; treat last 2 as program/step
    ns = "/".join(parts[:-2])
    return ns, parts[-2], parts[-1]


class Store:
    def __init__(self, path: str):
        self.path = path

    def experiment(self, experiment_id: ExperimentId, *, config: JsonDict) -> "Experiment":
        return Experiment(self.path, experiment_id, config=config)


class Experiment:
    def __init__(self, db_path: str, experiment_id: ExperimentId, *, config: JsonDict):
        self.db_path = db_path
        self.experiment_id = experiment_id
        self.config = config

    def bind(self, *, namespace: Namespace = "") -> "Context":
        return Context(self.db_path, self.experiment_id, namespace=namespace, config=self.config)


class Context:
    """
    Execution handle: (db_path, experiment_id, namespace).
    Exposes run_cached primitive + analysis helpers.
    """

    def __init__(self, db_path: str, experiment_id: ExperimentId, *, namespace: Namespace, config: JsonDict):
        self.db_path = db_path
        self.experiment_id = experiment_id
        self.namespace = namespace or ""
        self._config = config
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._run_locks_guard = asyncio.Lock()
        self._run_locks: dict[tuple[str, Key, int], asyncio.Lock] = {}

    def with_namespace(self, namespace: Namespace) -> "Context":
        return Context(self.db_path, self.experiment_id, namespace=namespace, config=self._config)

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL;")
                await db.execute("PRAGMA synchronous=NORMAL;")
                await db.execute("PRAGMA foreign_keys=ON;")
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS experiments (
                        experiment TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        config_json TEXT NOT NULL
                    );
                    """
                )
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS records (
                        experiment TEXT NOT NULL,
                        step_id TEXT NOT NULL,
                        namespace TEXT NOT NULL,
                        program TEXT NOT NULL,
                        step TEXT NOT NULL,
                        key TEXT NOT NULL,
                        rep INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL CHECK(status IN ('success','error')),
                        updated_at TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,

                        payload_json TEXT NULL,

                        error_type TEXT NULL,
                        error_msg TEXT NULL,
                        traceback TEXT NULL,

                        duration_ms INTEGER NULL,

                        PRIMARY KEY (experiment, step_id, key, rep),
                        FOREIGN KEY (experiment) REFERENCES experiments(experiment)
                    );
                    """
                )
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_records_lookup
                    ON records(experiment, namespace, program, step, rep);
                    """
                )
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_records_status
                    ON records(experiment, status, namespace, program, step);
                    """
                )

                await db.execute(
                    """
                    INSERT INTO experiments (experiment, created_at, config_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(experiment) DO NOTHING;
                    """,
                    (self.experiment_id, _utc_now_iso(), json.dumps(self._config, ensure_ascii=False)),
                )
                await db.commit()

            self._initialized = True

    def make_step_id(self, program: str, step: str) -> str:
        if self.namespace:
            return f"{self.namespace}/{program}/{step}"
        return f"{program}/{step}"

    async def _get_run_lock(self, lock_key: tuple[str, Key, int]) -> asyncio.Lock:
        async with self._run_locks_guard:
            lock = self._run_locks.get(lock_key)
            if lock is None:
                lock = asyncio.Lock()
                self._run_locks[lock_key] = lock
            return lock

    async def _upsert_record(
        self,
        *,
        step_id: str,
        key: Key,
        rep: int,
        status: Literal["success", "error"],
        updated_at: str,
        ns: str,
        program: str,
        step: str,
        duration_ms: int,
        payload_json: str | None = None,
        error_type: str | None = None,
        error_msg: str | None = None,
        trace: str | None = None,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys=ON;")
            await db.execute("BEGIN;")
            await db.execute(
                """
                INSERT INTO records (
                    experiment, step_id, namespace, program, step, key, rep,
                    status, updated_at, attempts,
                    payload_json,
                    error_type, error_msg, traceback,
                    duration_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment, step_id, key, rep) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    attempts=records.attempts + 1,
                    payload_json=excluded.payload_json,
                    error_type=excluded.error_type,
                    error_msg=excluded.error_msg,
                    traceback=excluded.traceback,
                    duration_ms=excluded.duration_ms;
                """,
                (
                    self.experiment_id,
                    step_id,
                    ns,
                    program,
                    step,
                    key,
                    rep,
                    status,
                    updated_at,
                    payload_json,
                    error_type,
                    error_msg,
                    trace,
                    duration_ms,
                ),
            )
            await db.commit()

    async def run_cached(
        self,
        *,
        step_id: str,
        key: Key,
        rep: int,
        fn: Callable[[], Awaitable[JsonDict]],
    ) -> JsonDict:
        await self._ensure_initialized()
        key = normalize_key(key)
        ns, program, step = _split_step_id(step_id)
        lock = await self._get_run_lock((step_id, key, rep))

        async with lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys=ON;")
                cursor = await db.execute(
                    """
                    SELECT status, payload_json
                    FROM records
                    WHERE experiment=? AND step_id=? AND key=? AND rep=?;
                    """,
                    (self.experiment_id, step_id, key, rep),
                )
                row = await cursor.fetchone()
                if row is not None:
                    status, payload_json = row
                    if status == "success":
                        return json.loads(payload_json)

            started = dt.datetime.now(dt.timezone.utc)
            try:
                payload = await fn()
                if not isinstance(payload, dict):
                    raise TypeError(f"Step {step_id} must return dict, got {type(payload)}")
                payload_json = json.dumps(payload, ensure_ascii=False)
                finished = dt.datetime.now(dt.timezone.utc)
                duration_ms = int((finished - started).total_seconds() * 1000)

                await self._upsert_record(
                    step_id=step_id,
                    key=key,
                    rep=rep,
                    status="success",
                    updated_at=_utc_now_iso(),
                    ns=ns,
                    program=program,
                    step=step,
                    duration_ms=duration_ms,
                    payload_json=payload_json,
                )
                return payload

            except Exception as e:
                finished = dt.datetime.now(dt.timezone.utc)
                duration_ms = int((finished - started).total_seconds() * 1000)
                error_type = type(e).__name__
                error_msg = str(e)
                trace = tb.format_exc()

                await self._upsert_record(
                    step_id=step_id,
                    key=key,
                    rep=rep,
                    status="error",
                    updated_at=_utc_now_iso(),
                    ns=ns,
                    program=program,
                    step=step,
                    duration_ms=duration_ms,
                    error_type=error_type,
                    error_msg=error_msg,
                    trace=trace,
                )
                raise

    async def records_df(
        self,
        *,
        program: str | None = None,
        step: str | None = None,
        status: Literal["success", "error"] | None = None,
        rep: int | None = None,
    ):
        await self._ensure_initialized()
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise RuntimeError("pandas is required for records_df/errors_df") from e

        clauses = ["experiment = ?"]
        params: list[Any] = [self.experiment_id]
        if program is not None:
            clauses.append("program = ?")
            params.append(program)
        if step is not None:
            clauses.append("step = ?")
            params.append(step)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if rep is not None:
            clauses.append("rep = ?")
            params.append(rep)

        where = " AND ".join(clauses)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys=ON;")
            rows = await db.execute_fetchall(
                f"""
                SELECT
                    experiment, namespace, program, step, step_id, key, rep,
                    status, updated_at, attempts, duration_ms,
                    payload_json, error_type, error_msg, traceback
                FROM records
                WHERE {where}
                ORDER BY namespace, program, step, rep, key;
                """,
                params,
            )
        cols = [
            "experiment",
            "namespace",
            "program",
            "step",
            "step_id",
            "key",
            "rep",
            "status",
            "updated_at",
            "attempts",
            "duration_ms",
            "payload_json",
            "error_type",
            "error_msg",
            "traceback",
        ]
        return pd.DataFrame(rows, columns=cols)

    async def errors_df(self, *, program: str | None = None, step: str | None = None):
        return await self.records_df(program=program, step=step, status="error")
