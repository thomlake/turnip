from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from .models import ScopeRecord, ScopeStatus


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ExperimentStorage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        await self._create_schema()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def _create_schema(self) -> None:
        conn = self._require_conn()
        await conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scope_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment TEXT NOT NULL,
                stage TEXT NOT NULL,
                key TEXT NOT NULL,
                trial INTEGER NOT NULL,
                scope TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('success', 'error')),
                data_json TEXT,
                exception_type TEXT,
                exception_message TEXT,
                traceback_text TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (experiment) REFERENCES experiments(experiment)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_scope_unique_attempt
                ON scope_calls (experiment, stage, key, trial, scope, attempt);

            CREATE INDEX IF NOT EXISTS idx_scope_lookup
                ON scope_calls (experiment, stage, key, trial, scope, status);

            CREATE INDEX IF NOT EXISTS idx_scope_latest_attempt
                ON scope_calls (experiment, stage, key, trial, scope, attempt DESC);
            """
        )
        await conn.commit()

    async def upsert_experiment(self, experiment: str, config_json: str) -> None:
        conn = self._require_conn()
        now = utc_now_iso()
        await conn.execute(
            """
            INSERT INTO experiments (experiment, config_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(experiment)
            DO UPDATE SET config_json = excluded.config_json, updated_at = excluded.updated_at
            """,
            (experiment, config_json, now, now),
        )
        await conn.commit()

    async def get_latest_scope_record(
        self,
        *,
        experiment: str,
        stage: str,
        key: str,
        trial: int,
        scope: str,
    ) -> ScopeRecord | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT experiment, stage, key, trial, scope, attempt, status, data_json,
                   exception_type, exception_message, traceback_text
            FROM scope_calls
            WHERE experiment = ? AND stage = ? AND key = ? AND trial = ? AND scope = ?
            ORDER BY attempt DESC
            LIMIT 1
            """,
            (experiment, stage, key, trial, scope),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return ScopeRecord(
            experiment=row["experiment"],
            stage=row["stage"],
            key=row["key"],
            trial=row["trial"],
            scope=row["scope"],
            attempt=row["attempt"],
            status=ScopeStatus(row["status"]),
            data_json=row["data_json"],
            exception_type=row["exception_type"],
            exception_message=row["exception_message"],
            traceback_text=row["traceback_text"],
        )

    async def insert_scope_record(
        self,
        *,
        experiment: str,
        stage: str,
        key: str,
        trial: int,
        scope: str,
        attempt: int,
        status: ScopeStatus,
        started_at: str,
        finished_at: str,
        data_json: str | None = None,
        exception_type: str | None = None,
        exception_message: str | None = None,
        traceback_text: str | None = None,
    ) -> None:
        conn = self._require_conn()
        await conn.execute(
            """
            INSERT INTO scope_calls (
                experiment, stage, key, trial, scope, attempt,
                status, data_json, exception_type, exception_message,
                traceback_text, started_at, finished_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment,
                stage,
                key,
                trial,
                scope,
                attempt,
                status.value,
                data_json,
                exception_type,
                exception_message,
                traceback_text,
                started_at,
                finished_at,
            ),
        )
        await conn.commit()

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("storage is not connected")
        return self._conn
