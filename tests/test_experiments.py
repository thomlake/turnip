from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from turnip import MapExecutionError, MissingRunContextError, Program, scope


def read_scope_rows(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT experiment, stage, key, trial, scope, attempt, status, exception_type
        FROM scope_calls
        ORDER BY key, scope, attempt
        """
    )
    rows = cur.fetchall()
    conn.close()
    return rows


@pytest.mark.asyncio
async def test_scope_cache_hit_skips_execution(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.sqlite3"
    call_count = {"assistant": 0, "user": 0}

    @scope
    async def call_assistant(messages: list[dict]) -> dict:
        call_count["assistant"] += 1
        return {"role": "assistant", "content": f"a-{len(messages)}"}

    @scope
    async def call_sim_user(messages: list[dict]) -> dict:
        call_count["user"] += 1
        return {"role": "user", "content": f"u-{len(messages)}"}

    async def run_user_simulation(item: dict) -> dict:
        sim_user_messages = [{"role": "system", "content": item["user_system"]}]
        assistant_messages = [{"role": "system", "content": item["assistant_system"]}]
        for _ in range(2):
            assistant_message = await call_assistant(assistant_messages)
            assistant_messages.append(assistant_message)
            sim_user_messages.append({"role": "user", "content": assistant_message["content"]})
            user_message = await call_sim_user(sim_user_messages)
            sim_user_messages.append(user_message)
        return {"user": sim_user_messages, "assistant": assistant_messages}

    items = [{"id": "one", "user_system": "u", "assistant_system": "a"}]
    program = Program("demo", {"model": "fake"}, db_path=db_path)

    first = await program.map(run_user_simulation, items, stage="simulate", key="id")
    second = await program.map(run_user_simulation, items, stage="simulate", key="id")

    assert first == second
    assert call_count == {"assistant": 2, "user": 2}

    rows = read_scope_rows(db_path)
    assert len(rows) == 4
    assert all(r[6] == "success" for r in rows)


@pytest.mark.asyncio
async def test_resume_retries_only_failed_scopes(tmp_path: Path) -> None:
    db_path = tmp_path / "resume.sqlite3"
    flaky_state: dict[str, int] = {}
    executed: list[str] = []

    @scope
    async def stable(item: dict) -> dict:
        executed.append(f"stable:{item['id']}")
        return {"ok": item["id"]}

    @scope
    async def flaky(item: dict) -> dict:
        executed.append(f"flaky:{item['id']}")
        count = flaky_state.get(item["id"], 0)
        flaky_state[item["id"]] = count + 1
        if item["id"] == "b" and count == 0:
            raise RuntimeError("transient")
        return {"ok": item["id"]}

    async def workflow(item: dict) -> dict:
        a = await stable(item)
        b = await flaky(item)
        return {"a": a, "b": b}

    items = [{"id": "a"}, {"id": "b"}]
    program = Program("resume-demo", {"model": "fake"}, db_path=db_path)

    with pytest.raises(MapExecutionError):
        await program.map(workflow, items, stage="stage1", key="id", max_concurrency=2)

    # second run should skip previously successful scopes and retry only failed one
    results = await program.map(workflow, items, stage="stage1", key="id", max_concurrency=2)
    assert len(results) == 2

    rows = read_scope_rows(db_path)
    success_rows = [r for r in rows if r[6] == "success"]
    error_rows = [r for r in rows if r[6] == "error"]

    assert len(error_rows) == 1
    assert error_rows[0][2] == "b"
    assert error_rows[0][4].endswith("flaky[0]")
    assert len(success_rows) == 4

    # initial run executes 4 scopes; retry run executes only the previously failed flaky scope
    assert executed.count("stable:a") == 1
    assert executed.count("flaky:a") == 1
    assert executed.count("stable:b") == 1
    assert executed.count("flaky:b") == 2


@pytest.mark.asyncio
async def test_scope_requires_program_context() -> None:
    @scope
    async def some_scope() -> dict:
        return {"ok": True}

    with pytest.raises(MissingRunContextError):
        await some_scope()


@pytest.mark.asyncio
async def test_non_json_serializable_scope_output_is_recorded_as_error(tmp_path: Path) -> None:
    db_path = tmp_path / "serialize.sqlite3"

    @scope
    async def bad_scope(_: dict) -> set:
        return {1, 2, 3}

    async def workflow(item: dict) -> dict:
        await bad_scope(item)
        return {"ok": True}

    program = Program("serialize-demo", {"model": "fake"}, db_path=db_path)

    with pytest.raises(MapExecutionError):
        await program.map(workflow, [{"id": "one"}], stage="stage-ser", key="id")

    rows = read_scope_rows(db_path)
    assert len(rows) == 1
    assert rows[0][6] == "error"
    assert rows[0][7] == "SerializationError"
