import asyncio
import sqlite3
from pathlib import Path

from turnip.program import Program, Trials, step
from turnip.storage import Context, Store


class OneStepProgram(Program[dict, dict]):
    def key(self, item: dict) -> str:
        return str(item["id"])

    @step
    async def enrich(self, item: dict) -> dict:
        base = item.get("id", item.get("value"))
        if base is None:
            raise KeyError("id")
        return {"value": base * 2}

    async def run(self, item: dict) -> dict:
        return await self.enrich(item)


def _fetch_records(db_path: Path):
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT key, rep, status, attempts, payload_json, error_type
            FROM records
            ORDER BY key, rep
            """
        ).fetchall()
        return rows
    finally:
        conn.close()


def _fetch_steps(db_path: Path):
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT step
            FROM records
            ORDER BY step
            """
        ).fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


def test_run_cached_dedupes_inflight_calls(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    ctx = Store(str(db_path)).experiment("exp1", config={}).bind(namespace="ns")

    state = {"calls": 0}

    async def run_once():
        async def fn():
            state["calls"] += 1
            await asyncio.sleep(0.05)
            return {"ok": True}

        return await ctx.run_cached(step_id="prog/step", key="k1", rep=0, fn=fn)

    async def main():
        out = await asyncio.gather(run_once(), run_once(), run_once())
        assert all(o == {"ok": True} for o in out)

    asyncio.run(main())
    assert state["calls"] == 1


def test_resume_success_skips_and_error_retries(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    ctx = Store(str(db_path)).experiment("exp2", config={}).bind(namespace="ns")

    state = {"calls": 0, "fail": True}

    async def fn():
        state["calls"] += 1
        if state["fail"]:
            raise ValueError("boom")
        return {"ok": True}

    async def main():
        try:
            await ctx.run_cached(step_id="prog/step", key="k1", rep=0, fn=fn)
        except ValueError:
            pass

        state["fail"] = False
        out = await ctx.run_cached(step_id="prog/step", key="k1", rep=0, fn=fn)
        assert out == {"ok": True}

        # Should hit cache after success.
        out2 = await ctx.run_cached(step_id="prog/step", key="k1", rep=0, fn=fn)
        assert out2 == {"ok": True}

    asyncio.run(main())
    assert state["calls"] == 2
    rows = _fetch_records(db_path)
    assert rows == [("k1", 0, "success", 2, '{"ok": true}', None)]


def test_step_decorator_supports_bare_and_named_forms(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    ctx = Store(str(db_path)).experiment("exp_step_forms", config={}).bind(namespace="ns")

    class DecoratorFormsProgram(Program[dict, dict]):
        def key(self, item: dict) -> str:
            return str(item["id"])

        @step
        async def plain(self, item: dict) -> dict:
            return {"plain": item["id"]}

        @step("custom")
        async def named(self, item: dict) -> dict:
            return {"named": item["id"]}

        async def run(self, item: dict) -> dict:
            left = await self.plain(item)
            right = await self.named(item)
            return {**left, **right}

    prog = DecoratorFormsProgram(ctx)

    async def main():
        trials = await prog.apply([{"id": 1}])
        assert trials.rep(0) == [{"plain": 1, "named": 1}]

    asyncio.run(main())

    assert prog.plain.__step_name__ == "plain"  # type: ignore[attr-defined]
    assert prog.named.__step_name__ == "custom"  # type: ignore[attr-defined]
    assert _fetch_steps(db_path) == ["custom", "plain"]


def test_program_apply_sequence_and_batch_rep_mapping(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    ctx = Store(str(db_path)).experiment("exp3", config={}).bind(namespace="ns")
    prog = OneStepProgram(ctx)

    async def main():
        seq_batch = await prog.apply([{"id": 1}, {"id": 2}], repeat=2)
        assert seq_batch.repeat == 2
        assert seq_batch.keys(0) == ["1", "2"]
        assert seq_batch.keys(1) == ["1", "2"]

        aligned = await prog.apply(seq_batch, repeat=1)
        assert aligned.repeat == 2
        assert aligned.keys(0) == ["1", "2"]
        assert aligned.keys(1) == ["1", "2"]

        cross = await prog.apply(seq_batch, repeat=3)
        assert cross.repeat == 6
        # rep_id = in_rep * repeat + out_rep
        assert cross.keys(0) == ["1", "2"]
        assert cross.keys(2) == ["1", "2"]
        assert cross.keys(3) == ["1", "2"]
        assert cross.keys(5) == ["1", "2"]

    asyncio.run(main())


def test_apply_collect_returns_outcome_without_raising(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    ctx = Store(str(db_path)).experiment("exp4", config={}).bind(namespace="ns")

    class MaybeFail(Program[dict, dict]):
        def key(self, item: dict) -> str:
            return str(item["id"])

        @step
        async def risky(self, item: dict) -> dict:
            if item.get("fail"):
                raise RuntimeError("bad")
            return {"id": item["id"]}

        async def run(self, item: dict) -> dict:
            return await self.risky(item)

    prog = MaybeFail(ctx)

    async def main():
        trials = await prog.apply(
            [{"id": 1}, {"id": 2, "fail": True}],
            on_error="collect",
        )
        assert trials.repeat == 1
        good, bad = trials.rep(0)
        assert good == {"id": 1}
        assert bad.ok is False
        assert "RuntimeError: bad" in (bad.error or "")

    asyncio.run(main())


def test_batch_key_isolation_under_concurrency(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    ctx = Store(str(db_path)).experiment("exp5", config={}).bind(namespace="ns")

    class CoordinatedProgram(Program[dict, dict]):
        def __init__(self, ctx: Context, expected: int):
            super().__init__(ctx)
            self.expected = expected
            self.ready = 0
            self.ready_lock = asyncio.Lock()
            self.all_ready = asyncio.Event()

        def key(self, item: dict) -> str:
            # Intentionally different from trials key override path.
            return f"id-{item['id']}"

        @step
        async def record(self, item: dict) -> dict:
            return {"id": item["id"]}

        async def run(self, item: dict) -> dict:
            async with self.ready_lock:
                self.ready += 1
                if self.ready == self.expected:
                    self.all_ready.set()
            await self.all_ready.wait()
            return await self.record(item)

    items = [{"id": i} for i in range(8)]
    provided_keys = [f"key-{i}" for i in range(8)]
    in_batch = Trials(repeat=1, keys_by_rep=[provided_keys], items_by_rep=[items])
    prog = CoordinatedProgram(ctx, expected=len(items))

    async def main():
        out = await prog.apply(in_batch, concurrency=8)
        assert out.repeat == 1
        assert out.keys(0) == provided_keys

    asyncio.run(main())

    rows = _fetch_records(db_path)
    stored_keys = [r[0] for r in rows]
    assert stored_keys == provided_keys
