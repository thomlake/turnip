# turnip 🫜

Async runtime for define-by-run LLM agent experiments.

## Features

- Async dataset execution via `Program.map(...)`
- Scope-level persistence with `@scope` (SQLite)
- Resume support: reruns skip successful calls and retry failed/missing ones
- High-level identifiers: `experiment`, `stage`, `key`, `trial`
- Low-level identifiers: deterministic scoped call paths (trace-like)

## Install

This repo uses Poetry:

```bash
poetry install
```

## Quickstart

```python
from turnip import Program, scope


@scope
async def call_sim_user(messages: list[dict]) -> dict:
    return {"role": "user", "content": "simulated user reply"}


@scope
async def call_assistant(messages: list[dict]) -> dict:
    return {"role": "assistant", "content": "assistant reply"}


async def run_user_simulation(item: dict, turns: int = 1) -> dict:
    sim_user_messages = [{"role": "system", "content": item["user_system"]}]
    assistant_messages = [
        {"role": "system", "content": item["assistant_system"]},
        {"role": "user", "content": item["sim_user_seed"]},
    ]

    for _ in range(turns):
        assistant_message = await call_assistant(assistant_messages)
        assistant_messages.append(assistant_message)

        sim_user_messages.append({"role": "user", "content": assistant_message["content"]})
        user_message = await call_sim_user(sim_user_messages)
        sim_user_messages.append(user_message)

    return {"user": sim_user_messages, "assistant": assistant_messages}


async def main(items: list[dict]) -> list[dict]:
    program = Program(experiment="demo", config={"model": "gpt-5"})
    return await program.map(
        run_user_simulation,
        items,
        key="id",
        repeat=1,
        max_concurrency=32,
    )
```

## Core API

### `Program`

```python
Program(experiment: str, config: dict[str, Any], *, db_path: str | Path | None = None)
```

- `experiment`: experiment name
- `config`: stored as JSON in the experiment database
- `db_path`: defaults to `.turnip/experiments/{experiment}.sqlite3`

### `Program.map`

```python
await Program.map(
    fn,
    items,
    *,
    repeat: int = 1,
    stage: str | None = None,
    key: str | Callable[[dict[str, Any]], str],
    max_concurrency: int = 32,
)
```

- `repeat`: number of trials per input item (`trial` values are `0..repeat-1`)
- `stage`: logical stage name (defaults to `fn.__name__`)
- `key`: either item field name (like `"id"`) or extractor function
- `max_concurrency`: async task fanout limit

### `@scope`

Decorate async functions whose outputs should be cached at scope level.

Behavior:

- Requires active `Program.map` context
- On call, checks latest stored record for current `(experiment, stage, key, trial, scope)`
- If latest status is success, returns cached JSON result
- Otherwise executes function and stores either:
  - success result JSON, or
  - error metadata (type, message, traceback)

## Resume Semantics

On rerun with the same `experiment + stage + key + trial`:

- successful scope calls are reused
- failed or missing scope calls are re-executed

This allows interrupted/partially failed runs to complete without repeating finished work.

## Storage Model

Each scope call records:

- `experiment`, `stage`, `key`, `trial`
- `scope` (deterministic path like `run_user_simulation[0]/call_assistant[0]`)
- `status` (`success` or `error`)
- `data_json` (for successes)
- `exception_type`, `exception_message`, `traceback_text` (for errors)
- timestamps and `attempt` counter

## Development

Run tests:

```bash
poetry run pytest -q
```
