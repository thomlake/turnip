# Turnip 🫜

A lightweight storage and execution framework for LLM-style experiments.

Turnip provides:

* Persistent caching of step-level results
* Automatic resume semantics
* First-class repetition (`rep`) support
* Async concurrent execution with concurrency control
* SQLite-backed storage (WAL enabled)
* Minimal abstraction surface (no DAG engine, no heavy framework)

It is designed for iterative LLM / agent workflows where:

* Each item has a stable identity (`key`)
* Work is structured into cached steps
* Experiments may be repeated for variance
* You want reproducibility without orchestration overhead

---

# Core Concepts

## Experiment

An experiment is a named run with associated configuration.

```python
from turnip import Store

store = Store("runs.db")
exp = store.experiment("run_2026_03_03_001", config={"model": "gpt-5"})
ctx = exp.bind(namespace="baseline")
```

* Configuration is stored once per experiment.
* All records are scoped to `(experiment, step_id, key, rep)`.

---

## Program

A `Program` defines a per-item computation.

You implement:

* `key(item)` → stable identity
* `run(item)` → orchestration logic
* `@step` methods → cached units returning JSON-serializable dicts

```python
from turnip import Program, step

class ScoreUserItem(Program[dict, dict]):

    def key(self, item: dict) -> str:
        return (item["user_id"], item["item_id"])

    @step()
    async def score(self, item: dict) -> dict:
        return {"score": 0.87, "rationale": "..."}

    async def run(self, item: dict) -> dict:
        s = await self.score(item)
        return {"user_id": item["user_id"], **s}
```

---

## Step Caching

Each `@step` method is cached under:

```
(experiment, namespace, program_name, step_name, key, rep)
```

Behavior:

* If `status == success` → cached payload is returned
* If `status == error` → step is retried
* On success → payload overwrites previous record
* On error → error + traceback are stored

Resume is automatic. Force requires explicit deletion (not implicit recompute).

---

## Repetition (`rep`)

Repetition is a first-class execution dimension.

```python
trials = await program.apply(items, repeat=5)
```

This produces reps `0..4`.

If you rerun with `repeat=5` and only `rep=0` exists, only `rep=1..4` will execute.

Repetition is stored as a database column:

```
rep INTEGER NOT NULL DEFAULT 0
```

---

## Trials

`apply()` returns `Trials`.

`Trials` preserves repetition alignment and item identity.

```python
trials.repeat                # number of reps
trials.rep(0)                # items for rep 0
trials.iter_reps()           # iterate (rep, keys, items)
```

### Composition

Aligned forwarding:

```python
b1 = await program1.apply(items, repeat=5)
b2 = await program2.apply(b1)  # preserves rep alignment
```

Forward only rep 0:

```python
b2 = await program2.apply(b1.rep(0))
```

Cross product (explicit):

```python
b2 = await program2.apply(b1, repeat=3)
```

---

# Execution Model

`Program.apply()`:

* Supports `Sequence` or `Trials` input
* Controls concurrency via semaphore
* Threads `rep` via `ContextVar`
* Handles error modes:

  * `"raise"` (default)
  * `"collect"` → returns `Outcome`

---

# Storage

SQLite backend (WAL enabled).

Schema (simplified):

```
experiments(
    experiment PRIMARY KEY,
    created_at,
    config_json
)

records(
    experiment,
    step_id,
    namespace,
    program,
    step,
    key,
    rep,
    status,
    attempts,
    payload_json,
    error_type,
    error_msg,
    traceback,
    duration_ms,
    PRIMARY KEY (experiment, step_id, key, rep)
)
```

Guarantees:

* One canonical record per `(experiment, step_id, key, rep)`
* Errors stored until overwritten by success
* Attempts counter increments on each execution

---

# Analysis

Built-in helpers:

```python
df = await ctx.records_df(program="ScoreUserItem")
errors = await ctx.errors_df()
```

Returns a pandas DataFrame with:

* experiment
* namespace
* program
* step
* key
* rep
* status
* payload_json
* error fields
* duration
* attempts

Payload remains JSON; caller can normalize as needed.

---

# Design Philosophy

Turnip intentionally avoids:

* DAG schedulers
* Graph inference
* Magic orchestration
* Framework-level abstractions

Instead:

* Orchestration is plain Python
* Caching is step-scoped
* Repetition is explicit
* Composition is transparent

This keeps the system:

* Easy to reason about
* Easy to debug
* Easy to extend
* Hard to accidentally over-abstract

---

# Example

```python
from turnip import Program, Store, step

class Describe(Program[dict, dict]):
    def key(self, item): return item["id"]

    @step()
    async def describe(self, item):
        return {"desc": f"desc({item['id']})"}

    async def run(self, item):
        d = await self.describe(item)
        return {**item, **d}


async def main():
    store = Store("runs.db")
    ctx = store.experiment("demo", config={}).bind()

    items = [{"id": "A"}, {"id": "B"}]

    trials = await Describe(ctx).apply(items, repeat=3)
    print(trials.rep(0))
```

---

# Dependencies

* Python 3.11+
* `aiosqlite`
* `openai`
* `pandas` (optional, for analysis helpers)

---

# Roadmap (Possible Extensions)

* Delete / force recompute APIs
* Attempt-level history table
* Step-level invalidation
* Deterministic seed injection per rep
* Persistent connection pooling
* DuckDB backend

---

# Why This Exists

Turnip is designed for LLM/agent experiments where:

* Outputs are JSON-like
* Failures need to be inspectable
* Resume must be automatic
* Variance runs are common
* Orchestration should remain explicit

It provides just enough structure to make experimentation disciplined, without turning into a workflow engine.
