import contextvars
import json
import traceback
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar

from turnip.errors import DataCorruptionError, MissingRunContextError, SerializationError
from turnip.models import ScopeStatus
from turnip.storage import ExperimentStorage, utc_now_iso

T = TypeVar("T")


@dataclass(slots=True)
class RunContext:
    experiment: str
    stage: str
    key: str
    trial: int
    storage: ExperimentStorage
    stack: list[str] = field(default_factory=list)
    counters: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    def current_parent_path(self) -> str:
        return "/".join(self.stack)

    def next_scope_path(self, fn_name: str) -> str:
        parent = self.current_parent_path()
        counter_key = (parent, fn_name)
        index = self.counters[counter_key]
        self.counters[counter_key] += 1
        segment = f"{fn_name}[{index}]"
        return f"{parent}/{segment}" if parent else segment


_RUN_CONTEXT: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar("turnip_run_context", default=None)


def set_run_context(context: RunContext) -> contextvars.Token[RunContext | None]:
    return _RUN_CONTEXT.set(context)


def reset_run_context(token: contextvars.Token[RunContext | None]) -> None:
    _RUN_CONTEXT.reset(token)


def get_run_context() -> RunContext:
    context = _RUN_CONTEXT.get()
    if context is None:
        raise MissingRunContextError("@scope function called without active Program.map context")
    return context


def scope(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    if not callable(fn):
        raise TypeError("scope decorator requires a callable")

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        context = get_run_context()
        scope_path = context.next_scope_path(fn.__name__)

        latest = await context.storage.get_latest_scope_record(
            experiment=context.experiment,
            stage=context.stage,
            key=context.key,
            trial=context.trial,
            scope=scope_path,
        )

        if latest is not None and latest.status == ScopeStatus.SUCCESS:
            if latest.data_json is None:
                raise DataCorruptionError(
                    f"scope '{scope_path}' marked success but has null data (attempt={latest.attempt})"
                )
            try:
                return json.loads(latest.data_json)
            except json.JSONDecodeError as exc:
                raise DataCorruptionError(
                    f"scope '{scope_path}' has invalid cached JSON (attempt={latest.attempt})"
                ) from exc

        next_attempt = 1 if latest is None else latest.attempt + 1
        started_at = utc_now_iso()
        context.stack.append(scope_path.split("/")[-1])
        try:
            result = await fn(*args, **kwargs)
            try:
                data_json = json.dumps(result)
            except (TypeError, ValueError) as exc:
                raise SerializationError(scope_path, result) from exc
            finished_at = utc_now_iso()
            await context.storage.insert_scope_record(
                experiment=context.experiment,
                stage=context.stage,
                key=context.key,
                trial=context.trial,
                scope=scope_path,
                attempt=next_attempt,
                status=ScopeStatus.SUCCESS,
                data_json=data_json,
                started_at=started_at,
                finished_at=finished_at,
            )
            return result
        except Exception as exc:
            finished_at = utc_now_iso()
            await context.storage.insert_scope_record(
                experiment=context.experiment,
                stage=context.stage,
                key=context.key,
                trial=context.trial,
                scope=scope_path,
                attempt=next_attempt,
                status=ScopeStatus.ERROR,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                traceback_text=traceback.format_exc(),
                started_at=started_at,
                finished_at=finished_at,
            )
            raise
        finally:
            context.stack.pop()

    return wrapper
