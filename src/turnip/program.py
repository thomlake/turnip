import asyncio
import contextvars
import functools
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Iterable, Literal, Optional, Sequence, Union

from .storage import Context, normalize_key
from .types import JsonDict, Key, P, R, T

_current_rep: contextvars.ContextVar[int] = contextvars.ContextVar("turnip_program_current_rep", default=0)
_current_key: contextvars.ContextVar[Key | None] = contextvars.ContextVar("turnip_program_current_key", default=None)


@dataclass(frozen=True)
class Outcome(Generic[T]):
    ok: bool
    value: Optional[T] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class Trials(Generic[T]):
    """
    Minimal repetition-aware container.
    - repeat: number of reps in this trial set
    - keys_by_rep[r][i] corresponds to items_by_rep[r][i]
    """

    repeat: int
    keys_by_rep: list[list[Key]]
    items_by_rep: list[list[T]]

    def __post_init__(self) -> None:
        if self.repeat != len(self.items_by_rep) or self.repeat != len(self.keys_by_rep):
            raise ValueError("Trials repeat mismatch with keys/items dimensions.")
        for r in range(self.repeat):
            if len(self.keys_by_rep[r]) != len(self.items_by_rep[r]):
                raise ValueError(f"Trials keys/items length mismatch at rep {r}.")

    def rep(self, r: int) -> list[T]:
        return self.items_by_rep[r]

    def keys(self, r: int) -> list[Key]:
        return self.keys_by_rep[r]

    def iter_reps(self) -> Iterable[tuple[int, list[Key], list[T]]]:
        for r in range(self.repeat):
            yield r, self.keys_by_rep[r], self.items_by_rep[r]


def step(name: str | None = None):
    """
    Decorate an async Program method to be cached as a step.
    The wrapped method returns a JsonDict.
    """

    def deco(fn: Callable[..., Awaitable[JsonDict]]):
        if not inspect.iscoroutinefunction(fn):
            raise TypeError("@step can only decorate async functions")

        step_name = name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(self: "Program[Any, Any]", item: Any, *args: Any, **kwargs: Any) -> JsonDict:
            ctx = self.ctx
            program_name = self.program_name or self.__class__.__name__
            step_id = ctx.make_step_id(program_name, step_name)

            rep = _current_rep.get()
            key_override = _current_key.get()
            key = key_override if key_override is not None else self.key(item)

            async def call() -> JsonDict:
                return await fn(self, item, *args, **kwargs)

            return await ctx.run_cached(step_id=step_id, key=key, rep=rep, fn=call)

        wrapper.__step_name__ = step_name  # type: ignore[attr-defined]
        return wrapper

    return deco


class Program(Generic[P, R]):
    """
    Per-item computation; cached steps are @step-decorated methods returning JsonDict.
    Program.run() orchestrates steps and returns output R.
    """

    program_name: Optional[str] = None

    def __init__(self, ctx: Context):
        self.ctx = ctx

    def key(self, item: P) -> Key:
        raise NotImplementedError

    async def run(self, item: P) -> R:
        raise NotImplementedError

    async def apply(
        self,
        items: Sequence[P] | Trials[P],
        *,
        repeat: int = 1,
        concurrency: int = 50,
        on_error: Literal["raise", "collect"] = "raise",
    ) -> Trials[Union[R, Outcome[R]]]:
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        sem = asyncio.Semaphore(concurrency)

        async def run_one(item: P, *, rep_id: int, key: Key) -> Union[R, Outcome[R]]:
            rep_token = _current_rep.set(rep_id)
            key_token = _current_key.set(key)
            try:
                async with sem:
                    out = await self.run(item)
                return out
            except Exception as e:
                if on_error == "raise":
                    raise
                return Outcome[R](ok=False, value=None, error=f"{type(e).__name__}: {e}")
            finally:
                _current_key.reset(key_token)
                _current_rep.reset(rep_token)

        if isinstance(items, Trials):
            in_repeat = items.repeat
            in_keys_by_rep = items.keys_by_rep
            in_items_by_rep = items.items_by_rep

            if repeat == 1:
                out_repeat = in_repeat
                out_groups: list[list[Union[R, Outcome[R]]]] = []
                out_keys: list[list[Key]] = []
                for in_r in range(in_repeat):
                    tasks = [
                        asyncio.create_task(
                            run_one(
                                in_items_by_rep[in_r][i],
                                rep_id=in_r,
                                key=in_keys_by_rep[in_r][i],
                            )
                        )
                        for i in range(len(in_items_by_rep[in_r]))
                    ]
                    out = await asyncio.gather(*tasks)
                    out_groups.append(list(out))
                    out_keys.append(list(in_keys_by_rep[in_r]))
                return Trials(repeat=out_repeat, keys_by_rep=out_keys, items_by_rep=out_groups)

            out_repeat = in_repeat * repeat
            out_groups = [[] for _ in range(out_repeat)]
            out_keys = [[] for _ in range(out_repeat)]
            for in_r in range(in_repeat):
                for out_r in range(repeat):
                    rep_id = in_r * repeat + out_r
                    tasks = [
                        asyncio.create_task(
                            run_one(in_items_by_rep[in_r][i], rep_id=rep_id, key=in_keys_by_rep[in_r][i])
                        )
                        for i in range(len(in_items_by_rep[in_r]))
                    ]
                    out = await asyncio.gather(*tasks)
                    out_groups[rep_id] = list(out)
                    out_keys[rep_id] = list(in_keys_by_rep[in_r])
            return Trials(repeat=out_repeat, keys_by_rep=out_keys, items_by_rep=out_groups)

        keys = [normalize_key(self.key(item)) for item in items]
        out_groups: list[list[Union[R, Outcome[R]]]] = []
        out_keys: list[list[Key]] = []

        for rep_id in range(repeat):
            tasks = [asyncio.create_task(run_one(items[i], rep_id=rep_id, key=keys[i])) for i in range(len(items))]
            out = await asyncio.gather(*tasks)
            out_groups.append(list(out))
            out_keys.append(list(keys))

        return Trials(repeat=repeat, keys_by_rep=out_keys, items_by_rep=out_groups)
