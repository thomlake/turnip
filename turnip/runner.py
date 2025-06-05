from __future__ import annotations

import abc
from typing import Any, Optional

from .providers.base import LLMProvider
from .storage import LLMResult, LLMResponse, ResultStore


class ConversationRunner(abc.ABC):
    """Base class for processing an instance using an LLM."""

    def __init__(self, provider: LLMProvider, store: Optional[ResultStore] = None) -> None:
        self.provider = provider
        self.store = store

    async def process(
        self,
        initial_state: Any,
        *,
        experiment: str = "default",
        run: str = "default",
        instance: str = "default",
        parameters: Optional[dict[str, Any]] = None,
    ) -> Any:
        state = initial_state
        while True:
            prompt = self.render_prompt(state)
            cached = await self._fetch_cached(experiment, run, instance)
            if cached is not None:
                response_text = cached.response.content
            else:
                llm_response = await self.provider.completion(
                    [{"role": "user", "content": prompt}],
                    **(parameters or {}),
                )
                await self._save_result(experiment, run, instance, llm_response)
                response_text = llm_response.content
            state = self.update_state(state, response_text)
            if self.stop(state):
                break
        return state

    async def _fetch_cached(
        self, experiment: str, run: str, instance: str
    ) -> Optional[LLMResult]:
        if self.store is None:
            return None
        return await self.store.fetch(experiment, run, instance)

    async def _save_result(
        self,
        experiment: str,
        run: str,
        instance: str,
        response: LLMResponse,
    ) -> None:
        if self.store is None:
            return
        await self.store.insert(
            LLMResult(
                experiment=experiment,
                run=run,
                instance=instance,
                response=response,
            )
        )

    @abc.abstractmethod
    def render_prompt(self, state: Any) -> str:
        """Convert the state into a prompt."""

    @abc.abstractmethod
    def update_state(self, state: Any, response: str) -> Any:
        """Update the state given the LLM response."""

    @abc.abstractmethod
    def stop(self, state: Any) -> bool:
        """Return True if processing should stop."""
