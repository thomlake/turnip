from __future__ import annotations

import abc
from typing import Any, Optional

from .providers.base import LLMProvider
from .storage import LLMResult, ResultStore


class TurnProcessor(abc.ABC):
    """Base class for processing an instance using an LLM."""

    def __init__(self, provider: LLMProvider, store: Optional[ResultStore] = None) -> None:
        self.provider = provider
        self.store = store

    async def process(self, initial_state: Any) -> Any:
        state = initial_state
        while True:
            prompt = self.render_prompt(state)
            cached = await self._fetch_cached(prompt)
            if cached is not None:
                response = cached.response
            else:
                response = await self.provider.completion(prompt)
                await self._save_result(prompt, response)
            state = self.update_state(state, response)
            if self.stop(state):
                break
        return state

    async def _fetch_cached(self, prompt: str) -> Optional[LLMResult]:
        if self.store is None:
            return None
        return await self.store.fetch(prompt)

    async def _save_result(self, prompt: str, response: str) -> None:
        if self.store is None:
            return
        await self.store.insert(LLMResult(prompt=prompt, response=response))

    @abc.abstractmethod
    def render_prompt(self, state: Any) -> str:
        """Convert the state into a prompt."""

    @abc.abstractmethod
    def update_state(self, state: Any, response: str) -> Any:
        """Update the state given the LLM response."""

    @abc.abstractmethod
    def stop(self, state: Any) -> bool:
        """Return True if processing should stop."""
