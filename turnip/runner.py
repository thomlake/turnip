from __future__ import annotations

import abc
import hashlib
import json
from typing import Any, Optional

from .providers.base import LLMProvider
from .storage import (
    CacheStore,
    LLMResult,
    LLMResponse,
    ResultStore,
)


class ConversationRunner(abc.ABC):
    """Base class for processing an instance using an LLM."""

    def __init__(
        self,
        provider: LLMProvider,
        store: Optional[ResultStore] = None,
        cache: Optional[CacheStore] = None,
    ) -> None:
        self.provider = provider
        self.store = store
        self.cache = cache

    async def process(
        self,
        initial_state: Any,
        *,
        project: str = "default",
        experiment: str = "default",
        run: str = "default",
        instance: str = "default",
        parameters: Optional[dict[str, Any]] = None,
    ) -> Any:
        state = initial_state
        messages: list[dict[str, str]] = []
        turn = 0
        while True:
            prompt = self.render_prompt(state)
            messages.append({"role": "user", "content": prompt})
            cache_key = self._cache_key(messages, parameters)
            cached_resp = (
                await self.cache.fetch(cache_key) if self.cache is not None else None
            )
            if cached_resp is None:
                llm_response = await self.provider.completion(
                    messages,
                    **(parameters or {}),
                )
                if self.cache is not None:
                    await self.cache.insert(cache_key, llm_response)
            else:
                llm_response = cached_resp
            if self.store is not None:
                await self.store.insert(
                    LLMResult(
                        project=project,
                        experiment=experiment,
                        run=run,
                        instance=instance,
                        turn=turn,
                        cache_key=cache_key,
                        response=llm_response,
                    )
                )
            messages.append({"role": "assistant", "content": llm_response.content})
            state = self.update_state(state, llm_response.content)
            if self.stop(state):
                break
            turn += 1
        return state

    def _cache_key(
        self, messages: list[dict[str, str]], parameters: Optional[dict[str, Any]]
    ) -> str:
        data = {"messages": messages, "parameters": parameters or {}}
        payload = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()



    @abc.abstractmethod
    def render_prompt(self, state: Any) -> str:
        """Convert the state into a prompt."""

    @abc.abstractmethod
    def update_state(self, state: Any, response: str) -> Any:
        """Update the state given the LLM response."""

    @abc.abstractmethod
    def stop(self, state: Any) -> bool:
        """Return True if processing should stop."""
