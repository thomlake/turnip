from __future__ import annotations

import abc
from typing import Any, Dict

class LLMProvider(abc.ABC):
    """Abstract base class for asynchronous LLM providers."""

    @abc.abstractmethod
    async def completion(self, prompt: str, **kwargs: Any) -> str:
        """Return a completion for the given prompt."""
        raise NotImplementedError
