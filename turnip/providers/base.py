from __future__ import annotations

import abc
from typing import Any, Dict, List

from ..storage import LLMResponse

class LLMProvider(abc.ABC):
    """Abstract base class for asynchronous LLM providers."""

    @abc.abstractmethod
    async def completion(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """Return a completion for the given chat messages."""
        raise NotImplementedError
