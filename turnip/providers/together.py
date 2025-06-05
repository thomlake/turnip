from __future__ import annotations

import asyncio
import json
import os
import urllib.request

from .base import LLMProvider
from ..storage import LLMResponse


class TogetherProvider(LLMProvider):
    """Provider for Together.ai models."""

    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.2") -> None:
        self.model = model
        self.api_key = os.environ.get("TOGETHER_API_KEY", "")
        self.url = "https://api.together.xyz/v1/chat/completions"

    def _post(self, messages: list[dict[str, str]], params: dict[str, object]) -> LLMResponse:
        payload = {"model": self.model, "messages": messages}
        payload.update(params)
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req) as resp:
            result = json.load(resp)
        choice = result["choices"][0]
        message = choice["message"]
        tool_calls = choice.get("tool_calls")
        return LLMResponse(
            messages=messages,
            content=message["content"],
            parameters=params if params else None,
            tool_calls=tool_calls,
        )

    async def completion(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> LLMResponse:
        return await asyncio.to_thread(self._post, messages, kwargs)
