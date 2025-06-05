from __future__ import annotations

import asyncio
import json
import os
import urllib.request

from .base import LLMProvider
from ..storage import LLMResponse


class VLLMProvider(LLMProvider):
    """Provider for vLLM server."""

    def __init__(self, url: str = "http://localhost:8000") -> None:
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.model = os.environ.get("VLLM_MODEL", "")
        self.api_key = os.environ.get("VLLM_API_KEY", "")

    def _post(self, messages: list[dict[str, str]], params: dict[str, object]) -> LLMResponse:
        payload = {"model": self.model, "messages": messages}
        payload.update(params)
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
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
