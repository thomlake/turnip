from __future__ import annotations

import asyncio
import json
import os
import urllib.request

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """Minimal async OpenAI provider using the HTTP API."""

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.url = "https://api.openai.com/v1/chat/completions"

    def _post(self, prompt: str) -> str:
        data = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req) as resp:
            payload = json.load(resp)
        return payload["choices"][0]["message"]["content"]

    async def completion(self, prompt: str, **kwargs: object) -> str:
        return await asyncio.to_thread(self._post, prompt)
