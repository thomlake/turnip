from __future__ import annotations

import asyncio
import json
import os
import urllib.request

from .base import LLMProvider


class TogetherProvider(LLMProvider):
    """Provider for Together.ai models."""

    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.2") -> None:
        self.model = model
        self.api_key = os.environ.get("TOGETHER_API_KEY", "")
        self.url = "https://api.together.xyz/v1/chat/completions"

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
