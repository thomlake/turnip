from __future__ import annotations

import asyncio
import json
import os
import urllib.request

from .base import LLMProvider


class VLLMProvider(LLMProvider):
    """Provider for vLLM server."""

    def __init__(self, url: str = "http://localhost:8000") -> None:
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.model = os.environ.get("VLLM_MODEL", "")
        self.api_key = os.environ.get("VLLM_API_KEY", "")

    def _post(self, prompt: str) -> str:
        data = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req) as resp:
            payload = json.load(resp)
        return payload["choices"][0]["message"]["content"]

    async def completion(self, prompt: str, **kwargs: object) -> str:
        return await asyncio.to_thread(self._post, prompt)
