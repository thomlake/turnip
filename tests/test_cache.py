import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from turnip import TurnProcessor
from turnip.providers.base import LLMProvider
from turnip.storage import LLMResult


class DummyProvider(LLMProvider):
    def __init__(self):
        self.calls = 0

    async def completion(self, prompt: str, **kwargs: object) -> str:
        self.calls += 1
        return prompt + " response"


class MemoryStore:
    def __init__(self):
        self.data = {}

    async def fetch(self, prompt):
        if prompt in self.data:
            r = self.data[prompt]
            return LLMResult(prompt, r)
        return None

    async def insert(self, result):
        self.data[result.prompt] = result.response


class EchoProcessor(TurnProcessor):
    def render_prompt(self, state):
        return state

    def update_state(self, state, response):
        return response

    def stop(self, state):
        return state.endswith("response")


async def run():
    provider = DummyProvider()
    store = MemoryStore()
    proc = EchoProcessor(provider, store)
    result1 = await proc.process("hello")
    result2 = await proc.process("hello")
    assert result1 == result2 == "hello response"
    assert provider.calls == 1


def test_cache():
    asyncio.run(run())
