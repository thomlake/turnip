import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from turnip import ConversationRunner
from turnip.providers.base import LLMProvider
from turnip.storage import LLMResult, LLMResponse


class DummyProvider(LLMProvider):
    def __init__(self):
        self.calls = 0

    async def completion(self, messages, **kwargs: object) -> LLMResponse:
        self.calls += 1
        prompt = messages[-1]["content"]
        return LLMResponse(messages=messages, content=prompt + " response")


class MemoryStore:
    def __init__(self):
        self.data = {}

    async def fetch(self, experiment, run, instance):
        key = (experiment, run, instance)
        return self.data.get(key)

    async def insert(self, result):
        key = (result.experiment, result.run, result.instance)
        self.data[key] = result


class EchoProcessor(ConversationRunner):
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
    result1 = await proc.process("hello", experiment="exp", run="run1", instance="id")
    result2 = await proc.process("hello", experiment="exp", run="run1", instance="id")
    assert result1 == result2 == "hello response"
    assert provider.calls == 1


def test_cache():
    asyncio.run(run())
