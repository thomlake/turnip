import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from turnip import ConversationRunner
from turnip.providers.base import LLMProvider
from turnip.storage import LLMResponse


class DummyProvider(LLMProvider):
    def __init__(self):
        self.calls = 0

    async def completion(self, messages, **kwargs: object) -> LLMResponse:
        self.calls += 1
        prompt = messages[-1]["content"]
        return LLMResponse(messages=messages, content=prompt + " response")


class EchoProcessor(ConversationRunner):
    def render_prompt(self, state):
        return state

    def update_state(self, state, response):
        return response

    def stop(self, state):
        # stop after one turn
        return state.endswith("response")


async def run():
    provider = DummyProvider()
    proc = EchoProcessor(provider)
    result = await proc.process("hello")
    assert result == "hello response"
    assert provider.calls == 1


def test_echo_processor():
    asyncio.run(run())
