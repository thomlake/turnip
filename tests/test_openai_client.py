import asyncio

import pytest

import turnip.openai_client as oc
from turnip.openai_client import OpenAIClient


def test_set_openai_rate_limit_rejects_non_positive():
    with pytest.raises(ValueError, match="max_requests must be > 0"):
        oc.set_openai_rate_limit(0)
    with pytest.raises(ValueError, match="max_requests must be > 0"):
        oc.set_openai_rate_limit(-1)


def test_get_openai_rate_limiter_rejects_non_positive_on_first_call(monkeypatch):
    monkeypatch.setattr(oc, "_OPENAI_RATE_LIMITER", None)
    with pytest.raises(ValueError, match="max_requests must be > 0"):
        oc.get_openai_rate_limiter(0)


def test_extract_retry_after_seconds_handles_valid_and_invalid_headers():
    exc_with_header = Exception("x")
    exc_with_header.response = type("Resp", (), {"headers": {"retry-after": "1.5"}})()
    assert OpenAIClient._extract_retry_after_seconds(exc_with_header) == 1.5

    exc_invalid = Exception("x")
    exc_invalid.response = type("Resp", (), {"headers": {"Retry-After": "abc"}})()
    assert OpenAIClient._extract_retry_after_seconds(exc_invalid) is None


def test_compute_retry_delay_prefers_retry_after_and_caps_backoff():
    client = OpenAIClient.__new__(OpenAIClient)
    client.initial_backoff = 0.5
    client.max_backoff = 2.0

    exc_with_header = Exception("x")
    exc_with_header.response = type("Resp", (), {"headers": {"retry-after": "9"}})()
    assert client._compute_retry_delay(exc_with_header, attempt=0) == 2.0

    exc_no_header = Exception("x")
    exc_no_header.response = type("Resp", (), {"headers": {}})()
    assert client._compute_retry_delay(exc_no_header, attempt=0) == 0.5
    assert client._compute_retry_delay(exc_no_header, attempt=1) == 1.0
    assert client._compute_retry_delay(exc_no_header, attempt=10) == 2.0


def test_set_throttle_and_wait_for_throttle():
    client = OpenAIClient.__new__(OpenAIClient)
    client._throttle_lock = asyncio.Lock()
    client._throttle_until = 0.0

    async def main():
        await client._set_throttle(0.01)
        await client._wait_for_throttle()

    asyncio.run(main())
