"""Tests for the OpenAI provider that run fully offline."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from universal_llm.core import LLMRequest, Message, ToolSpec
from universal_llm.providers.base import ProviderConfig, RetryConfig
from universal_llm.providers.openai import OpenAIProvider


@pytest.fixture()
def openai_provider():
    config = ProviderConfig(
        base_url="https://example.com",
        api_key="secret",
        retry=RetryConfig(attempts=3, min_backoff=0.01, backoff_factor=2.0, max_backoff=1.0, jitter=0.0),
    )
    return OpenAIProvider(config)


def _chat_request() -> LLMRequest:
    return LLMRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="hello")],
        tools=[ToolSpec(name="math", parameters={"type": "object"})],
        extra={"foo": "bar"},
    )


def test_openai_provider_normalizes_tool_calls(openai_provider):
    captured = {}

    def fake_request(method, url, headers=None, params=None, json_body=None):  # pragma: no cover - exercised via test
        captured.update({
            "method": method,
            "url": url,
            "headers": headers,
            "json": json_body,
        })
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Tool call",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "math", "arguments": "{\"x\": 1}"},
                            }
                        ],
                    }
                }
            ],
            "usage": {"total_tokens": 3},
        }

    openai_provider._sync_client = SimpleNamespace(request=fake_request)

    response = openai_provider.chat(_chat_request())

    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["json"]["tools"][0]["function"]["name"] == "math"
    assert response.tool_calls[0].name == "math"
    assert response.tool_calls[0].arguments == "{\"x\": 1}"
    assert response.usage["total_tokens"] == 3


def test_openai_provider_streams_chunked_responses(openai_provider):
    chunks = [
        {"choices": [{"message": {"role": "assistant", "content": "chunk-1"}}]},
        {"choices": [{"message": {"role": "assistant", "content": "chunk-2"}}]},
    ]

    def fake_request(*_, **__):  # pragma: no cover - exercised via test
        return {"chunks": chunks}

    openai_provider._sync_client = SimpleNamespace(request=fake_request)

    responses = list(openai_provider.stream(_chat_request()))

    assert [resp.message.content for resp in responses] == ["chunk-1", "chunk-2"]


def test_openai_provider_retries_with_backoff(monkeypatch, openai_provider):
    attempts = {"count": 0}
    delays: list[float] = []

    def flaky_request(*_, **__):  # pragma: no cover - exercised via test
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("boom")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    openai_provider._sync_client = SimpleNamespace(request=flaky_request)
    monkeypatch.setattr("universal_llm.core._sleep", delays.append)

    response = openai_provider.chat(_chat_request())

    assert attempts["count"] == 3
    assert delays == pytest.approx([0.01, 0.02])
    assert response.message.content == "ok"


def test_openai_provider_embeddings(monkeypatch, openai_provider):
    captured = {}

    def fake_request(method, url, headers=None, params=None, json_body=None):  # pragma: no cover - exercised via test
        captured.update({
            "method": method,
            "url": url,
            "json": json_body,
        })
        return {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "usage": {"total_tokens": 1},
        }

    openai_provider._sync_client = SimpleNamespace(request=fake_request)

    request = LLMRequest(
        model="text-embedding-3-small",
        messages=[Message(role="user", content="hello embeddings")],
    )

    response = openai_provider.embed(request)

    assert captured["url"] == "https://example.com/v1/embeddings"
    assert captured["json"] == {
        "model": "text-embedding-3-small",
        "input": ["hello embeddings"],
    }
    assert response.usage["embeddings"][0]["embedding"] == [0.1, 0.2, 0.3]
