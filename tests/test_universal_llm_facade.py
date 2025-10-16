"""Tests exercising the high-level :class:`universal_llm.LLM` facade."""
from __future__ import annotations

from typing import Iterator

from universal_llm import LLM
from universal_llm.core import LLMResponse, Message, ToolSpec


class _StubProvider:
    """In-memory provider that records requests for inspection."""

    def __init__(self, name: str, log: list[tuple[str, object]]) -> None:
        self._name = name
        self._log = log

    def chat(self, request, **kwargs):  # type: ignore[override]
        self._log.append((self._name, request))
        return LLMResponse(message=Message(role="assistant", content=f"{self._name}-reply"))

    def stream(self, request, **kwargs) -> Iterator[LLMResponse]:  # type: ignore[override]
        self._log.append((self._name, request))
        yield LLMResponse(message=Message(role="assistant", content=f"{self._name}-chunk"))

    def embed(self, request, **kwargs):  # type: ignore[override]
        self._log.append((self._name, request))
        return LLMResponse(message=Message(role="assistant", content=f"{self._name}-embedding"))


def _make_llm(monkeypatch, log: list[tuple[str, object]]) -> LLM:
    def fake_instantiate(self, provider_name: str):  # type: ignore[override]
        return _StubProvider(provider_name, log)

    monkeypatch.setattr(LLM, "_instantiate_provider", fake_instantiate, raising=False)
    config = {
        "providers": {
            "openai": {"api_key": "token", "base_url": "https://example.com"},
            "anthropic": {"api_key": "token"},
        }
    }
    return LLM(config=config)


def test_llm_auto_selects_provider_and_normalizes_inputs(monkeypatch):
    log: list[tuple[str, object]] = []
    llm = _make_llm(monkeypatch, log)

    response = llm.chat(
        "claude-sonnet",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"name": "math", "parameters": {"type": "object"}}],
    )

    assert response.message.content == "anthropic-reply"
    provider_name, request = log[0]
    assert provider_name == "anthropic"
    assert isinstance(request.messages[0], Message)
    assert isinstance(request.tools[0], ToolSpec)


def test_llm_explicit_provider_override(monkeypatch):
    log: list[tuple[str, object]] = []
    llm = _make_llm(monkeypatch, log)

    response = llm.chat(
        "claude-sonnet",
        messages=[{"role": "user", "content": "hello"}],
        provider="openai",
    )

    assert response.message.content == "openai-reply"
    provider_name, _ = log[0]
    assert provider_name == "openai"


def test_llm_supports_claude_3_5_sonnet(monkeypatch):
    """Verify that Claude 3.5 Sonnet models are routed to Anthropic provider."""
    log: list[tuple[str, object]] = []
    llm = _make_llm(monkeypatch, log)

    # Test the latest Claude 3.5 Sonnet model
    response = llm.chat(
        "claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Explain quantum computing"}],
    )

    assert response.message.content == "anthropic-reply"
    provider_name, request = log[0]
    assert provider_name == "anthropic"
    assert request.model == "claude-3-5-sonnet-20241022"
