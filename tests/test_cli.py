"""Tests for the Typer-based universal_llm CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest
from typer.testing import CliRunner

from universal_llm import cli


runner = CliRunner()


@dataclass
class _DummyMessage:
    role: str
    content: Any


class _DummyResponse:
    """Lightweight stand-in for :class:`LLMResponse`."""

    def __init__(self, content: str, usage: Dict[str, Any] | None = None) -> None:
        self.message = _DummyMessage(role="assistant", content=content)
        self.tool_calls: List[Any] = []
        self.usage = usage or {}
        self.raw: Dict[str, Any] = {"content": content}

    def model_dump(self) -> Dict[str, Any]:  # pragma: no cover - exercised via CLI
        return {
            "message": {"role": self.message.role, "content": self.message.content},
            "usage": self.usage,
        }

    def dict(self) -> Dict[str, Any]:  # pragma: no cover - fallback for pydantic v1
        return self.model_dump()


@pytest.fixture
def dummy_llm(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Monkeypatch ``cli.LLM`` with a deterministic test double."""

    calls: Dict[str, Any] = {}

    class _StubLLM:
        def __init__(self, config: Dict[str, Any] | None = None) -> None:
            calls["config"] = config

        def chat(self, *, messages: List[Dict[str, Any]], tools=None, **params: Any) -> _DummyResponse:
            calls["chat"] = {"messages": messages, "tools": tools, "params": params}
            return _DummyResponse("assistant reply", usage={"prompt_tokens": 10})

        def stream(self, *, messages: List[Dict[str, Any]], **params: Any):
            calls["stream"] = {"messages": messages, "params": params}
            return iter([_DummyResponse("chunk one "), _DummyResponse("chunk two")])

        def embed(self, *, inputs: List[str], **params: Any) -> Dict[str, Any]:
            calls["embed"] = {"inputs": inputs, "params": params}
            return {"embeddings": [[float(index)] for index, _ in enumerate(inputs)]}

    monkeypatch.setattr(cli, "LLM", _StubLLM)
    return calls


def test_chat_plain_text(dummy_llm: Dict[str, Any]) -> None:
    result = runner.invoke(cli.app, ["chat", "Hello", "--model", "test-model"])
    assert result.exit_code == 0
    assert "assistant reply" in result.stdout
    assert dummy_llm["chat"]["messages"][0]["content"] == "Hello"


def test_stream_json(dummy_llm: Dict[str, Any]) -> None:
    result = runner.invoke(
        cli.app,
        ["stream", "Hello", "--model", "test-model", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [chunk["message"]["content"] for chunk in payload] == ["chunk one ", "chunk two"]
    assert dummy_llm["stream"]["params"]["model"] == "test-model"


def test_tools_loads_spec(tmp_path: Path, dummy_llm: Dict[str, Any]) -> None:
    tools_file = tmp_path / "tools.json"
    spec = [{"name": "solve", "description": "Do work"}]
    tools_file.write_text(json.dumps(spec), encoding="utf-8")
    result = runner.invoke(
        cli.app,
        [
            "tools",
            "Plan",
            "--model",
            "test-model",
            "--tools",
            str(tools_file),
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert dummy_llm["chat"]["tools"] == spec


def test_embed_reads_stdin(dummy_llm: Dict[str, Any]) -> None:
    result = runner.invoke(
        cli.app,
        ["embed", "-", "--model", "embedding-model"],
        input="Universal design\n",
    )
    assert result.exit_code == 0
    assert dummy_llm["embed"]["inputs"] == ["Universal design"]
    embeddings = json.loads(result.stdout)
    assert embeddings["embeddings"][0] == [0.0]


def test_chat_missing_credentials_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ErrorLLM:
        def __init__(self, *_, **__):
            pass

        def chat(self, **_):
            raise RuntimeError("OpenAI provider requires an API key")

    monkeypatch.setattr(cli, "LLM", _ErrorLLM)
    result = runner.invoke(cli.app, ["chat", "Hello", "--model", "test-model"])
    assert result.exit_code != 0
    assert "Provide the required credentials" in result.stderr
