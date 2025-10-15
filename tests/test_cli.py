from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from universal_llm import cli


runner = CliRunner()


class DummyResponse:
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)

    def model_dump(self) -> dict[str, object]:  # pragma: no cover - simple serialization helper
        return {"message": {"content": self.message.content}}


def test_chat_invokes_llm(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummyLLM:
        def __init__(self, config=None):
            captured["config"] = config

        def chat(self, **kwargs):
            captured["chat"] = kwargs
            return DummyResponse("Hello world")

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"providers": {"openai": {"api_key": "test"}}}))

    monkeypatch.setattr(cli, "LLM", DummyLLM)

    result = runner.invoke(
        cli.app,
        [
            "chat",
            "Hello",
            "--model",
            "gpt-4o",
            "--config",
            str(config_path),
            "--provider",
            "openai",
            "--temperature",
            "0.2",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["config"] == json.loads(config_path.read_text())
    assert captured["chat"]["provider"] == "openai"
    assert captured["chat"]["temperature"] == 0.2
    assert captured["chat"]["messages"][0]["content"] == "Hello"
    payload = json.loads(result.output)
    assert payload["message"]["content"] == "Hello world"


def test_stream_uses_sync_iterator(monkeypatch):
    outputs = []

    class DummyLLM:
        def __init__(self, config=None):
            pass

        def stream(self, **kwargs):
            outputs.append(kwargs)
            yield DummyResponse("A")
            yield DummyResponse("B")

    monkeypatch.setattr(cli, "LLM", DummyLLM)

    result = runner.invoke(
        cli.app,
        ["stream", "Prompt", "--model", "gpt-4o"],
    )

    assert result.exit_code == 0, result.output
    assert outputs[0]["messages"][0]["content"] == "Prompt"
    assert result.output.strip() == "AB"


def test_tools_loads_spec(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummyLLM:
        def __init__(self, config=None):
            pass

        def chat(self, **kwargs):
            captured.update(kwargs)
            return DummyResponse("tool-call")

    tools_path = tmp_path / "tools.json"
    tools_path.write_text(json.dumps({"tools": []}))

    monkeypatch.setattr(cli, "LLM", DummyLLM)

    result = runner.invoke(
        cli.app,
        [
            "tools",
            "Prompt",
            "--model",
            "gpt-4o",
            "--tools",
            str(tools_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["tools"] == {"tools": []}
    assert json.loads(result.output)["message"]["content"] == "tool-call"


def test_embed_defaults_to_json(monkeypatch):
    class DummyLLM:
        def __init__(self, config=None):
            pass

        def embed(self, **kwargs):
            assert kwargs["inputs"] == ["text"]
            return DummyResponse("embedded")

    monkeypatch.setattr(cli, "LLM", DummyLLM)

    result = runner.invoke(
        cli.app,
        ["embed", "text", "--model", "text-embedding"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["message"]["content"] == "embedded"


@pytest.mark.parametrize("command", ["chat", "stream", "tools", "embed"])
def test_missing_model_errors(monkeypatch, tmp_path, command):
    class DummyLLM:
        def __init__(self, config=None):
            pass

        def chat(self, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("chat should not be invoked")

        def stream(self, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("stream should not be invoked")

        def embed(self, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("embed should not be invoked")

    monkeypatch.setattr(cli, "LLM", DummyLLM)

    args = [command, "prompt"]
    if command == "tools":
        tools_path = tmp_path / "tools.json"
        tools_path.write_text("{}")
        args.extend(["--tools", str(tools_path)])
    result = runner.invoke(cli.app, args)
    assert result.exit_code != 0
    assert "--model is required" in result.output


def test_credentials_error_is_helpful(monkeypatch):
    class DummyLLM:
        def __init__(self, config=None):
            pass

        def chat(self, **kwargs):
            raise RuntimeError("Missing API key")

    monkeypatch.setattr(cli, "LLM", DummyLLM)

    result = runner.invoke(
        cli.app,
        ["chat", "prompt", "--model", "gpt-4o"],
    )

    assert result.exit_code != 0
    assert "Provide credentials" in result.output
