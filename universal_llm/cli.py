"""Typer-powered command line interface for the Universal LLM facade."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

import typer

try:  # pragma: no cover - optional dependency for end users
    from . import LLM
except Exception:  # pragma: no cover - exposed as runtime error in CLI
    LLM = None  # type: ignore[assignment]


app = typer.Typer(
    add_completion=False,
    help=(
        "Interact with Large Language Models through the provider-agnostic "
        "Universal LLM facade. Commands support prompts passed as arguments "
        "or via standard input."
    ),
)


def _load_config(path: Optional[Path]) -> Optional[dict[str, Any]]:
    """Load an optional configuration file in JSON or YAML format."""

    if path is None:
        return None
    if not path.exists():
        raise typer.BadParameter(f"Configuration file '{path}' was not found.")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency guard
            raise typer.BadParameter(
                "PyYAML is required to parse YAML configuration files. "
                "Install it with 'pip install universal-llm[yaml]'."
            ) from exc
        loaded = yaml.safe_load(text) or {}
    else:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Failed to parse configuration JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise typer.BadParameter("Configuration root must be an object mapping provider names to settings.")
    return loaded


def _load_tools(path: Path) -> Any:
    """Load a tool specification from a JSON file."""

    if not path.exists():
        raise typer.BadParameter(f"Tool specification '{path}' does not exist.")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Tool specification is not valid JSON: {exc}") from exc
    return data


def _message_to_text(message: Any) -> str:
    """Convert a response message into printable text."""

    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts: List[str] = []
        for entry in content:
            if isinstance(entry, dict):
                value = entry.get("text") or entry.get("data") or ""
                parts.append(str(value))
            else:
                value = getattr(entry, "data", getattr(entry, "text", ""))
                if isinstance(value, dict):
                    parts.append(str(value.get("text", "")))
                else:
                    parts.append(str(value))
        return "".join(parts).strip()
    return str(content)


def _serialize_response(response: Any) -> Any:
    """Serialize ``LLMResponse``-like objects to built-in Python structures."""

    if hasattr(response, "model_dump"):
        return response.model_dump()  # type: ignore[attr-defined]
    if hasattr(response, "dict"):
        return response.dict()  # type: ignore[attr-defined]
    if hasattr(response, "__dict__"):
        return {key: value for key, value in vars(response).items() if not key.startswith("_")}
    return response


def _ensure_client(config: Optional[Path]) -> Any:
    """Instantiate the ``LLM`` facade, surfacing helpful credential errors."""

    if LLM is None:  # pragma: no cover - depends on packaging extras
        typer.echo(
            "The universal LLM facade is not installed. Install optional providers or "
            "ensure 'universal_llm' is fully configured.",
            err=True,
        )
        raise typer.Exit(1)

    config_data = _load_config(config)
    try:
        return LLM(config=config_data)
    except Exception as exc:  # pragma: no cover - facade specific failures
        _raise_cli_error(exc)
        raise typer.Exit(1)  # Unreachable, satisfies type-checkers


def _raise_cli_error(exc: Exception) -> None:
    """Render an informative error message and abort the command."""

    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if any(keyword in lowered for keyword in ("credential", "api key", "apikey", "token")):
        message = (
            f"{message}\nProvide the required credentials via environment variables or "
            "a configuration file supplied with --config."
        )
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(1) from exc


def _read_prompt(argument: Optional[str]) -> str:
    """Resolve the user prompt either from an argument or standard input."""

    if argument and argument != "-":
        return argument
    data = sys.stdin.read()
    if not data.strip():
        raise typer.BadParameter("Provide a prompt argument or pipe text via standard input.")
    return data.strip()


def _render_response(response: Any, *, json_output: bool) -> None:
    """Print a response as JSON or plain text."""

    if json_output:
        typer.echo(json.dumps(_serialize_response(response), indent=2, ensure_ascii=False))
        return
    message = getattr(response, "message", None)
    if message is not None:
        typer.echo(_message_to_text(message))
        return
    typer.echo(str(response))


def _consume_stream(chunks: Iterable[Any], *, json_output: bool) -> None:
    """Consume a synchronous stream of ``LLMResponse`` objects."""

    if json_output:
        payload = [_serialize_response(chunk) for chunk in chunks]
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    for chunk in chunks:
        message = getattr(chunk, "message", None)
        text = _message_to_text(message) if message is not None else str(chunk)
        if not text:
            continue
        typer.echo(text, nl=False)
    typer.echo()


async def _consume_async_stream(chunks: Any, *, json_output: bool) -> None:
    """Consume an asynchronous stream of ``LLMResponse`` objects."""

    if json_output:
        payload: list[Any] = []
        async for chunk in chunks:
            payload.append(_serialize_response(chunk))
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    async for chunk in chunks:
        message = getattr(chunk, "message", None)
        text = _message_to_text(message) if message is not None else str(chunk)
        if not text:
            continue
        typer.echo(text, nl=False)
    typer.echo()


def _collect_texts(arguments: List[str]) -> List[str]:
    """Collect embedding texts from arguments and standard input."""

    values: List[str] = []
    read_stdin = False
    for value in arguments:
        if value == "-":
            read_stdin = True
        else:
            values.append(value)
    if not values or read_stdin:
        values.append(_read_prompt(None))
    return values


def _shared_options(
    config: Optional[Path],
    provider: Optional[str],
    model: Optional[str],
    temperature: Optional[float],
) -> dict[str, Any]:
    if not model:
        raise typer.BadParameter("--model is required for this command.")
    params: dict[str, Any] = {"model": model}
    if provider:
        params["provider"] = provider
    if temperature is not None:
        params["temperature"] = temperature
    return {"config": config, "parameters": params}


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="Prompt to send to the assistant. Use '-' to read from stdin."),
    *,
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to a JSON or YAML configuration file."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Override the auto-selected provider."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model identifier to target."),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Sampling temperature."),
    json_output: bool = typer.Option(False, "--json/--no-json", help="Emit JSON instead of plain text."),
) -> None:
    """Send a prompt and print the assistant response."""

    options = _shared_options(config, provider, model, temperature)
    client = _ensure_client(options["config"])
    message_payload = [{"role": "user", "content": _read_prompt(prompt)}]
    try:
        response = client.chat(messages=message_payload, **options["parameters"])
    except Exception as exc:  # pragma: no cover - provider specific failure paths
        _raise_cli_error(exc)
    _render_response(response, json_output=json_output)


@app.command()
def stream(
    prompt: str = typer.Argument(..., help="Prompt to stream from the assistant. Use '-' for stdin."),
    *,
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to a JSON or YAML configuration file."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Override the auto-selected provider."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model identifier to target."),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Sampling temperature."),
    json_output: bool = typer.Option(False, "--json/--no-json", help="Emit JSON instead of plain text."),
) -> None:
    """Stream a response, yielding tokens as they arrive."""

    options = _shared_options(config, provider, model, temperature)
    client = _ensure_client(options["config"])
    message_payload = [{"role": "user", "content": _read_prompt(prompt)}]
    try:
        stream_method = getattr(client, "stream", None)
        async_stream_method = getattr(client, "astream", None)
        if callable(stream_method):
            chunks = stream_method(messages=message_payload, **options["parameters"])
            _consume_stream(chunks, json_output=json_output)
            return
        if callable(async_stream_method):
            asyncio.run(
                _consume_async_stream(
                    async_stream_method(messages=message_payload, **options["parameters"]),
                    json_output=json_output,
                )
            )
            return
        raise RuntimeError("Streaming is not supported by the selected provider.")
    except Exception as exc:  # pragma: no cover - provider specific failure paths
        _raise_cli_error(exc)


@app.command()
def tools(
    prompt: str = typer.Argument(..., help="Prompt to send when tools are available. '-' reads stdin."),
    *,
    tools_path: Path = typer.Option(..., "--tools", "-T", help="Path to a JSON tool specification."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to a JSON or YAML configuration file."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Override the auto-selected provider."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model identifier to target."),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Sampling temperature."),
    json_output: bool = typer.Option(True, "--json/--no-json", help="Emit JSON (default for tool invocations)."),
) -> None:
    """Send a prompt with tool definitions loaded from disk."""

    tools_spec = _load_tools(tools_path)
    options = _shared_options(config, provider, model, temperature)
    client = _ensure_client(options["config"])
    message_payload = [{"role": "user", "content": _read_prompt(prompt)}]
    try:
        response = client.chat(messages=message_payload, tools=tools_spec, **options["parameters"])
    except Exception as exc:  # pragma: no cover - provider specific failure paths
        _raise_cli_error(exc)
    _render_response(response, json_output=json_output)


@app.command()
def embed(
    texts: Optional[List[str]] = typer.Argument(None, help="Texts to embed. Use '-' or omit to read from stdin."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to a JSON or YAML configuration file."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Override the auto-selected provider."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model identifier to target."),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Sampling temperature."),
    json_output: bool = typer.Option(True, "--json/--no-json", help="Emit JSON embeddings (recommended)."),
) -> None:
    """Generate embeddings for one or more pieces of text."""

    options = _shared_options(config, provider, model, temperature)
    client = _ensure_client(options["config"])
    entries = _collect_texts(list(texts or []))
    try:
        response = client.embed(inputs=entries, **options["parameters"])
    except Exception as exc:  # pragma: no cover - provider specific failure paths
        _raise_cli_error(exc)
    _render_response(response, json_output=json_output)


def main() -> None:
    """Entry point compatible with ``python -m universal_llm.cli``."""

    app()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
