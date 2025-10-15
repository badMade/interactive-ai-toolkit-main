"""Command line interface for interacting with the Universal LLM facade."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import typer

try:  # pragma: no cover - optional dependency injection point
    from .facade import LLM  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback when facade is missing
    LLM = None  # type: ignore[assignment]

app = typer.Typer(
    add_completion=False,
    help=(
        "Interact with language models configured through the Universal LLM "
        "facade. Pass prompts directly on the command line or via standard "
        "input."
    ),
)


def _ensure_llm_available() -> type:
    """Return the LLM facade or exit with a helpful error message."""

    if LLM is None:  # pragma: no cover - depends on optional extra
        typer.echo(
            "The universal_llm.facade module is not available. Install the "
            "package that provides the LLM facade or ensure it is on PYTHONPATH.",
            err=True,
        )
        raise typer.Exit(1)
    return LLM


def _instantiate_llm(
    *,
    config: Path | None,
    provider: str | None,
    model: str | None,
    temperature: float | None,
):
    """Create an ``LLM`` instance using the provided CLI options."""

    facade = _ensure_llm_available()
    kwargs: dict[str, Any] = {}
    if config is not None:
        kwargs["config_path"] = config
    if provider is not None:
        kwargs["provider"] = provider
    if model is not None:
        kwargs["model"] = model
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        return facade(**kwargs)
    except FileNotFoundError as exc:
        missing = config if config is not None else exc.filename
        typer.echo(f"Configuration file not found: {missing}", err=True)
        raise typer.Exit(1) from exc
    except TypeError as exc:
        if config is not None and hasattr(facade, "from_config"):
            try:
                return facade.from_config(  # type: ignore[attr-defined]
                    config_path=config,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                )
            except FileNotFoundError as exc:
                typer.echo(f"Configuration file not found: {config}", err=True)
                raise typer.Exit(1) from exc
            except Exception as exc:  # pragma: no cover - facade specific
                _raise_cli_error(exc)
        _raise_cli_error(exc)
    except Exception as exc:  # pragma: no cover - facade specific
        _raise_cli_error(exc)
    raise typer.Exit(1)


def _read_text(value: str | None) -> str:
    """Resolve a prompt argument, reading stdin when needed."""

    if value and value != "-":
        return value
    data = sys.stdin.read().strip()
    if not data:
        raise typer.BadParameter("Provide a prompt argument or pipe text via stdin.")
    return data


def _raise_cli_error(exc: Exception) -> None:
    """Emit a friendly error message and exit the CLI."""

    message = str(exc).strip() or exc.__class__.__name__
    lower_message = message.lower()
    if any(keyword in lower_message for keyword in ("credential", "api key", "apikey", "token")):
        message = (
            f"{message}\nProvide the required credentials via environment variables "
            "or a configuration file passed with --config."
        )
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(1) from exc


def _render_response(response: Any, *, json_output: bool) -> None:
    """Pretty-print responses for text or JSON output modes."""

    if json_output:
        typer.echo(json.dumps(response, indent=2, ensure_ascii=False))
        return
    if isinstance(response, (dict, list)):
        typer.echo(json.dumps(response, indent=2, ensure_ascii=False))
    else:
        typer.echo(str(response))


def _consume_stream(chunks: Iterable[Any], *, json_output: bool) -> None:
    """Stream incremental results to stdout."""

    if json_output:
        collected: list[Any] = []
        for chunk in chunks:
            if chunk is None:
                continue
            collected.append(chunk)
        typer.echo(json.dumps(collected, indent=2, ensure_ascii=False))
        return
    for chunk in chunks:
        if not chunk:
            continue
        typer.echo(str(chunk), nl=False)
    typer.echo()


async def _consume_async_stream(chunks: Any, *, json_output: bool) -> None:
    """Consume an asynchronous stream of results."""

    if json_output:
        collected: list[Any] = []
        async for chunk in chunks:
            if chunk is None:
                continue
            collected.append(chunk)
        typer.echo(json.dumps(collected, indent=2, ensure_ascii=False))
        return
    async for chunk in chunks:
        if not chunk:
            continue
        typer.echo(str(chunk), nl=False)
    typer.echo()


def _shared_options(
    config: Path | None,
    provider: str | None,
    model: str | None,
    temperature: float | None,
) -> dict[str, Any]:
    """Bundle constructor options for reuse."""

    return {
        "config": config,
        "provider": provider,
        "model": model,
        "temperature": temperature,
    }


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="User prompt or '-' to read from stdin."),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a configuration file understood by the LLM facade.",
    ),
    provider: str | None = typer.Option(None, "--provider", "-p", help="Provider alias to select."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model identifier to use."),
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        "-t",
        help="Sampling temperature forwarded to the provider.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Emit JSON instead of plain text.",
    ),
) -> None:
    """Generate a single non-streaming response from the model."""

    llm = _instantiate_llm(**_shared_options(config, provider, model, temperature))
    messages = [{"role": "user", "content": _read_text(prompt)}]
    try:
        response = llm.chat(messages=messages, json_output=json_output)
    except Exception as exc:  # pragma: no cover - facade specific
        _raise_cli_error(exc)
    _render_response(response, json_output=json_output)


@app.command()
def stream(
    prompt: str = typer.Argument(..., help="User prompt or '-' to read from stdin."),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a configuration file understood by the LLM facade.",
    ),
    provider: str | None = typer.Option(None, "--provider", "-p", help="Provider alias to select."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model identifier to use."),
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        "-t",
        help="Sampling temperature forwarded to the provider.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Emit JSON instead of plain text.",
    ),
) -> None:
    """Stream a response from the model, printing tokens as they arrive."""

    llm = _instantiate_llm(**_shared_options(config, provider, model, temperature))
    messages = [{"role": "user", "content": _read_text(prompt)}]
    try:
        if hasattr(llm, "stream"):
            _consume_stream(llm.stream(messages=messages, json_output=json_output), json_output=json_output)
            return
        if hasattr(llm, "astream"):
            asyncio.run(
                _consume_async_stream(
                    llm.astream(messages=messages, json_output=json_output),
                    json_output=json_output,
                )
            )
            return
        typer.echo("Streaming is not supported by the selected provider.", err=True)
        raise typer.Exit(1)
    except Exception as exc:  # pragma: no cover - facade specific
        _raise_cli_error(exc)


@app.command()
def tools(
    prompt: str = typer.Argument(..., help="User prompt or '-' to read from stdin."),
    tools_path: Path = typer.Option(
        ..., "--tools", "-T", help="Path to a JSON tool specification.", exists=True, readable=True
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a configuration file understood by the LLM facade.",
    ),
    provider: str | None = typer.Option(None, "--provider", "-p", help="Provider alias to select."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model identifier to use."),
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        "-t",
        help="Sampling temperature forwarded to the provider.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Emit JSON instead of plain text.",
    ),
) -> None:
    """Invoke the model with a tool specification loaded from disk."""

    try:
        with tools_path.open("r", encoding="utf-8") as handle:
            tools_spec = json.load(handle)
    except FileNotFoundError as exc:
        typer.echo(f"Tool specification file not found: {tools_path}", err=True)
        raise typer.Exit(1) from exc
    except json.JSONDecodeError as exc:
        typer.echo(
            f"Failed to parse tool specification {tools_path}: {exc}",
            err=True,
        )
        raise typer.Exit(1) from exc

    llm = _instantiate_llm(**_shared_options(config, provider, model, temperature))
    messages = [{"role": "user", "content": _read_text(prompt)}]
    try:
        response = llm.chat(messages=messages, tools=tools_spec, json_output=json_output)
    except Exception as exc:  # pragma: no cover - facade specific
        _raise_cli_error(exc)
    _render_response(response, json_output=json_output)


@app.command()
def embed(
    texts: list[str] = typer.Argument(
        None,  # type: ignore[arg-type]
        help="Texts to embed. Use '-' or omit to read from stdin.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a configuration file understood by the LLM facade.",
    ),
    provider: str | None = typer.Option(None, "--provider", "-p", help="Provider alias to select."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model identifier to use."),
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        "-t",
        help="Sampling temperature forwarded to the provider.",
    ),
    json_output: bool = typer.Option(
        True,
        "--json/--no-json",
        help="Emit JSON (recommended for embeddings).",
    ),
) -> None:
    """Generate embeddings for one or more texts."""

    entries: list[str] = []
    if texts:
        for value in texts:
            if value == "-":
                continue
            entries.append(value)
    if not entries:
        entries = [_read_text(None)]

    llm = _instantiate_llm(**_shared_options(config, provider, model, temperature))
    try:
        response = llm.embed(texts=entries)
    except Exception as exc:  # pragma: no cover - facade specific
        _raise_cli_error(exc)
    _render_response(response, json_output=json_output)


def main() -> None:
    """Entry-point compatible with ``python -m universal_llm.cli``."""

    app()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
