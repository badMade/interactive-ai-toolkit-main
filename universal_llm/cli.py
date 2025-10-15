"""Command line interface for the universal LLM facade."""
from __future__ import annotations

import asyncio
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, List, Optional

import typer


LLM: Any | None = None


app = typer.Typer(
    add_completion=False,
    help=(
        "Interact with models exposed through universal_llm.LLM. Prompts can be "
        "passed as command arguments or piped via standard input."
    ),
)


class CLIError(RuntimeError):
    """Error raised when the CLI cannot complete an operation."""


def _ensure_llm() -> Any:
    global LLM
    if LLM is not None:
        return LLM
    try:
        module = import_module("universal_llm")
    except Exception as exc:  # pragma: no cover - depends on installation state
        raise CLIError(
            "Failed to import universal_llm. Install the package and its optional "
            "facade dependencies."
        ) from exc
    facade = getattr(module, "LLM", None)
    if facade is None:  # pragma: no cover - optional extra missing
        raise CLIError(
            "The universal_llm LLM facade is unavailable. Install the optional "
            "dependencies that provide it or ensure it is importable."
        )
    LLM = facade
    return facade


def _load_config(config_path: Optional[Path]) -> Optional[dict[str, Any]]:
    if config_path is None:
        return None
    if not config_path.exists():
        raise CLIError(f"Configuration file not found: {config_path}")

    content = config_path.read_text(encoding="utf-8")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency missing
            raise CLIError(
                "Failed to parse configuration as JSON and PyYAML is not available "
                "for YAML decoding."
            ) from exc
        try:
            data = yaml.safe_load(content)  # type: ignore[no-untyped-call]
        except Exception as exc:  # pragma: no cover - YAML parse error
            raise CLIError(f"Failed to parse configuration {config_path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise CLIError("Configuration must evaluate to an object mapping.")
    return data


def _create_llm(config_path: Optional[Path]) -> Any:
    facade = _ensure_llm()
    config = _load_config(config_path)
    try:
        return facade(config=config)
    except Exception as exc:  # pragma: no cover - depends on facade implementation
        _raise_cli_error(exc)
        raise typer.Exit(1)  # unreachable, but keeps type checkers satisfied


def _require_model(model: Optional[str]) -> str:
    if not model:
        raise typer.BadParameter("--model is required for this command")
    return model


def _read_text(value: Optional[str]) -> str:
    if value and value != "-":
        return value
    data = sys.stdin.read()
    if not data.strip():
        raise typer.BadParameter("Provide a prompt argument or pipe content via stdin.")
    return data.strip()


def _message_to_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        fragments: List[str] = []
        for part in content:
            data = getattr(part, "data", None)
            if isinstance(part, dict):
                data = part.get("data")
            if isinstance(data, dict):
                text = data.get("text")
                if text:
                    fragments.append(str(text))
                    continue
            fragments.append(str(data if data is not None else getattr(part, "text", part)))
        return "\n".join(fragment for fragment in fragments if fragment)
    return str(content)


def _response_to_jsonable(response: Any) -> Any:
    if hasattr(response, "model_dump"):
        return response.model_dump()  # type: ignore[attr-defined]
    if hasattr(response, "dict"):
        return response.dict()  # type: ignore[attr-defined]
    if isinstance(response, (list, tuple)):
        return [_response_to_jsonable(item) for item in response]
    return response


def _render_response(response: Any, *, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(_response_to_jsonable(response), indent=2, ensure_ascii=False))
        return
    if hasattr(response, "message"):
        typer.echo(_message_to_text(getattr(response, "message")))
        return
    typer.echo(str(response))


def _render_stream(stream: Iterable[Any], *, json_output: bool) -> None:
    if json_output:
        payload = [_response_to_jsonable(chunk) for chunk in stream]
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    for chunk in stream:
        if hasattr(chunk, "message"):
            typer.echo(_message_to_text(getattr(chunk, "message")), nl=False)
        else:
            typer.echo(str(chunk), nl=False)
    typer.echo()


async def _render_async_stream(stream: Any, *, json_output: bool) -> None:
    if json_output:
        payload: List[Any] = []
        async for chunk in stream:
            payload.append(_response_to_jsonable(chunk))
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    async for chunk in stream:
        if hasattr(chunk, "message"):
            typer.echo(_message_to_text(getattr(chunk, "message")), nl=False)
        else:
            typer.echo(str(chunk), nl=False)
    typer.echo()


def _raise_cli_error(exc: Exception) -> None:
    message = str(exc).strip() or exc.__class__.__name__
    lower = message.lower()
    if any(keyword in lower for keyword in ("credential", "api key", "apikey", "token")):
        message = (
            f"{message}\n"
            "Provide credentials through environment variables or a configuration file passed with --config."
        )
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(1)


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="Prompt text or '-' to read from stdin."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Provider override."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model identifier."),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", "-t", help="Sampling temperature forwarded to the LLM."),
    json_output: bool = typer.Option(False, "--json/--no-json", help="Emit JSON instead of plain text."),
) -> None:
    llm = _create_llm(config)
    prompt_text = _read_text(prompt)
    try:
        response = llm.chat(
            model=_require_model(model),
            messages=[{"role": "user", "content": prompt_text}],
            provider=provider,
            temperature=temperature,
            json_output=json_output,
        )
    except Exception as exc:  # pragma: no cover - provider specific
        _raise_cli_error(exc)
    _render_response(response, json_output=json_output)


@app.command()
def stream(
    prompt: str = typer.Argument(..., help="Prompt text or '-' to read from stdin."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Provider override."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model identifier."),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", "-t", help="Sampling temperature forwarded to the LLM."),
    json_output: bool = typer.Option(False, "--json/--no-json", help="Emit JSON instead of plain text."),
) -> None:
    llm = _create_llm(config)
    prompt_text = _read_text(prompt)
    messages = [{"role": "user", "content": prompt_text}]
    try:
        if hasattr(llm, "stream"):
            stream_iter = llm.stream(
                model=_require_model(model),
                messages=messages,
                provider=provider,
                temperature=temperature,
                json_output=json_output,
            )
            _render_stream(stream_iter, json_output=json_output)
            return
        if hasattr(llm, "astream"):
            async_stream = llm.astream(
                model=_require_model(model),
                messages=messages,
                provider=provider,
                temperature=temperature,
                json_output=json_output,
            )
            asyncio.run(_render_async_stream(async_stream, json_output=json_output))
            return
        raise CLIError("Streaming is not supported by the configured LLM facade.")
    except CLIError as exc:
        _raise_cli_error(exc)
    except Exception as exc:  # pragma: no cover - provider specific
        _raise_cli_error(exc)


@app.command()
def tools(
    prompt: str = typer.Argument(..., help="Prompt text or '-' to read from stdin."),
    tools_path: Path = typer.Option(..., "--tools", "-T", exists=True, readable=True, help="JSON tool specification."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Provider override."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model identifier."),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", "-t", help="Sampling temperature forwarded to the LLM."),
    json_output: bool = typer.Option(False, "--json/--no-json", help="Emit JSON instead of plain text."),
) -> None:
    llm = _create_llm(config)
    prompt_text = _read_text(prompt)
    try:
        with tools_path.open("r", encoding="utf-8") as handle:
            tools_spec = json.load(handle)
    except json.JSONDecodeError as exc:
        _raise_cli_error(CLIError(f"Failed to parse tool specification {tools_path}: {exc}"))
        return

    try:
        response = llm.chat(
            model=_require_model(model),
            messages=[{"role": "user", "content": prompt_text}],
            provider=provider,
            temperature=temperature,
            tools=tools_spec,
            json_output=json_output,
        )
    except Exception as exc:  # pragma: no cover
        _raise_cli_error(exc)
    _render_response(response, json_output=json_output)


@app.command()
def embed(
    texts: Optional[List[str]] = typer.Argument(None, metavar="[TEXT]...", help="Texts to embed; use '-' or stdin."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Provider override."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Embedding model identifier."),
    json_output: bool = typer.Option(True, "--json/--no-json", help="Emit JSON (recommended for embeddings)."),
) -> None:
    llm = _create_llm(config)
    payload: List[str] = [value for value in texts if value != "-"] if texts else []
    if not payload:
        payload.append(_read_text(None))

    try:
        response = llm.embed(
            model=_require_model(model),
            inputs=payload,
            provider=provider,
        )
    except Exception as exc:  # pragma: no cover
        _raise_cli_error(exc)
    _render_response(response, json_output=json_output)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
