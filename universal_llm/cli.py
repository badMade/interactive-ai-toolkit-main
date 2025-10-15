"""Command line interface for the universal LLM facade."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import typer

from . import LLM
from .core import Message

app = typer.Typer(help="Interact with multiple LLM providers through a single CLI.")


def _load_config(path: Optional[Path]) -> Optional[Dict[str, object]]:
    """Load an optional configuration file in JSON or YAML format."""

    if path is None:
        return None
    if not path.exists():
        raise typer.BadParameter(f"Configuration file '{path}' does not exist.")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency guard
            raise typer.BadParameter(
                "PyYAML is required to read YAML configuration files. Install with 'pip install "
                "universal-llm[yaml]'."
            ) from exc
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise typer.BadParameter("Configuration root must be a mapping of provider settings.")
    return loaded


def _build_messages(user_prompt: str, system_prompt: Optional[str]) -> List[Message]:
    messages: List[Message] = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    messages.append(Message(role="user", content=user_prompt))
    return messages


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="User message to send to the model."),
    model: str = typer.Option(..., "--model", "-m", help="Model identifier to target."),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Force a specific provider instead of auto-selection."
    ),
    system: Optional[str] = typer.Option(
        None, "--system", help="Optional system message prepended to the conversation."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to a JSON or YAML configuration file."
    ),
    show_usage: bool = typer.Option(
        False, "--show-usage", help="Display token usage metadata when available."
    ),
) -> None:
    """Send a single prompt to the configured model and print the response."""

    resolved_config = _load_config(config)
    client = LLM(config=resolved_config)
    messages = _build_messages(prompt, system)
    response = client.chat(model=model, messages=messages, provider=provider)
    typer.echo(response.message.content)
    if show_usage and response.usage:
        typer.echo()
        typer.echo("Usage: " + json.dumps(response.usage, indent=2))


if __name__ == "__main__":
    app()
