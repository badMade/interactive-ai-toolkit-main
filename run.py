#!/usr/bin/env python3
"""Convenience wrapper to launch the Inclusive AI Toolkit CLI."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parent


def get_project_root() -> Path:
    """Return the directory that contains the repository's entry points."""

    return PROJECT_ROOT


def get_virtualenv_path() -> Path:
    """Return the path where the project's virtual environment resides."""

    return get_project_root() / ".venv"


def get_virtualenv_python_path(venv_path: Path) -> Path:
    """Return the Python interpreter inside the virtual environment."""

    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def get_requirements_script_path() -> Path:
    """Return the helper script responsible for synchronizing dependencies."""

    return get_project_root() / "scripts" / "ensure_requirements.py"


def is_running_inside_virtualenv(venv_path: Path) -> bool:
    """Return ``True`` when the current interpreter belongs to *venv_path*."""

    resolved_venv = venv_path.resolve()
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env and Path(virtual_env).resolve() == resolved_venv:
        return True

    executable_path = Path(sys.executable).resolve()
    try:
        executable_path.relative_to(resolved_venv)
        return True
    except ValueError:
        pass

    prefix_path = Path(sys.prefix).resolve()
    try:
        prefix_path.relative_to(resolved_venv)
        return True
    except ValueError:
        return False


def ensure_virtual_environment() -> None:
    """Guarantee that the CLI executes from the managed virtual environment."""

    venv_path = get_virtualenv_path()
    if is_running_inside_virtualenv(venv_path):
        return

    ensure_script = get_requirements_script_path()
    if not ensure_script.is_file():
        raise RuntimeError(
            "Missing helper script to synchronize project dependencies: "
            f"{ensure_script}"
        )

    try:
        subprocess.run(
            [sys.executable, str(ensure_script)],
            check=True,
            text=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to prepare the project's virtual environment."
        ) from exc

    venv_python = get_virtualenv_python_path(venv_path)
    if not venv_python.exists():
        raise RuntimeError(
            "The virtual environment does not expose a Python interpreter. "
            "Re-run scripts/ensure_requirements.py and try again."
        )

    command = [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_path)
    bin_dir = venv_python.parent
    path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([str(bin_dir), path]) if path else str(bin_dir)

    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=False,
            env=env,
        )
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "Failed to launch the project's virtual environment interpreter."
        ) from exc

    raise SystemExit(result.returncode)


def _missing_whisper_message(exc: ModuleNotFoundError) -> str | None:
    """Return the message to show when Whisper is not installed."""

    message = str(exc).strip()
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, ModuleNotFoundError) and getattr(current, "name", None) == "whisper":
            return message or str(current).strip()
        current = current.__cause__
    return None


def load_transcribe_module() -> ModuleType:
    """Safely load the ``transcribe`` module that powers the CLI."""

    try:
        return importlib.import_module("transcribe")
    except ModuleNotFoundError as exc:
        whisper_message = _missing_whisper_message(exc)
        if whisper_message is not None:
            print(whisper_message, file=sys.stderr)
            raise SystemExit(1) from None
        raise RuntimeError(
            "Could not import the 'transcribe' module required to "
            "launch the program."
        ) from exc


def main() -> None:
    """Import and execute the project's primary command-line entry point."""

    ensure_virtual_environment()
    module = load_transcribe_module()
    try:
        launch = module.main
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "The 'transcribe' module does not expose a callable 'main'."
        ) from exc
    launch()


if __name__ == "__main__":
    main()
