#!/usr/bin/env python3
"""Convenience wrapper to launch the Inclusive AI Toolkit CLI."""
from __future__ import annotations

import importlib
from types import ModuleType


def load_transcribe_module() -> ModuleType:
    """Safely load the ``transcribe`` module that powers the CLI."""
    try:
        return importlib.import_module("transcribe")
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "Could not import the 'transcribe' module required to "
            "launch the program."
        ) from exc


def main() -> None:
    """Import and execute the project's primary command-line entry point."""
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
