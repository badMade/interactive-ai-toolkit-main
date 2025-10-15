"""High-level utilities for interacting with Large Language Models."""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    from .facade import LLM  # type: ignore F401
except ModuleNotFoundError:  # pragma: no cover - facade installed separately
    __all__: list[str] = []
else:  # pragma: no cover - re-export when available
    __all__ = ["LLM"]
