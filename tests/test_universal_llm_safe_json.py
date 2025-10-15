"""Tests for the :func:`universal_llm.core.safe_json` helper."""
from __future__ import annotations

import json

from universal_llm.core import safe_json


def test_safe_json_retries_with_regex_guardrails(monkeypatch):
    """Ensure non-JSON wrappers are stripped via the retry loop."""

    calls = []
    original_loads = json.loads

    def fake_loads(payload: str):
        calls.append(payload)
        return original_loads(payload)

    monkeypatch.setattr("universal_llm.core.json.loads", fake_loads)

    payload = 'while(1); {"value": 42}'
    result = safe_json(payload)

    assert result == {"value": 42}
    assert calls == [payload, '{"value": 42}']


def test_safe_json_returns_empty_dict_when_retries_exhausted():
    """If no JSON fragment can be recovered, an empty dict is returned."""

    assert safe_json("<not json>") == {}


def test_safe_json_accepts_bytes_input():
    """Binary payloads are decoded before parsing."""

    data = b'{"status": "ok"}'
    assert safe_json(data) == {"status": "ok"}
