from __future__ import annotations

import pytest

import run


def test_load_transcribe_module_preserves_dependency_error(monkeypatch) -> None:
    friendly_message = (
        "OpenAI Whisper is not installed. Install it with 'pip install "
        "openai-whisper' or run setup_env.py to configure the environment."
    )

    def fake_import_module(name: str):
        assert name == "transcribe"
        raise ModuleNotFoundError(friendly_message, name="whisper")

    monkeypatch.setattr(run.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        run.load_transcribe_module()

    assert str(exc_info.value) == friendly_message
    assert exc_info.value.name == "whisper"
