from __future__ import annotations

import pytest

import run


def test_load_transcribe_module_reports_missing_whisper(monkeypatch, capsys) -> None:
    friendly_message = (
        "OpenAI Whisper is not installed. Install it with 'pip install "
        "openai-whisper' or run setup_env.py to configure the environment."
    )

    def fake_import_module(name: str):
        assert name == "transcribe"
        raise ModuleNotFoundError(friendly_message, name="whisper")

    monkeypatch.setattr(run.importlib, "import_module", fake_import_module)

    with pytest.raises(SystemExit) as exc_info:
        run.load_transcribe_module()

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == friendly_message
