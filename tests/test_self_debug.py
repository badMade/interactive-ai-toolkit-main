from __future__ import annotations

from typing import Callable

import pytest

from scripts import self_debug


def _build_which(mapping: dict[str, str]) -> Callable[[str], str | None]:
    def _which(name: str) -> str | None:
        return mapping.get(name)

    return _which


def test_determine_markitdown_launcher_prefers_uvx(tmp_path) -> None:
    uvx_path = str(tmp_path / "uvx")
    which = _build_which({"uvx": uvx_path})

    command = self_debug.determine_markitdown_launcher(which=which, find_spec=lambda _: None)

    assert command == [uvx_path, "markitdown"]


def test_determine_markitdown_launcher_falls_back_to_uv(tmp_path) -> None:
    uv_path = str(tmp_path / "uv")
    which = _build_which({"uv": uv_path})

    command = self_debug.determine_markitdown_launcher(which=which, find_spec=lambda _: None)

    assert command == [uv_path, "tool", "run", "markitdown"]


def test_determine_markitdown_launcher_uses_python_module(monkeypatch, tmp_path) -> None:
    which = _build_which({})
    sentinel = object()

    command = self_debug.determine_markitdown_launcher(
        which=which,
        find_spec=lambda name: sentinel if name == "uv" else None,
    )

    assert command[:3] == [self_debug.sys.executable, "-m", "uv"]
    assert command[-2:] == ["run", "markitdown"]


def test_determine_markitdown_launcher_errors_when_missing() -> None:
    which = _build_which({})

    with pytest.raises(self_debug.DiagnosticError):
        self_debug.determine_markitdown_launcher(which=which, find_spec=lambda _: None)


def test_diagnose_markitdown_reports_status(monkeypatch) -> None:
    def fake_determine(**_: object) -> list[str]:
        return ["uvx", "markitdown"]

    monkeypatch.setattr(self_debug, "determine_markitdown_launcher", fake_determine)

    result = self_debug.diagnose_markitdown()

    assert result.status == "available"
    assert "uvx" in result.details


def test_diagnose_whisper_available() -> None:
    result = self_debug.diagnose_whisper(importer=lambda name: object())

    assert result.status == "available"
    assert "successfully" in result.details


def test_diagnose_whisper_unavailable() -> None:
    def _importer(_: str) -> None:
        raise ModuleNotFoundError("whisper")

    result = self_debug.diagnose_whisper(importer=_importer)

    assert result.status == "unavailable"
    assert self_debug.transcribe.MISSING_WHISPER_MESSAGE == result.recommendation


def test_diagnose_numpy_available() -> None:
    module = type("NumPy", (), {"__version__": "1.26.4"})()

    result = self_debug.diagnose_numpy(importer=lambda _: module)

    assert result.status == "available"
    assert "1.26.4" in result.details


def test_diagnose_numpy_rejects_major_two() -> None:
    module = type("NumPy", (), {"__version__": "2.0.0"})()

    result = self_debug.diagnose_numpy(importer=lambda _: module)

    assert result.status == "unavailable"
    assert "numpy<2" in (result.recommendation or "")


def test_diagnose_numpy_missing() -> None:
    def _importer(_: str) -> None:
        raise ModuleNotFoundError("numpy")

    result = self_debug.diagnose_numpy(importer=_importer)

    assert result.status == "unavailable"
    assert "numpy<2" in (result.recommendation or "")


def test_main_respects_json_flag(monkeypatch, capsys) -> None:
    markitdown_result = self_debug.DiagnosticResult(
        name="markitdown",
        status="available",
        details="uvx markitdown",
        recommendation=None,
    )
    numpy_result = self_debug.DiagnosticResult(
        name="numpy",
        status="available",
        details="1.26.4",
        recommendation=None,
    )
    whisper_result = self_debug.DiagnosticResult(
        name="whisper",
        status="available",
        details="module",
        recommendation=None,
    )
    monkeypatch.setattr(self_debug, "diagnose_numpy", lambda: numpy_result)
    monkeypatch.setattr(self_debug, "diagnose_markitdown", lambda: markitdown_result)
    monkeypatch.setattr(self_debug, "diagnose_whisper", lambda: whisper_result)

    exit_code = self_debug.main(["--json"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "markitdown" in output
    assert "numpy" in output
    assert "whisper" in output
    assert output.strip().startswith("[")


def test_main_handles_error_status(monkeypatch, capsys) -> None:
    markitdown_result = self_debug.DiagnosticResult(
        name="markitdown",
        status="unavailable",
        details="missing command",
        recommendation="install uv",
    )
    numpy_result = self_debug.DiagnosticResult(
        name="numpy",
        status="available",
        details="1.26.4",
        recommendation=None,
    )
    whisper_result = self_debug.DiagnosticResult(
        name="whisper",
        status="available",
        details="module",
        recommendation=None,
    )
    monkeypatch.setattr(self_debug, "diagnose_numpy", lambda: numpy_result)
    monkeypatch.setattr(self_debug, "diagnose_markitdown", lambda: markitdown_result)
    monkeypatch.setattr(self_debug, "diagnose_whisper", lambda: whisper_result)

    exit_code = self_debug.main([])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "missing command" in output
    assert "install uv" in output
