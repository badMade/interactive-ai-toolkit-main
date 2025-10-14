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


def test_main_respects_json_flag(monkeypatch, capsys) -> None:
    result = self_debug.DiagnosticResult(
        name="markitdown",
        status="available",
        details="uvx markitdown",
        recommendation=None,
    )
    monkeypatch.setattr(self_debug, "diagnose_markitdown", lambda: result)

    exit_code = self_debug.main(["--json"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "markitdown" in output
    assert output.strip().startswith("[")


def test_main_handles_error_status(monkeypatch, capsys) -> None:
    result = self_debug.DiagnosticResult(
        name="markitdown",
        status="unavailable",
        details="missing command",
        recommendation="install uv",
    )
    monkeypatch.setattr(self_debug, "diagnose_markitdown", lambda: result)

    exit_code = self_debug.main([])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "missing command" in output
    assert "install uv" in output
