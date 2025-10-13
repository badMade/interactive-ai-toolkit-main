from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path

import pytest

import run


def test_is_running_inside_virtualenv_with_env_var(monkeypatch, tmp_path: Path) -> None:
    venv_path = tmp_path / ".venv"
    monkeypatch.setattr(run, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_path))

    assert run.is_running_inside_virtualenv(venv_path) is True


def test_is_running_inside_virtualenv_with_executable(monkeypatch, tmp_path: Path) -> None:
    venv_path = tmp_path / ".venv"
    executable = venv_path / "bin" / "python"
    monkeypatch.setattr(run, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(run.sys, "executable", str(executable))
    monkeypatch.setattr(run.sys, "prefix", str(venv_path))

    assert run.is_running_inside_virtualenv(venv_path) is True


def test_ensure_virtual_environment_reexecutes_when_outside(monkeypatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    ensure_script = scripts_dir / "ensure_requirements.py"
    ensure_script.write_text("print('noop')\n")

    venv_path = tmp_path / ".venv"
    bin_dir = venv_path / "bin"
    bin_dir.mkdir(parents=True)
    venv_python = bin_dir / "python"
    venv_python.write_text("#!/usr/bin/env python3\n")
    (venv_path / "pyvenv.cfg").write_text("home = /usr/bin/python\n")

    monkeypatch.setattr(run, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run, "is_running_inside_virtualenv", lambda _: False)
    monkeypatch.setattr(run, "get_virtualenv_python_path", lambda _: venv_python)
    monkeypatch.setattr(run.sys, "argv", ["run.py", "--flag"])
    monkeypatch.setattr(run.sys, "executable", "/usr/bin/python3")

    executed_commands: list[dict[str, object]] = []

    def fake_run(
        command: list[str] | tuple[str, ...],
        *,
        check: bool,
        text: bool,
        capture_output: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        executed_commands.append(
            {
                "command": list(command),
                "check": check,
                "capture_output": capture_output,
                "env": env,
            }
        )
        if check and command == ["/usr/bin/python3", str(ensure_script)]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[0] == str(venv_python):
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        run.ensure_virtual_environment()

    assert exc_info.value.code == 0
    assert executed_commands[0]["command"] == ["/usr/bin/python3", str(ensure_script)]
    assert executed_commands[0]["check"] is True
    assert executed_commands[1]["command"][0] == str(venv_python)
    assert executed_commands[1]["command"][1] == str(Path(run.__file__).resolve())
    assert executed_commands[1]["command"][2:] == ["--flag"]
    env = executed_commands[1]["env"]
    assert isinstance(env, dict)
    assert env["VIRTUAL_ENV"] == str(venv_path)
    assert env["PATH"].split(os.pathsep)[0] == str(bin_dir)


def test_ensure_virtual_environment_noop_inside(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run, "is_running_inside_virtualenv", lambda _: True)

    def fail_run(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("subprocess.run should not be invoked")

    monkeypatch.setattr(run.subprocess, "run", fail_run)

    run.ensure_virtual_environment()


def test_main_reports_missing_whisper(monkeypatch, capsys) -> None:
    monkeypatch.setattr(run, "ensure_virtual_environment", lambda: None)

    message = (
        "OpenAI Whisper is not installed. Install it with 'pip install openai-whisper' "
        "or run setup_env.py to configure the environment."
    )

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "transcribe":
            cause = ModuleNotFoundError("No module named 'whisper'", name="whisper")
            raise ModuleNotFoundError(message) from cause
        return real_import_module(name)

    monkeypatch.setattr(run.importlib, "import_module", fake_import_module)

    with pytest.raises(SystemExit) as exc_info:
        run.main()

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == message
    assert "Traceback" not in captured.err
