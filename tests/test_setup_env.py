from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path
import subprocess

import pytest

import setup_env


class FakeRunner:
    def __init__(self, responses: list[tuple[str, str]] | None = None, fail_on: int | None = None):
        self.responses = responses or [("", "")] * 10
        self.fail_on = fail_on
        self.calls: list[tuple[list[str], bool]] = []

    def __call__(self, command, *, check, text, capture_output):
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise subprocess.CalledProcessError(returncode=1, cmd=command)
        self.calls.append((command, capture_output))
        stdout, stderr = self.responses.pop(0)
        return SimpleNamespace(stdout=stdout, stderr=stderr)


@pytest.fixture()
def requirements_file(tmp_path: Path) -> Path:
    path = tmp_path / "requirements.txt"
    path.write_text("numpy\n")
    return path


def test_ensure_requirements_file_missing(tmp_path: Path) -> None:
    with pytest.raises(setup_env.SetupError):
        setup_env.ensure_requirements_file(tmp_path / "requirements.txt")


def test_ensure_requirements_file_success(requirements_file: Path) -> None:
    resolved = setup_env.ensure_requirements_file(requirements_file)
    assert resolved == requirements_file


def test_create_virtualenv_skips_when_existing(tmp_path: Path) -> None:
    venv_path = tmp_path / ".venv"
    venv_path.mkdir()
    (venv_path / "pyvenv.cfg").write_text("")
    runner = FakeRunner()
    setup_env.create_virtualenv(venv_path, runner=runner)
    assert runner.calls == []


def test_create_virtualenv_invokes_runner(tmp_path: Path) -> None:
    venv_path = tmp_path / ".venv"
    runner = FakeRunner()
    setup_env.create_virtualenv(venv_path, runner=runner)
    assert runner.calls[0][0][0:3] == [sys.executable, "-m", "venv"]


def test_get_venv_python_path_posix(tmp_path: Path) -> None:
    path = setup_env.get_venv_python_path(tmp_path / ".venv")
    assert path.as_posix().endswith(".venv/bin/python")


def test_get_venv_python_path_windows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(setup_env.os, "name", "nt", raising=False)
    path = setup_env.get_venv_python_path(tmp_path / ".venv")
    assert path.as_posix().endswith(".venv/Scripts/python.exe")


def test_install_requirements_runs_expected_commands(requirements_file: Path) -> None:
    runner = FakeRunner()
    setup_env.install_requirements(Path("/tmp/python"), requirements_file, runner=runner)
    assert runner.calls[0][0][:4] == ["/tmp/python", "-m", "pip", "install"]
    assert runner.calls[1][0][:5] == ["/tmp/python", "-m", "pip", "install", "-r"]


def test_verify_installation_success(tmp_path: Path) -> None:
    runner = FakeRunner(
        responses=[
            ("Python 3.11.0", ""),
            ("pip 23.0", ""),
            ("1.0.0", ""),
            ("2.0.0", ""),
            ("0.12.1", ""),
            ("202311", ""),
        ]
    )
    setup_env.verify_installation(
        tmp_path / ".venv/bin/python",
        packages=("torch", "transformers", "soundfile", "whisper"),
        runner=runner,
    )
    assert len(runner.calls) == 6


def test_verify_installation_failure(tmp_path: Path) -> None:
    runner = FakeRunner(
        responses=[
            ("Python 3.11.0", ""),
            ("pip 23.0", ""),
            ("1.0.0", ""),
        ],
        fail_on=3,
    )
    with pytest.raises(setup_env.SetupError):
        setup_env.verify_installation(
            tmp_path / ".venv/bin/python",
            packages=("torch", "transformers"),
            runner=runner,
        )
