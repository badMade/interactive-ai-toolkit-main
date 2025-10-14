"""Test suite for the setup_env module.

This module contains unit tests for environment setup functionality including
requirements validation, Python version checks, virtualenv creation, dependency
installation, FFmpeg availability checks, and installation verification.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from collections import deque, namedtuple
from pathlib import Path

import pytest

import ffmpeg_support
import setup_env
from shared_messages import MISSING_WHISPER_MESSAGE


class FakeRunner:
    """Mock subprocess runner for testing command execution.

    This class simulates subprocess.run() behavior with configurable responses,
    optional failure injection, or explicit side effects.
    """

    def __init__(
        self,
        responses: list[tuple[str, str]] | None = None,
        fail_on: int | None = None,
        side_effects: list[tuple[str, str] | BaseException] | None = None,
    ):
        if side_effects is not None:
            self.side_effects = deque(side_effects)
            self.responses = None
        else:
            self.side_effects = None
            self.responses = deque(responses or [("", "")] * 10)
        self.fail_on = fail_on
        self.calls: list[tuple[list[str], bool]] = []
        self.environments: list[dict[str, str] | None] = []

    def __call__(
        self,
        command,
        *,
        check,
        text,
        capture_output,
        env=None,
    ):
        """Execute a fake subprocess command with configurable behavior."""
        if self.side_effects is not None:
            if not self.side_effects:
                raise AssertionError("No more side effects configured")
            effect = self.side_effects.popleft()
            self.calls.append((command, capture_output))
            if isinstance(effect, BaseException):
                raise effect
            stdout, stderr = effect
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=stdout,
                stderr=stderr,
            )

        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise subprocess.CalledProcessError(returncode=1, cmd=command)
        self.calls.append((command, capture_output))
        self.environments.append(env)
        stdout, stderr = self.responses.popleft()
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=stdout,
            stderr=stderr,
        )


@pytest.fixture(name="requirements_file")
def _requirements_file(tmp_path: Path) -> Path:
    """Create a temporary requirements.txt file for testing."""
    path = tmp_path / "requirements.txt"
    path.write_text("numpy<2\n")
    return path


def test_ensure_requirements_file_missing(tmp_path: Path) -> None:
    """Test that ensure_requirements_file raises
    SetupError when file is missing."""
    with pytest.raises(setup_env.SetupError):
        setup_env.ensure_requirements_file(tmp_path / "requirements.txt")


def test_ensure_requirements_file_success(requirements_file: Path) -> None:
    """Test that ensure_requirements_files
    uccessfully resolves an existing file."""
    resolved = setup_env.ensure_requirements_file(requirements_file)
    assert resolved == requirements_file


VersionInfo = namedtuple("VersionInfo", ["major", "minor", "micro"])


def test_ensure_supported_python_raises_for_older_runtime(monkeypatch) -> None:
    """Test that ensure_supported_python raises
    SetupError for Python < 3.12."""
    monkeypatch.setattr(
        setup_env.sys,
        "version_info",
        VersionInfo(3, 9, 18),
        raising=False,
    )
    with pytest.raises(setup_env.SetupError) as error:
        setup_env.ensure_supported_python()
    assert "Python 3.12 or newer is required" in str(error.value)


def test_ensure_supported_python_accepts_supported_runtime(
    monkeypatch,
) -> None:
    """Test that ensure_supported_python accepts Python >= 3.12."""
    monkeypatch.setattr(
        setup_env.sys,
        "version_info",
        VersionInfo(3, 12, 5),
        raising=False,
    )
    setup_env.ensure_supported_python()


def test_create_virtualenv_skips_when_existing(tmp_path: Path) -> None:
    """Test that create_virtualenv skips creation when
    virtualenv already exists."""
    venv_path = tmp_path / ".venv"
    venv_path.mkdir()
    (venv_path / "pyvenv.cfg").write_text("version = 3.12.0\n")
    runner = FakeRunner()
    setup_env.create_virtualenv(venv_path, runner=runner)
    assert not runner.calls, "Runner should not be called when venv exists"


def test_create_virtualenv_invokes_runner(tmp_path: Path) -> None:
    """Test that create_virtualenv invokes the runner
    with correct venv command."""
    venv_path = tmp_path / ".venv"
    runner = FakeRunner()
    setup_env.create_virtualenv(venv_path, runner=runner)
    assert runner.calls[0][0][0:3] == [sys.executable, "-m", "venv"]


def test_get_venv_python_path_posix(tmp_path: Path) -> None:
    """Test that get_venv_python_path returns correct path
    for POSIX systems."""
    path = setup_env.get_venv_python_path(tmp_path / ".venv")
    assert path.as_posix().endswith(".venv/bin/python")


def test_get_venv_python_path_windows(monkeypatch, tmp_path: Path) -> None:
    """Test that get_venv_python_path returns correct path
    for Windows systems."""
    monkeypatch.setattr(
        setup_env.os,
        "name",
        "nt",
        raising=False,
    )
    path = setup_env.get_venv_python_path(tmp_path / ".venv")
    assert path.as_posix().endswith(".venv/Scripts/python.exe")


def test_install_requirements_runs_expected_commands(
    requirements_file: Path,
) -> None:
    """Test that install_requirements runs pip install and
    pip install -r commands."""
    runner = FakeRunner()
    setup_env.install_requirements(
        Path("/tmp/python"), requirements_file, runner=runner
    )
    upgrade_command, upgrade_capture = runner.calls[0]
    assert upgrade_command[:4] == ["/tmp/python", "-m", "pip", "install"]
    assert "--upgrade" in upgrade_command
    assert "pip" in upgrade_command
    reinstall_env = runner.environments[1]
    assert runner.environments[0] is not None
    assert runner.environments[0].get("PIP_REQUIRE_VIRTUALENV") == "1"
    reinstall_command, reinstall_capture = runner.calls[1]
    assert reinstall_command[:4] == ["/tmp/python", "-m", "pip", "install"]
    assert "--upgrade" in reinstall_command
    assert "--force-reinstall" in reinstall_command
    assert "-r" in reinstall_command
    assert reinstall_command[-1] == str(requirements_file)
    assert reinstall_env is not None
    assert reinstall_env.get("PIP_REQUIRE_VIRTUALENV") == "1"


def test_ensure_ffmpeg_available_success() -> None:
    """Test that ensure_ffmpeg_available succeeds when FFmpeg is installed."""
    runner = FakeRunner(responses=[("ffmpeg version 6.1", "")])
    setup_env.ensure_ffmpeg_available(runner=runner)
    assert runner.calls == [(["ffmpeg", "-version"], True)]


def test_ensure_ffmpeg_available_windows_guidance(monkeypatch) -> None:
    """Test that Windows users receive manual installation guidance."""
    runner = FakeRunner(
        side_effects=[
            subprocess.CalledProcessError(
                returncode=1,
                cmd=["ffmpeg", "-version"],
            )
        ]
    )
    monkeypatch.setattr(setup_env.os, "name", "nt", raising=False)
    monkeypatch.setattr(setup_env.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(ffmpeg_support.os, "name", "nt", raising=False)

    with pytest.raises(setup_env.SetupError) as error:
        setup_env.ensure_ffmpeg_available(runner=runner)

    assert "winget install" in str(error.value)


def test_ensure_ffmpeg_available_triggers_script(monkeypatch) -> None:
    """Test automatic installer invocation when FFmpeg is initially missing."""
    script_effects = [
        subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg", "-version"],
        ),
        ("installation output", ""),
        ("ffmpeg version 6.1", ""),
    ]
    runner = FakeRunner(side_effects=script_effects)

    monkeypatch.setattr(setup_env.os, "name", "posix", raising=False)
    monkeypatch.setattr(setup_env.sys, "platform", "linux", raising=False)
    monkeypatch.setattr(ffmpeg_support.os, "name", "posix", raising=False)

    setup_env.ensure_ffmpeg_available(runner=runner)

    assert len(runner.calls) == 3
    assert runner.calls[0][0] == ["ffmpeg", "-version"]
    assert runner.calls[1][0][0].endswith("scripts/install_ffmpeg.sh")
    assert runner.calls[2][0] == ["ffmpeg", "-version"]


def test_ensure_ffmpeg_available_script_failure(monkeypatch) -> None:
    """Test that script failure propagates descriptive error details."""
    runner = FakeRunner(
        side_effects=[
            subprocess.CalledProcessError(
                returncode=1,
                cmd=["ffmpeg", "-version"],
            ),
            subprocess.CalledProcessError(
                returncode=1,
                cmd=["scripts/install_ffmpeg.sh"],
                stderr="installation failed",
            ),
        ]
    )

    monkeypatch.setattr(setup_env.os, "name", "posix", raising=False)
    monkeypatch.setattr(setup_env.sys, "platform", "linux", raising=False)
    monkeypatch.setattr(ffmpeg_support.os, "name", "posix", raising=False)

    with pytest.raises(setup_env.SetupError) as error:
        setup_env.ensure_ffmpeg_available(runner=runner)

    message = str(error.value)
    assert "Automatic FFmpeg installation failed" in message
    assert "installation failed" in message


def test_verify_installation_success(tmp_path: Path) -> None:
    """Test that verify_installation succeeds with all packages installed."""
    packages = setup_env.DEFAULT_VALIDATION_PACKAGES
    package_responses = [
        ("1.26.4", "") if name == "numpy" else (f"{name} 1.0.0", "")
        for name in packages
    ]
    runner = FakeRunner(
        responses=[
            ("Python 3.11.0", ""),
            ("pip 23.0", ""),
            *package_responses,
        ]
    )
    setup_env.verify_installation(
        tmp_path / ".venv/bin/python",
        packages=packages,
        runner=runner,
    )
    assert len(runner.calls) == 2 + len(packages)


def test_verify_installation_rejects_numpy_two(tmp_path: Path) -> None:
    """Setup should fail fast when NumPy 2.x is detected."""

    runner = FakeRunner(
        responses=[
            ("Python 3.11.0", ""),
            ("pip 23.0", ""),
            ("2.0.0", ""),
        ]
    )

    with pytest.raises(setup_env.SetupError) as error:
        setup_env.verify_installation(
            tmp_path / ".venv/bin/python",
            packages=("numpy",),
            runner=runner,
        )

    assert "numpy<2" in str(error.value)


def test_verify_installation_failure(tmp_path: Path) -> None:
    """Test that verify_installation raises
    SetupError when package check fails."""
    packages = ("torch", "transformers")
    runner = FakeRunner(
        responses=[
            ("Python 3.11.0", ""),
            ("pip 23.0", ""),
            *[(f"{name} 1.0.0", "") for name in packages],
        ],
        fail_on=3,
    )
    with pytest.raises(setup_env.SetupError):
        setup_env.verify_installation(
            tmp_path / ".venv/bin/python",
            packages=packages,
            runner=runner,
        )


def test_verify_installation_missing_whisper(tmp_path: Path) -> None:
    """Verify that missing Whisper yields user-friendly setup guidance."""

    packages = ("torch", "whisper")
    runner = FakeRunner(
        responses=[
            ("Python 3.11.0", ""),
            ("pip 23.0", ""),
            ("torch 1.0.0", ""),
        ],
        fail_on=3,
    )

    with pytest.raises(setup_env.SetupError) as error:
        setup_env.verify_installation(
            tmp_path / ".venv/bin/python",
            packages=packages,
            runner=runner,
        )

    assert MISSING_WHISPER_MESSAGE in str(error.value)


def test_main_writes_rotating_log_with_packages(monkeypatch, tmp_path: Path) -> None:
    """Ensure main() writes package data to the configured log file."""

    log_path = tmp_path / "logs" / "test_setup_env.log"
    monkeypatch.setenv(setup_env.LOG_PATH_ENV_VAR, str(log_path))

    responses = deque(
        [
            "Python 3.11.0",
            "pip 23.0",
            *(
                "1.26.4"
                if name == "numpy"
                else f"{name}-version"
                for name in setup_env.DEFAULT_VALIDATION_PACKAGES
            ),
        ]
    )

    def fake_run_command(command, *, capture_output=False, env=None, runner=None):
        if capture_output:
            if not responses:
                raise AssertionError("Unexpected command invocation")
            return responses.popleft()
        return ""

    monkeypatch.setattr(setup_env, "run_command", fake_run_command)
    monkeypatch.setattr(setup_env, "ensure_ffmpeg_available", lambda **_: None)
    monkeypatch.setattr(setup_env, "create_virtualenv", lambda *_, **__: None)
    monkeypatch.setattr(setup_env, "install_requirements", lambda *_, **__: None)
    monkeypatch.setattr(setup_env, "read_required_packages", lambda *_: {})
    monkeypatch.setattr(setup_env, "write_setup_log", lambda *_, **__: None)

    exit_code = setup_env.main()
    assert exit_code == 0

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        handler.flush()

    assert log_path.exists(), "Expected log file to be created"
    log_contents = log_path.read_text(encoding="utf-8")

    for package in setup_env.DEFAULT_VALIDATION_PACKAGES:
        assert package in log_contents
    assert "INSTALLATION_SUMMARY" in log_contents

    for handler in list(root_logger.handlers):
        handler.close()
        root_logger.removeHandler(handler)
