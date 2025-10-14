"""Utility helpers to detect and install FFmpeg when missing."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol


class CommandRunner(Protocol):
    """Protocol representing the subset of ``subprocess.run`` we rely upon."""

    def __call__(
        self,
        command: list[str],
        *,
        check: bool,
        text: bool,
        capture_output: bool,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        ...


class FFmpegInstallationError(RuntimeError):
    """Raised when FFmpeg cannot be located or installed automatically."""


_PROJECT_ROOT = Path(__file__).resolve().parent
_INSTALL_SCRIPT = _PROJECT_ROOT / "scripts" / "install_ffmpeg.sh"


def _default_runner(
    command: list[str],
    *,
    check: bool,
    text: bool,
    capture_output: bool,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke ``subprocess.run`` with the expected keyword arguments."""

    return subprocess.run(  # noqa: PLW1510 - intended to propagate exceptions
        command,
        check=check,
        text=text,
        capture_output=capture_output,
        env=env,
    )


def _run_command(
    command: list[str],
    *,
    capture_output: bool,
    runner: CommandRunner | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Execute *command* and return combined stdout/stderr when captured."""

    active_runner = runner or _default_runner
    try:
        result = active_runner(
            command,
            check=True,
            text=True,
            capture_output=capture_output,
            env=env,
        )
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        missing = command[0] if command else "command"
        raise FFmpegInstallationError(
            f"Required command not found: {missing}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stdout = getattr(exc, "stdout", "") or getattr(exc, "output", "")
        stderr = getattr(exc, "stderr", "")
        details = f"{stdout}{stderr}".strip()
        message = f"Command failed: {' '.join(command)}"
        if details:
            message = f"{message}\n{details}"
        raise FFmpegInstallationError(message) from exc

    if capture_output:
        output = (result.stdout or "") + (result.stderr or "")
        return output.strip()
    return ""


def _windows_guidance() -> str:
    """Return installation guidance tailored for Windows environments."""

    return (
        "FFmpeg is required but was not detected. Install it from "
        "https://ffmpeg.org/download.html or run `winget install Gyan.FFmpeg`."
    )


def _posix_guidance() -> str:
    """Return a concise description of the supported POSIX installation path."""

    return (
        "FFmpeg is required but automatic installation is only supported on "
        "POSIX-compliant systems. Install FFmpeg manually from "
        "https://ffmpeg.org/download.html."
    )


def ensure_ffmpeg_available(*, runner: CommandRunner | None = None) -> None:
    """Ensure that FFmpeg is installed, invoking the bundled script if needed."""

    try:
        version_output = _run_command(
            ["ffmpeg", "-version"],
            capture_output=True,
            runner=runner,
        )
        if version_output:
            logging.info("FFmpeg detected: %s", version_output.splitlines()[0])
        return
    except FFmpegInstallationError as probe_error:
        platform = sys.platform.lower()
        is_windows = os.name == "nt" or platform.startswith("win")
        if is_windows:
            raise FFmpegInstallationError(_windows_guidance()) from probe_error

        if os.name != "posix":
            raise FFmpegInstallationError(_posix_guidance()) from probe_error

        if not _INSTALL_SCRIPT.is_file():
            raise FFmpegInstallationError(
                "FFmpeg is required but the installer script is missing."
            ) from probe_error

        try:
            _run_command([str(_INSTALL_SCRIPT)], capture_output=False, runner=runner)
        except FFmpegInstallationError as install_error:
            raise FFmpegInstallationError(
                "FFmpeg is required but automatic installation failed.\n"
                f"{install_error}"
            ) from install_error

        version_output = _run_command(
            ["ffmpeg", "-version"],
            capture_output=True,
            runner=runner,
        )
        if version_output:
            logging.info("FFmpeg installed: %s", version_output.splitlines()[0])
            return

        raise FFmpegInstallationError(
            "FFmpeg installation did not succeed even after running the installer."
        )
