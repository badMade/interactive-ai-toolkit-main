"""Utilities for ensuring FFmpeg is available on the host system."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Protocol


PROJECT_ROOT = Path(__file__).resolve().parent
INSTALL_SCRIPT_RELATIVE_PATH = Path("scripts") / "install_ffmpeg.sh"

FFMPEG_INSTALL_MESSAGE = (
    "ffmpeg is required to transcribe audio. "
    "Run scripts/install_ffmpeg.sh or install it with Homebrew "
    "(`brew install ffmpeg`) or apt (`sudo apt-get install ffmpeg`). "
    "Alternatively, run `pip install imageio[ffmpeg]` to install a "
    "Python-managed binary. On Windows, install FFmpeg from "
    "https://ffmpeg.org/download.html or run `winget install Gyan.FFmpeg`."
)


class CommandRunner(Protocol):
    """Protocol matching the subset of ``subprocess.run``
    used in this module."""

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
    """Raised when FFmpeg cannot be installed or detected."""


def _default_install_script_path() -> Path:
    return PROJECT_ROOT / INSTALL_SCRIPT_RELATIVE_PATH


def _default_runner(
    command: list[str],
    *,
    check: bool,
    text: bool,
    capture_output: bool,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: PLW1510 - allow exceptions to surface
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
    """Execute *command* and return combined stdout/stderr when requested."""

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


def _build_install_command(script_path: Path) -> list[str]:
    if os.name != "posix":
        raise FFmpegInstallationError(
            "Automatic FFmpeg installation is only supported on POSIX systems."
        )
    if not script_path.exists():
        raise FFmpegInstallationError(
            f"FFmpeg install script not found at {script_path}."
        )

    if os.access(script_path, os.X_OK):
        return [str(script_path)]
    return ["bash", str(script_path)]


def install_ffmpeg(
    *,
    script_path: Path | None = None,
    runner: CommandRunner | None = None,
) -> str:
    """Execute the bundled installation script and return its output."""

    resolved_script = script_path or _default_install_script_path()
    command = _build_install_command(resolved_script)
    return _run_command(command, capture_output=True, runner=runner)


def ensure_ffmpeg_available(
    *,
    allow_auto_install: bool = True,
    script_path: Path | None = None,
    runner: CommandRunner | None = None,
) -> str | None:
    """Ensure FFmpeg is available, optionally installing it automatically."""

    def _probe() -> str:
        return _run_command(
            ["ffmpeg", "-version"],
            capture_output=True,
            runner=runner,
        )

    try:
        version_output = _probe()
        return version_output or None
    except FFmpegInstallationError as probe_error:
        if not allow_auto_install:
            raise FFmpegInstallationError(
                FFMPEG_INSTALL_MESSAGE) from probe_error

        try:
            install_output = install_ffmpeg(
                script_path=script_path,
                runner=runner,
            )
        except FFmpegInstallationError as error:
            raise FFmpegInstallationError(
                f"Automatic FFmpeg installation failed.\n{error}"
            ) from error

        try:
            version_output = _probe()
        except FFmpegInstallationError as verify_error:
            raise FFmpegInstallationError(
                "FFmpeg installation completed but verification failed."
            ) from verify_error

        combined = "\n".join(
            piece.strip()
            for piece in (install_output, version_output)
            if piece and piece.strip()
        )
        return combined or None
